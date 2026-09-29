"""
Fused Schur-complement matvec kernel from FlashSinkhorn (ICML 2026).
https://github.com/ot-triton-lab/flash-sinkhorn/
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class FlashSinkhornFusedSchurMatvec(TritonPTXKernel):
    """Compute a fused symmetric Sinkhorn update for source and target potentials."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "block_m": (16, 32, 64, 128, 256),
        "block_n": (16, 32, 64, 128, 256),
        "block_k": (16, 32, 64, 128, 256),
        "num_warps": (4, 8),
        "num_stages": (2, 3, 4),
    }

    def __init__(self, *, ptx=None):
        self.source_size = 1024
        self.target_size = 1024
        self.feature_dim = 64
        self.eps = 1.0
        self.alpha = 0.5
        self.damping_f = 1.0
        self.damping_g = 1.0
        self.coord_scale = 2.0
        self.half_cost_scale = 1.0
        self.lambda_x = 1.0
        self.lambda_y = 0.0
        self.block_m = 64
        self.block_n = 64
        self.block_k = 64
        self.num_warps = 4
        self.num_stages = 2
        self.use_exp2 = True
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,  # Source coordinates: [n, d] (NOT pre-scaled!)
        y_ptr,  # Target coordinates: [m, d] (NOT pre-scaled!)
        f_hat_ptr,  # Current shifted f potential: [n]
        g_hat_ptr,  # Current shifted g potential: [m]
        log_a_ptr,  # Log source weights: [n]
        log_b_ptr,  # Log target weights: [m]
        f_out_ptr,  # Output shifted f potential: [n]
        g_out_ptr,  # Output shifted g potential: [m]
        # OTDD label cost parameters (optional)
        label_x_ptr,  # int32 labels for x: [n] or dummy if not used
        label_y_ptr,  # int32 labels for y: [m] or dummy if not used
        W_ptr,  # Label cost matrix: [V, V] flattened or dummy
        n: tl.constexpr,
        m: tl.constexpr,
        V: tl.constexpr,  # Number of classes (1 if no label cost)
        stride_x0: tl.constexpr,
        stride_x1: tl.constexpr,
        stride_y0: tl.constexpr,
        stride_y1: tl.constexpr,
        eps: tl.constexpr,  # Regularization parameter
        alpha: tl.constexpr,  # Symmetric averaging weight
        damping_f: tl.constexpr,  # Unbalanced OT damping for f
        damping_g: tl.constexpr,  # Unbalanced OT damping for g
        coord_scale: tl.constexpr,  # 2 * cost_scale
        half_cost_scale: tl.constexpr,  # cost_scale
        lambda_x: tl.constexpr,  # Weight for Euclidean cost
        lambda_y: tl.constexpr,  # Weight for label cost
        CACHE_KEY_N: tl.constexpr,  # Bucketed n for autotune cache
        CACHE_KEY_M: tl.constexpr,  # Bucketed m for autotune cache
        D: tl.constexpr,
        ALLOW_TF32: tl.constexpr,
        DTYPE_ID: tl.constexpr,  # For autotune cache consistency (harmonized with alternating)
        USE_EXP2: tl.constexpr,
        USE_LABEL_COST: tl.constexpr,  # Whether to use label cost (compile-time)
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused symmetric Sinkhorn step: computes both f and g updates in one kernel.

        Grid structure: (blocks_f + blocks_g,) where blocks_f = cdiv(n, BLOCK_M)
        - Programs 0..blocks_f-1: compute f-update (LSE over m dimension)
        - Programs blocks_f..end: compute g-update (LSE over n dimension)

        This matches the GeomLoss kernel structure for maximum efficiency.

        KEY OPTIMIZATION: Uses x, y directly (no Q, K pre-allocation needed).
        The scale factor (2*cost_scale) is passed as coord_scale and applied in-kernel.

        Bias computation is done inline:
        - f-update: u = ĝ/ε + log(b)
        - g-update: v = f̂/ε + log(a)
        """
        pid = tl.program_id(0)
        inv_eps = 1.0 / eps
        blocks_f = tl.cdiv(n, BLOCK_M)

        # Constants for exp2/log2 optimization
        log2e = 1.4426950408889634
        ln2 = 0.6931471805599453
        # =========================================================================
        # F-UPDATE: first blocks_f programs
        # =========================================================================
        if pid < blocks_f:
            offs_i = pid * BLOCK_M + tl.arange(0, BLOCK_M)
            # Load current f_hat for symmetric averaging
            f_old = tl.load(f_hat_ptr + offs_i).to(tl.float32)

            # Load labels for this block if using label cost
            if USE_LABEL_COST:
                label_i = tl.load(label_x_ptr + offs_i).to(tl.int32)

            # Online LSE accumulators
            m_i = tl.full([BLOCK_M], -float("inf"), tl.float32)
            s_i = tl.zeros([BLOCK_M], tl.float32)

            # Iterate over all j (target points)
            for j0 in range(0, m, BLOCK_N):
                j0 = tl.multiple_of(j0, BLOCK_N)
                offs_j = j0 + tl.arange(0, BLOCK_N)
                # Load ĝ and log(b) to compute bias: u = ĝ/ε + log(b)
                g_hat = tl.load(g_hat_ptr + offs_j, eviction_policy="evict_first").to(
                    tl.float32
                )
                log_b = tl.load(log_b_ptr + offs_j, eviction_policy="evict_first").to(
                    tl.float32
                )

                # Compute x @ y.T via tiled matmul (NOT pre-scaled Q @ K!)
                dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
                for k0 in range(0, D, BLOCK_K):
                    k0 = tl.multiple_of(k0, BLOCK_K)
                    offs_k = k0 + tl.arange(0, BLOCK_K)
                    x_block = tl.load(
                        x_ptr
                        + offs_i[:, None] * stride_x0
                        + offs_k[None, :] * stride_x1,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    y_block = tl.load(
                        y_ptr
                        + offs_j[None, :] * stride_y0
                        + offs_k[:, None] * stride_y1,
                        eviction_policy="evict_first",
                    ).to(tl.float32)
                    dot += tl.dot(x_block, y_block, allow_tf32=ALLOW_TF32)

                # Compute label cost if enabled
                # The cost term 2*cs*x·y contributes POSITIVELY to the exponent (lower cost = higher prob)
                # Label cost must be SUBTRACTED to increase cost for mismatched labels
                if USE_LABEL_COST:
                    label_j = tl.load(
                        label_y_ptr + offs_j, eviction_policy="evict_first"
                    ).to(tl.int32)
                    # Compute flattened indices into W: W[label_i, label_j]
                    w_idx = label_i[:, None] * V + label_j[None, :]
                    # Gather from W (label cost matrix)
                    w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
                    # Label cost is SUBTRACTED (divided by eps) from the shifted dot product
                    # Combined: lambda_x * (2*cs*dot) - lambda_y * (cs * w_cost)
                    # Scale factors: coord_scale = 2*cost_scale, half_cost_scale = cost_scale
                    effective_dot = (
                        lambda_x * dot * coord_scale
                        - lambda_y * half_cost_scale * w_cost
                    )
                else:
                    effective_dot = dot * coord_scale

                # Form logits: effective_dot/ε + ĝ/ε + log(b)
                if USE_EXP2:
                    inv_eps_log2 = inv_eps * log2e
                    bias = g_hat * inv_eps_log2 + log_b * log2e
                    vals = tl.fma(effective_dot, inv_eps * log2e, bias[None, :])
                else:
                    bias = g_hat * inv_eps + log_b
                    vals = effective_dot * inv_eps + bias[None, :]
                # Online LSE update
                block_max = tl.max(vals, axis=1)
                new_m = tl.maximum(m_i, block_max)
                if USE_EXP2:
                    rescale = tl.exp2(m_i - new_m)
                    w = tl.exp2(vals - new_m[:, None])
                else:
                    rescale = tl.exp(m_i - new_m)
                    w = tl.exp(vals - new_m[:, None])
                s_i = s_i * rescale + tl.sum(w, axis=1)
                m_i = new_m

            # Final LSE and output
            s_i_safe = tl.maximum(s_i, 1e-40)
            if USE_EXP2:
                lse = (m_i + tl.log2(s_i_safe)) * ln2
            else:
                lse = m_i + tl.log(s_i_safe)
            f_cand = -eps * damping_f * lse
            f_new = (1.0 - alpha) * f_old + alpha * f_cand
            tl.store(f_out_ptr + offs_i, f_new)
            return

        # =========================================================================
        # G-UPDATE: remaining programs (pid >= blocks_f)
        # =========================================================================
        pid_g = pid - blocks_f
        offs_j = pid_g * BLOCK_N + tl.arange(0, BLOCK_N)
        # Load current g_hat for symmetric averaging
        g_old = tl.load(g_hat_ptr + offs_j).to(tl.float32)

        # Load labels for this block if using label cost
        if USE_LABEL_COST:
            label_j = tl.load(label_y_ptr + offs_j).to(tl.int32)

        # Online LSE accumulators
        m_j = tl.full([BLOCK_N], -float("inf"), tl.float32)
        s_j = tl.zeros([BLOCK_N], tl.float32)

        # Iterate over all i (source points)
        for i0 in range(0, n, BLOCK_M):
            i0 = tl.multiple_of(i0, BLOCK_M)
            offs_i = i0 + tl.arange(0, BLOCK_M)
            # Load f̂ and log(a) to compute bias: v = f̂/ε + log(a)
            f_hat = tl.load(f_hat_ptr + offs_i, eviction_policy="evict_first").to(
                tl.float32
            )
            log_a = tl.load(log_a_ptr + offs_i, eviction_policy="evict_first").to(
                tl.float32
            )

            # Compute x @ y.T via tiled matmul (NOT pre-scaled Q @ K!)
            dot = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)
            for k0 in range(0, D, BLOCK_K):
                k0 = tl.multiple_of(k0, BLOCK_K)
                offs_k = k0 + tl.arange(0, BLOCK_K)
                x_block = tl.load(
                    x_ptr + offs_i[:, None] * stride_x0 + offs_k[None, :] * stride_x1,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                y_block = tl.load(
                    y_ptr + offs_j[None, :] * stride_y0 + offs_k[:, None] * stride_y1,
                    eviction_policy="evict_first",
                ).to(tl.float32)
                dot += tl.dot(x_block, y_block, allow_tf32=ALLOW_TF32)

            # Compute label cost if enabled (same formula as f-update)
            if USE_LABEL_COST:
                label_i = tl.load(
                    label_x_ptr + offs_i, eviction_policy="evict_first"
                ).to(tl.int32)
                # Compute flattened indices into W: W[label_i, label_j]
                w_idx = label_i[:, None] * V + label_j[None, :]
                # Gather from W (label cost matrix)
                w_cost = tl.load(W_ptr + w_idx).to(tl.float32)
                # Combined: lambda_x * (2*cs*dot) - lambda_y * (cs * w_cost)
                effective_dot = (
                    lambda_x * dot * coord_scale - lambda_y * half_cost_scale * w_cost
                )
            else:
                effective_dot = dot * coord_scale

            # Form logits: effective_dot/ε + f̂/ε + log(a) - reduce over i dimension
            # vals[i, j] = effective_dot_ij/ε + f̂_i/ε + log(a_i)
            if USE_EXP2:
                inv_eps_log2 = inv_eps * log2e
                bias = f_hat * inv_eps_log2 + log_a * log2e
                vals = tl.fma(effective_dot, inv_eps * log2e, bias[:, None])
            else:
                bias = f_hat * inv_eps + log_a
                vals = effective_dot * inv_eps + bias[:, None]
            # Online LSE update (reduce over axis=0, i.e., over i dimension)
            block_max = tl.max(vals, axis=0)
            new_m = tl.maximum(m_j, block_max)
            if USE_EXP2:
                rescale = tl.exp2(m_j - new_m)
                w = tl.exp2(vals - new_m[None, :])
            else:
                rescale = tl.exp(m_j - new_m)
                w = tl.exp(vals - new_m[None, :])
            s_j = s_j * rescale + tl.sum(w, axis=0)
            m_j = new_m

        # Final LSE and output
        s_j_safe = tl.maximum(s_j, 1e-40)
        if USE_EXP2:
            lse = (m_j + tl.log2(s_j_safe)) * ln2
        else:
            lse = m_j + tl.log(s_j_safe)
        g_cand = -eps * damping_g * lse
        g_new = (1.0 - alpha) * g_old + alpha * g_cand
        tl.store(g_out_ptr + offs_j, g_new)

    def get_random_input(self, fixed: bool = False):
        del fixed
        x = torch.randn(
            (self.source_size, self.feature_dim), device="cuda", dtype=torch.float16
        )
        y = torch.randn(
            (self.target_size, self.feature_dim), device="cuda", dtype=torch.float16
        )
        f = torch.randn(self.source_size, device="cuda", dtype=torch.float32)
        g = torch.randn(self.target_size, device="cuda", dtype=torch.float32)
        log_a = torch.full_like(f, -torch.log(torch.tensor(float(self.source_size))))
        log_b = torch.full_like(g, -torch.log(torch.tensor(float(self.target_size))))
        return x, y, f, g, log_a, log_b

    def get_shape_information(self) -> str:
        return (
            f"- x_ptr: float16 tensor with shape ({self.source_size}, "
            f"{self.feature_dim})\n"
            f"- y_ptr: float16 tensor with shape ({self.target_size}, "
            f"{self.feature_dim})\n"
            f"- f_hat_ptr, log_a_ptr, f_out_ptr: float32 tensors with shape "
            f"({self.source_size},)\n"
            f"- g_hat_ptr, log_b_ptr, g_out_ptr: float32 tensors with shape "
            f"({self.target_size},)"
        )

    def forward_triton(self, inputs, ptx=False):
        x, y, f_hat, g_hat, log_a, log_b = inputs
        f_out = torch.empty_like(f_hat)
        g_out = torch.empty_like(g_hat)
        dummy_labels = torch.empty(1, device=x.device, dtype=torch.int32)
        dummy_cost = torch.empty(1, device=x.device, dtype=torch.float32)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs(num_stages=self.num_stages)
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        grid = lambda meta: (
            triton.cdiv(self.source_size, self.block_m)
            + triton.cdiv(self.target_size, self.block_n),
        )
        kernel = launch_kernel[grid](
            x,
            y,
            f_hat,
            g_hat,
            log_a,
            log_b,
            f_out,
            g_out,
            dummy_labels,
            dummy_labels,
            dummy_cost,
            n=self.source_size,
            m=self.target_size,
            V=1,
            stride_x0=x.stride(0),
            stride_x1=x.stride(1),
            stride_y0=y.stride(0),
            stride_y1=y.stride(1),
            eps=self.eps,
            alpha=self.alpha,
            damping_f=self.damping_f,
            damping_g=self.damping_g,
            coord_scale=self.coord_scale,
            half_cost_scale=self.half_cost_scale,
            lambda_x=self.lambda_x,
            lambda_y=self.lambda_y,
            CACHE_KEY_N=self.source_size,
            CACHE_KEY_M=self.target_size,
            D=self.feature_dim,
            ALLOW_TF32=True,
            DTYPE_ID=0,
            USE_EXP2=self.use_exp2,
            USE_LABEL_COST=False,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return (f_out, g_out), kernel

    def forward_torch(self, inputs):
        x, y, f_hat, g_hat, log_a, log_b = inputs
        logits = self.coord_scale * (x.float() @ y.float().T) / self.eps
        f_lse = torch.logsumexp(logits + g_hat[None, :] / self.eps + log_b, dim=1)
        g_lse = torch.logsumexp(
            logits + f_hat[:, None] / self.eps + log_a[:, None], dim=0
        )
        f_out = (
            1.0 - self.alpha
        ) * f_hat - self.alpha * self.eps * self.damping_f * f_lse
        g_out = (
            1.0 - self.alpha
        ) * g_hat - self.alpha * self.eps * self.damping_g * g_lse
        return f_out, g_out
