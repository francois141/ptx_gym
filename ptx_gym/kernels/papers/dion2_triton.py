"""
Dion2 fused post-orthogonalization update kernel.
https://github.com/microsoft/dion/blob/main/dion/dion2_triton.py
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


def _build_index_map(indices: torch.Tensor, full_dim: int) -> torch.Tensor:
    """Map selected row indices to their positions in the update tensor."""
    index_map = torch.full(
        indices.shape[:-1] + (full_dim,),
        -1,
        dtype=torch.int32,
        device=indices.device,
    )
    positions = torch.arange(
        indices.shape[-1], dtype=torch.int32, device=indices.device
    ).expand_as(indices)
    return index_map.scatter_(-1, indices, positions)


class Dion2TritonPostOrthogonalize(TritonPTXKernel):
    """Fuse Dion2 weight decay with a selective row or column update."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {}

    def __init__(self, *, ptx=None, select_dim: int = -2):
        if select_dim not in (-2, -1):
            raise ValueError(f"select_dim must be -2 or -1, got {select_dim}")
        self.batch_size = 16
        self.rows = 512
        self.columns = 512
        self.selected_entries = 32
        self.select_dim = select_dim
        self.select_rows = False
        self.block_m = 1 if self.select_rows else 64
        self.block_n = 256 if self.select_rows else 64
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        X_ptr,
        U_ptr,
        map_ptr,
        a_ptr,
        b_ptr,
        M: tl.constexpr,
        N: tl.constexpr,
        x_stride_b: tl.constexpr,
        x_stride_m: tl.constexpr,
        x_stride_n: tl.constexpr,
        u_stride_b: tl.constexpr,
        u_stride_m: tl.constexpr,
        u_stride_n: tl.constexpr,
        map_stride_b: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        SELECT_ROWS: tl.constexpr,
    ):
        """Fused weight-decay + selective update kernel.

        For each element x[b, m, n]:
        - If row m (SELECT_ROWS) or col n (!SELECT_ROWS) is selected with
            position j in U: x = a*x - b*u[b, j, n] (or u[b, m, j])
        - Otherwise: x = a*x

        The masked load returns 0.0 for unselected entries, so the single
        expression ``a*x - b*u`` handles both cases with one FP rounding.

        ``a`` and ``b`` are passed as 0-d device tensors (loaded here) rather than host
        scalars so the step is CUDA-graph capturable: a ``.item()`` on the caller side is a
        host sync that CUDA graph capture forbids, and it would also freeze a scheduled LR at
        the capture-time value under replay. Loading them in-kernel reads the live values.
        """
        tl.static_assert(not SELECT_ROWS, "select row should be false")

        a = tl.load(a_ptr)
        b = tl.load(b_ptr)
        pid = tl.program_id(0)

        num_blocks_m = tl.cdiv(M, BLOCK_M)
        num_blocks_n = tl.cdiv(N, BLOCK_N)
        blocks_per_matrix = num_blocks_m * num_blocks_n

        batch_idx = pid // blocks_per_matrix
        local_pid = pid % blocks_per_matrix
        block_m = local_pid // num_blocks_n
        block_n = local_pid % num_blocks_n

        X_ptr += batch_idx * x_stride_b
        U_ptr += batch_idx * u_stride_b
        map_ptr += batch_idx * map_stride_b

        offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = block_n * BLOCK_N + tl.arange(0, BLOCK_N)

        x_ptrs = X_ptr + offs_m[:, None] * x_stride_m + offs_n[None, :] * x_stride_n
        x = tl.load(x_ptrs)


        if SELECT_ROWS:
            # index_map: (B, M) — position in U's k dimension, or -1
            map_vals = tl.load(map_ptr + offs_m)
            safe_map = tl.where(map_vals >= 0, map_vals, 0)

            # U: (B, k, N)
            u_ptrs = (
                U_ptr + safe_map[:, None] * u_stride_m + offs_n[None, :] * u_stride_n
            )
            u = tl.load(u_ptrs).to(tl.float32)
        else:
            # index_map: (B, N) — position in U's k dimension, or -1
            map_vals = tl.load(map_ptr + offs_n)
            safe_map = tl.where(map_vals >= 0, map_vals, 0)

            # U: (B, M, k)
            u_ptrs = (
                U_ptr + offs_m[:, None] * u_stride_m + safe_map[None, :] * u_stride_n
            )
            u = tl.load(u_ptrs).to(tl.float32)

        # Fused: a*x - b*u. Unselected entries have u=0, so result = a*x.
        result = a * x - b * u

        tl.store(x_ptrs, result)

    def get_random_input(self, fixed: bool = False):
        del fixed
        selected_dim = self.rows if self.select_rows else self.columns
        indices = torch.stack(
            [
                torch.randperm(selected_dim, device="cuda")[: self.selected_entries]
                for _ in range(self.batch_size)
            ]
        )
        x = torch.randn(
            (self.batch_size, self.rows, self.columns),
            device="cuda",
            dtype=torch.float32,
        )
        u_shape = (
            (self.batch_size, self.selected_entries, self.columns)
            if self.select_rows
            else (self.batch_size, self.rows, self.selected_entries)
        )
        u = torch.randn(
            u_shape,
            device="cuda",
            dtype=torch.float32,
        )
        base_lr = torch.tensor(1e-3, device="cuda")
        adjusted_lr = torch.tensor(2e-3, device="cuda")
        weight_decay = torch.tensor(1e-2, device="cuda")
        return x, u, indices, base_lr, adjusted_lr, weight_decay

    def get_shape_information(self) -> str:
        update_shape = (
            f"({self.batch_size}, {self.selected_entries}, {self.columns})"
            if self.select_rows
            else f"({self.batch_size}, {self.rows}, {self.selected_entries})"
        )
        map_dim = self.rows if self.select_rows else self.columns
        return (
            f"- x_ptr: float32 tensor with shape ({self.batch_size}, {self.rows}, "
            f"{self.columns})\n"
            f"- u_ptr: float32 tensor with shape {update_shape}\n"
            f"- map_ptr: int32 tensor with shape ({self.batch_size}, {map_dim})\n"
            "- a_ptr, b_ptr: scalar float32 tensors"
        )

    def forward_triton(self, inputs, ptx: bool = False):
        x, u, indices, base_lr, adjusted_lr, weight_decay = inputs
        x = x.clone()
        selected_dim = x.shape[-2] if self.select_rows else x.shape[-1]
        index_map = _build_index_map(indices, selected_dim)
        a = (1 - base_lr * weight_decay).to(torch.float32)
        b = adjusted_lr.to(torch.float32)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (
            x.shape[0]
            * triton.cdiv(x.shape[-2], meta["BLOCK_M"])
            * triton.cdiv(x.shape[-1], meta["BLOCK_N"]),
        )
        kernel = launch_kernel[grid](
            x,
            u,
            index_map,
            a,
            b,
            x.shape[-2],
            x.shape[-1],
            x.stride(0),
            x.stride(1),
            x.stride(2),
            u.stride(0),
            u.stride(1),
            u.stride(2),
            index_map.stride(0),
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            SELECT_ROWS=self.select_rows,
            **launch_kwargs,
        )
        return x, kernel

    def forward_torch(self, inputs):
        x, u, indices, base_lr, adjusted_lr, weight_decay = inputs
        x = x.clone()
        selected_updates = torch.zeros_like(x)
        if self.select_rows:
            selected_updates.scatter_(
                1,
                indices[..., None].expand(-1, -1, x.shape[-1]),
                u,
            )
        else:
            selected_updates.scatter_(
                2,
                indices[:, None, :].expand(-1, x.shape[-2], -1),
                u,
            )
        return (1 - base_lr * weight_decay) * x - adjusted_lr * selected_updates
