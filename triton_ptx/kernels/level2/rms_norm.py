from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class RMSNormKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        num_features=64,
        eps=1e-5,
        batch=8,
        height=64,
        width=64,
        block_size=256,
        num_warps=4,
        ptx=None,
    ):
        self.num_features = num_features
        self.eps = eps
        self.batch = batch
        self.height = height
        self.width = width
        self.block_size = block_size
        self.constexpr_values = {
            "BLOCK_SIZE": block_size,
            "MAX_FEATURES": num_features,
        }
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        output_ptr,
        total,
        features,
        spatial,
        eps,
        BLOCK_SIZE: tl.constexpr,
        MAX_FEATURES: tl.constexpr,
    ):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        feature = (offsets // spatial) % features
        base = offsets - feature * spatial
        feature_offsets = tl.arange(0, MAX_FEATURES)
        valid_features = feature_offsets < features

        values = tl.load(
            x_ptr + base[:, None] + feature_offsets[None, :] * spatial,
            mask=(offsets[:, None] < total) & valid_features[None, :],
            other=0.0,
        )
        mean_square = tl.sum(values * values, axis=1) / features
        x = tl.load(x_ptr + offsets, mask=offsets < total, other=0.0)
        tl.store(
            output_ptr + offsets, x / tl.sqrt(mean_square + eps), mask=offsets < total
        )

    def get_random_input(self):
        return torch.rand(
            (self.batch, self.num_features, self.height, self.width),
            device="cuda",
            dtype=torch.float32,
        )

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        output = torch.empty_like(x)
        total = x.numel()
        spatial = x.shape[2] * x.shape[3]
        grid = lambda meta: (triton.cdiv(total, meta["BLOCK_SIZE"]),)

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            x,
            output,
            total,
            x.shape[1],
            spatial,
            self.eps,
            BLOCK_SIZE=self.block_size,
            MAX_FEATURES=self.num_features,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        rms = torch.sqrt(torch.mean(inputs**2, dim=1, keepdim=True) + self.eps)
        return inputs / rms
