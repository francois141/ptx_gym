from __future__ import annotations

import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class LinearDivideSumScaleKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        input_size=32,
        hidden_size=32,
        scaling_factor=1.5,
        batch=32,
        block_m=32,
        block_n=32,
        block_k=32,
        num_warps=4,
        ptx=None,
    ):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.scaling_factor = scaling_factor
        self.batch = batch
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.constexpr_values = {
            "INPUT_SIZE": input_size,
            "HIDDEN_SIZE": hidden_size,
            "BLOCK_M": block_m,
            "BLOCK_N": block_n,
            "BLOCK_K": block_k,
        }
        self.num_warps = num_warps
        self.weight = torch.nn.Parameter(torch.randn((hidden_size, input_size), device="cuda"))
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr, weight_ptr, output_ptr, batch, scaling_factor, stride_xm, stride_wn,
        INPUT_SIZE: tl.constexpr, HIDDEN_SIZE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        sum_acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        offs_k = tl.arange(0, BLOCK_K)
        for n_start in range(0, HIDDEN_SIZE, BLOCK_N):
            offs_n = n_start + tl.arange(0, BLOCK_N)
            x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :]
            w_ptrs = weight_ptr + offs_n[None, :] * stride_wn + offs_k[:, None]
            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for k_start in range(0, tl.cdiv(INPUT_SIZE, BLOCK_K)):
                k_offsets = k_start * BLOCK_K + offs_k
                x = tl.load(x_ptrs, mask=(offs_m[:, None] < batch) & (k_offsets[None, :] < INPUT_SIZE), other=0.0)
                w = tl.load(w_ptrs, mask=(offs_n[None, :] < HIDDEN_SIZE) & (k_offsets[:, None] < INPUT_SIZE), other=0.0)
                acc = tl.dot(x, w, acc=acc, out_dtype=tl.float32, input_precision="ieee")
                x_ptrs += BLOCK_K
                w_ptrs += BLOCK_K
            sum_acc += tl.sum(acc / 2.0, axis=1)
        tl.store(output_ptr + offs_m, sum_acc * scaling_factor, mask=offs_m < batch)

    def get_random_input(self):
        return torch.rand((self.batch, self.input_size), device="cuda", dtype=torch.float32)

    def forward_triton(self, inputs, ptx: bool = False):
        x = inputs
        output = torch.empty((x.shape[0], 1), device=x.device, dtype=x.dtype)
        grid = (triton.cdiv(x.shape[0], self.block_m),)
        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            x,
            self.weight,
            output,
            x.shape[0],
            self.scaling_factor,
            x.stride(0),
            self.weight.stride(0),
            INPUT_SIZE=self.input_size,
            HIDDEN_SIZE=self.hidden_size,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x = torch.matmul(inputs, self.weight.T)
        x = x / 2
        x = torch.sum(x, dim=1, keepdim=True)
        return x * self.scaling_factor
