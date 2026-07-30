import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class RMSNormKernel(TritonPTXKernel):
    def __init__(self, *, num_warps=8, eps=1e-6, ptx=None):
        self.size = 4096
        self.eps = eps
        self.constexpr_values = {"BLOCK_SIZE": self.size, "EPS": eps}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
        EPS: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        values = tl.load(x_ptr + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + offsets).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / BLOCK_SIZE + EPS)
        tl.store(output_ptr + offsets, values * inverse_rms * weight)

    def get_random_input(self):
        return (
            torch.randn(self.size, device="cuda", dtype=torch.float32),
            torch.randn(self.size, device="cuda", dtype=torch.float32),
        )

    def forward_triton(self, inputs, ptx=False):
        x, weight = inputs
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (triton.cdiv(self.size, meta["BLOCK_SIZE"]),)
        launched_kernel = launch_kernel[grid](
            x,
            weight,
            output,
            BLOCK_SIZE=self.size,
            EPS=self.eps,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, inputs):
        x, weight = inputs
        inverse_rms = torch.rsqrt(torch.mean(x.square()) + self.eps)
        return x * inverse_rms * weight
