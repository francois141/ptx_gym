import torch
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class RMSNormFloat16Kernel(TritonPTXKernel):
    def __init__(self, *, eps=1e-6, ptx=None):
        self.size = 4096
        self.eps = eps
        self.constexpr_values = {"BLOCK_SIZE": self.size, "EPS": eps}
        self.num_warps = 8
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr, weight_ptr, output_ptr, BLOCK_SIZE: tl.constexpr, EPS: tl.constexpr
    ):
        offsets = tl.arange(0, BLOCK_SIZE)
        values = tl.load(x_ptr + offsets).to(tl.float32)
        weight = tl.load(weight_ptr + offsets).to(tl.float32)
        inverse_rms = tl.rsqrt(tl.sum(values * values, axis=0) / BLOCK_SIZE + EPS)
        tl.store(output_ptr + offsets, (values * inverse_rms * weight).to(tl.float16))

    def get_random_input(self, fixed: bool = False):
        return (
            torch.rand(self.size, device="cuda", dtype=torch.float16),
            torch.rand(self.size, device="cuda", dtype=torch.float16),
        )

    def get_shape_information(self) -> str:
        return (
            "- x_ptr: float16 tensor with shape (4096,)\n"
            "- weight_ptr: float16 tensor with shape (4096,)\n"
            "- output_ptr: float16 tensor with shape (4096,)"
        )

    def forward_triton(self, inputs, ptx=False):
        x, weight = inputs
        output = torch.empty_like(x)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        kernel = launch_kernel[(1,)](
            x,
            weight,
            output,
            BLOCK_SIZE=self.size,
            EPS=self.eps,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        x, weight = inputs
        return x * torch.rsqrt(torch.mean(x.square()) + self.eps) * weight
