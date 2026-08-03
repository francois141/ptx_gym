import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class GELUKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_size = 1024
        self.size = 4096
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        x = tl.load(x_ptr + offsets)

        x32 = x.to(tl.float32)
        inv_sqrt2 = 0.7071067811865476
        output = 0.5 * x32 * (1.0 + tl.math.erf(x32 * inv_sqrt2))
        tl.store(
            output_ptr + offsets, output.to(output_ptr.dtype.element_ty)
        )

    def get_random_input(self, fixed: bool = False):
        return torch.rand(self.size, device="cuda", dtype=torch.float32)

    def get_shape_information(self) -> str:
        return (
            f"- x_ptr: float32 tensor with shape ({self.size},)\n"
            f"- output_ptr: float32 tensor with shape ({self.size},)"
        )

    def forward_triton(self, x, ptx=False):
        n_elements = x.numel()
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            x,
            output,
            n_elements,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.nn.functional.gelu(inputs)
