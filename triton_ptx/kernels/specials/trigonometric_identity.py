import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class TrigonometricIdentityKernel(TritonPTXKernel):
    def __init__(
        self,
        *,
        block_size: int = 1024,
        num_warps: int = 4,
        ptx: object | None = None,
    ) -> None:
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        cosine = tl.cos(x)
        sine = tl.sin(x)
        tl.store(output_ptr + offsets, cosine * cosine + sine * sine, mask=mask)

    def get_random_input(self, size: int = 10_000_000) -> torch.Tensor:
        return (self._rand_1d(size) * 2.0 - 1.0) * torch.pi

    def forward_triton(
        self, inputs: torch.Tensor, ptx: bool = False
    ) -> tuple[torch.Tensor, object]:
        output = torch.empty_like(inputs)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = (triton.cdiv(inputs.numel(), self.block_size),)
        launched_kernel = launch_kernel[grid](
            inputs,
            output,
            inputs.numel(),
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.cos(inputs).square() + torch.sin(inputs).square()
