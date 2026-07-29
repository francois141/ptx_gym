import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class MonteCarloPiKernel(TritonPTXKernel):
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
    def kernel(x_ptr, y_ptr, output_ptr, n_samples, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_samples
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
        inside_circle = (x * x + y * y <= 1.0) & mask
        inside_count = tl.sum(inside_circle.to(tl.float32), axis=0)
        tl.atomic_add(output_ptr, inside_count * (4.0 / n_samples), sem="relaxed")

    def get_random_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        num_samples = 10_000_000
        coordinates = self._rand_1d(num_samples * 2) * 2.0 - 1.0
        return coordinates[:num_samples], coordinates[num_samples:]

    def forward_triton(
        self, inputs: tuple[torch.Tensor, torch.Tensor], ptx: bool = False
    ) -> tuple[torch.Tensor, object]:
        x, y = inputs
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        if x.numel() == 0:
            raise ValueError("at least one sample is required")
        output = torch.zeros((), device=x.device, dtype=torch.float32)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = (triton.cdiv(x.numel(), self.block_size),)
        launched_kernel = launch_kernel[grid](
            x,
            y,
            output,
            x.numel(),
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = inputs
        return ((x.square() + y.square()) <= 1.0).float().mean() * 4.0
