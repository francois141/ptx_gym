import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel


class DifferenceOfSquaresKernel(TritonPTXKernel):
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
    def kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        sum_squared = (x + y) * (x + y)
        difference_squared = (x - y) * (x - y)
        tl.store(output_ptr + offsets, sum_squared - difference_squared, mask=mask)

    def get_random_input(self) -> tuple[torch.Tensor, torch.Tensor]:
        size = 10_000_000
        return self._rand_1d(size) * 2.0 - 1.0, self._rand_1d(size) * 2.0 - 1.0

    def forward_triton(
        self, inputs: tuple[torch.Tensor, torch.Tensor], ptx: bool = False
    ) -> tuple[torch.Tensor, object]:
        x, y = inputs
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        output = torch.empty_like(x)
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
        """Evaluate the expression using PyTorch.

        Args:
            inputs: The ``x`` and ``y`` tensors.

        Returns:
            The elementwise expression values.
        """
        x, y = inputs
        return (x + y).square() - (x - y).square()
