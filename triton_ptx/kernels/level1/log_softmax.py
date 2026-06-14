import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

class LogSoftmaxKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=None):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        output_ptr,
        input_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        if tl.program_id(0) != 0:
            return
        maximum = -float("inf")
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            maximum = tl.maximum(maximum, tl.max(inputs, axis=0))

        denominator = 0.0
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            denominator += tl.sum(tl.exp(inputs - maximum), axis=0)
        logsumexp = tl.log(denominator)

        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            tl.store(output_ptr + offsets, inputs - maximum - logsumexp, mask=mask)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        output = torch.empty_like(inputs)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                output,
                inputs,
                n_elements,
                BLOCK_SIZE=self.block_size,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                output,
                inputs,
                n_elements,
                **self.ptx_launch_kwargs(),
            )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.log_softmax(inputs, dim=0)
