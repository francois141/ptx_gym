import torch
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.LLMs.qwen.linear import QwenLinearKernel


class MarinLinearKernel(TritonPTXKernel):
    __init__ = QwenLinearKernel.__init__
    kernel = staticmethod(QwenLinearKernel.kernel)
    get_shape_information = QwenLinearKernel.get_shape_information
    forward_triton = QwenLinearKernel.forward_triton
    forward_torch = QwenLinearKernel.forward_torch

    def get_random_input(self, fixed=False):
        return (
            torch.rand((1, 4096), device="cuda", dtype=torch.bfloat16),
            torch.rand((4096, 4096), device="cuda", dtype=torch.bfloat16),
        )
