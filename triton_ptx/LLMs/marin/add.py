import torch
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.LLMs.qwen.add import QwenAddKernel


class MarinAddKernel(TritonPTXKernel):
    __init__ = QwenAddKernel.__init__
    kernel = staticmethod(QwenAddKernel.kernel)
    forward_triton = QwenAddKernel.forward_triton
    forward_torch = QwenAddKernel.forward_torch

    def get_random_input(self, fixed=False):
        return tuple(
            torch.rand((1, 4096), device="cuda", dtype=torch.bfloat16) for _ in range(2)
        )

    def get_shape_information(self):
        return (
            "- left_ptr, right_ptr: bfloat16 tensors with shape (1, 4096)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 4096)"
        )
