import torch
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.LLMs.qwen.swiglu import QwenSwiGLUKernel


class MarinSwiGLUKernel(TritonPTXKernel):
    __init__ = QwenSwiGLUKernel.__init__
    kernel = staticmethod(QwenSwiGLUKernel.kernel)
    forward_triton = QwenSwiGLUKernel.forward_triton
    forward_torch = QwenSwiGLUKernel.forward_torch

    def get_random_input(self, fixed=False):
        return tuple(
            torch.rand((1, 14336), device="cuda", dtype=torch.bfloat16)
            for _ in range(2)
        )

    def get_shape_information(self):
        return (
            "- gate_ptr, up_ptr: bfloat16 tensors with shape (1, 14336)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 14336)"
        )
