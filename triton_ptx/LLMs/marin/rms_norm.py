import torch
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.LLMs.qwen.rms_norm import QwenRMSNormKernel


class MarinRMSNormKernel(TritonPTXKernel):
    __init__ = QwenRMSNormKernel.__init__
    kernel = staticmethod(QwenRMSNormKernel.kernel)
    forward_triton = QwenRMSNormKernel.forward_triton
    forward_torch = QwenRMSNormKernel.forward_torch

    def get_random_input(self, fixed=False):
        return (
            torch.rand((1, 4096), device="cuda", dtype=torch.bfloat16),
            torch.rand(4096, device="cuda", dtype=torch.bfloat16),
            1e-5,
        )

    def get_shape_information(self):
        return (
            "- input_ptr: bfloat16 tensor with shape (1, 4096)\n"
            "- weight_ptr: bfloat16 tensor with shape (4096,)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 4096)"
        )
