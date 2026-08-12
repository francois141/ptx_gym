import torch
from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.LLMs.qwen.embedding import QwenEmbeddingKernel


class MarinEmbeddingKernel(TritonPTXKernel):
    __init__ = QwenEmbeddingKernel.__init__
    kernel = staticmethod(QwenEmbeddingKernel.kernel)
    forward_triton = QwenEmbeddingKernel.forward_triton
    forward_torch = QwenEmbeddingKernel.forward_torch

    def get_random_input(self, fixed=False):
        return (
            torch.randint(0, 128, (1, 1), device="cuda"),
            torch.rand((128, 4096), device="cuda", dtype=torch.bfloat16),
        )

    def get_shape_information(self):
        return (
            "- input_ids_ptr: int64 tensor with shape (1, 1)\n"
            "- weight_ptr: bfloat16 tensor with shape (vocab_size, 4096)\n"
            "- output_ptr: bfloat16 tensor with shape (1, 1, 4096)"
        )
