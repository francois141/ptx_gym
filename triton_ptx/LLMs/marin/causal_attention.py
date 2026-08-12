from triton_ptx.kernels.base import TritonPTXKernel
from triton_ptx.LLMs.qwen.causal_attention import QwenCausalAttentionKernel


class MarinCausalAttentionKernel(TritonPTXKernel):
    __init__ = QwenCausalAttentionKernel.__init__
    kernel = staticmethod(QwenCausalAttentionKernel.kernel)
    get_random_input = QwenCausalAttentionKernel.get_random_input
    get_shape_information = QwenCausalAttentionKernel.get_shape_information
    forward_triton = QwenCausalAttentionKernel.forward_triton
    forward_torch = QwenCausalAttentionKernel.forward_torch
