"""Gemma 4 text-generation implementation."""

from .attention import GemmaAttentionKernel
from .gelu import GemmaGELUKernel
from .linear import GemmaLinearKernel
from .model import Gemma4ForConditionalGeneration, GemmaLLM
from .rms_norm import GemmaRMSNormKernel
from .rope import GemmaRoPEKernel

Gemma = GemmaLLM

__all__ = [
    "Gemma",
    "GemmaAttentionKernel",
    "Gemma4ForConditionalGeneration",
    "GemmaLLM",
    "GemmaGELUKernel",
    "GemmaLinearKernel",
    "GemmaRMSNormKernel",
    "GemmaRoPEKernel",
]
