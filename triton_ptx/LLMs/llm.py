"""Shared interface for language-model backends."""

from abc import ABC, abstractmethod


class LLM(ABC):
    """Contract used by the shared generation and measurement helpers."""

    @classmethod
    @abstractmethod
    def get_kernel_classes(cls):
        """Return the named kernel classes users must compile for this backend."""

    @abstractmethod
    def tokenize_messages(self, messages):
        """Convert chat messages into model input tokens."""

    @abstractmethod
    def generate_tokens(self, input_ids, max_new_tokens):
        """Generate tokens from already-tokenized input."""

    @abstractmethod
    def decode_response(self, output_ids, prompt_length):
        """Decode tokens generated after the prompt."""
