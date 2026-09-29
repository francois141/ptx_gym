import torch


class Float8KernelMixin:
    """Shared input conversion and reference helpers for Float8 kernels."""

    input_dtype = torch.float8_e4m3fn

    @classmethod
    def float8_inputs(cls, inputs, count: int):
        if isinstance(inputs, tuple):
            return tuple(
                value.to(cls.input_dtype) if index < count else value
                for index, value in enumerate(inputs)
            )
        return inputs.to(cls.input_dtype)

    @staticmethod
    def float32_inputs(inputs):
        if isinstance(inputs, tuple):
            return tuple(value.to(torch.float32) for value in inputs)
        return inputs.to(torch.float32)
