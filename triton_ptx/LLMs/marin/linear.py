import torch
import triton
import triton.language as tl
from torch.nn import functional
from triton_ptx.kernels.base import TritonPTXKernel


class MarinLinearKernel(TritonPTXKernel):
    def __init__(self, *, ptx=None):
        self.block_m = 16
        self.block_n = 64
        self.block_k = 32
        self.num_warps = 4
        self.num_stages = 3
        self.constexpr_values = {
            "BLOCK_M": self.block_m,
            "BLOCK_N": self.block_n,
            "BLOCK_K": self.block_k,
        }
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        input_ptr,
        weight_ptr,
        output_ptr,
        row_count,
        input_features,
        output_features,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row_offsets = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
        output_offsets = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
        reduction_offsets = tl.arange(0, BLOCK_K)
        input_ptrs = (
            input_ptr
            + row_offsets[:, None] * input_features
            + reduction_offsets[None, :]
        )
        weight_ptrs = (
            weight_ptr
            + output_offsets[None, :] * input_features
            + reduction_offsets[:, None]
        )
        accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for _ in tl.range(0, input_features, BLOCK_K):
            input_values = tl.load(
                input_ptrs,
                mask=row_offsets[:, None] < row_count,
                other=0.0,
            )
            weight_values = tl.load(weight_ptrs)
            accumulator += tl.dot(input_values, weight_values)
            input_ptrs += BLOCK_K
            weight_ptrs += BLOCK_K
        output_ptrs = (
            output_ptr
            + row_offsets[:, None] * output_features
            + output_offsets[None, :]
        )
        tl.store(
            output_ptrs,
            accumulator,
            mask=(row_offsets[:, None] < row_count)
            & (output_offsets[None, :] < output_features),
        )

    def get_random_input(self, fixed=False):
        return (
            torch.rand((1, 4096), device="cuda", dtype=torch.bfloat16),
            torch.rand((4096, 4096), device="cuda", dtype=torch.bfloat16),
        )

    def get_shape_information(self):
        return (
            "- input_ptr: bfloat16 tensor with shape (1, input_features)\n"
            "- weight_ptr: bfloat16 tensor with shape "
            "(output_features, input_features)\n"
            "- output_ptr: bfloat16 tensor with shape (1, output_features)"
        )

    def forward_triton(self, inputs, ptx=False):
        hidden_states, weight = inputs
        input_features = hidden_states.shape[-1]
        output_features, weight_input_features = weight.shape
        assert hidden_states.is_cuda, "Linear requires CUDA input."
        assert hidden_states.is_contiguous(), "Linear input must be contiguous."
        assert hidden_states.dtype in (torch.float16, torch.bfloat16), (
            "Linear input must use float16 or bfloat16."
        )
        assert weight.is_cuda, "Linear weights must be on CUDA."
        assert weight.is_contiguous(), "Linear weights must be contiguous."
        assert weight.dtype == hidden_states.dtype, (
            "Linear weights must match the input dtype."
        )
        assert weight.device == hidden_states.device, (
            "Linear weights must be on the input device."
        )
        assert weight_input_features == input_features, (
            "Linear weight input features must match the input."
        )
        assert input_features % self.block_k == 0, (
            f"Linear input features must be divisible by {self.block_k}."
        )
        assert output_features % self.block_n == 0, (
            f"Linear output features must be divisible by {self.block_n}."
        )
        flattened_input = hidden_states.reshape(-1, input_features)
        output = torch.empty(
            (flattened_input.shape[0], output_features),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs()
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        kernel = launch_kernel[
            (
                triton.cdiv(flattened_input.shape[0], self.block_m),
                triton.cdiv(output_features, self.block_n),
            )
        ](
            flattened_input,
            weight,
            output,
            flattened_input.shape[0],
            input_features,
            output_features,
            BLOCK_M=self.block_m,
            BLOCK_N=self.block_n,
            BLOCK_K=self.block_k,
            **launch_kwargs,
        )
        shape = (*hidden_states.shape[:-1], output_features)
        return output.reshape(shape), kernel

    def forward_torch(self, inputs):
        hidden_states, weight = inputs
        return functional.linear(hidden_states, weight)
