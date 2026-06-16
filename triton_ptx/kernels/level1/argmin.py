import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

class ArgminKernel(TritonPTXKernel):

    def __init__(self, block_size=1024, num_warps=4, ptx=None):
        self.block_size = block_size
        self.constexpr_values = {"BLOCK_SIZE": block_size}
        self.num_warps = num_warps
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        x_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)

        offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < n_elements

        x = tl.load(x_ptr + offs, mask=mask, other=float("inf")).to(tl.float32)

        local_val = tl.min(x, axis=0)
        local_idx = tl.argmin(x, axis=0)
        global_idx = pid * BLOCK_SIZE + local_idx

        val_bits = local_val.to(tl.uint32, bitcast=True)
        float_key = val_bits ^ tl.where((val_bits >> 31) != 0, 0xFFFFFFFF, 0x80000000)
        packed = (float_key.to(tl.uint64) << 32) | global_idx.to(tl.uint64)
        # Shift the unsigned ordering into the signed int64 domain used by atomic_min.
        packed = (packed ^ 0x8000000000000000).to(tl.int64, bitcast=True)

        tl.atomic_min(out_ptr, packed, sem="relaxed")

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        out = torch.full((1,), torch.iinfo(torch.int64).max, device=inputs.device, dtype=torch.int64)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            launch_kernel = self.compiled_kernel
            launch_kwargs = dict(num_warps=self.num_warps)
        else:
            launch_kernel = self.compiled_kernel_ptx
            launch_kwargs = self.ptx_launch_kwargs()

        kernel = launch_kernel[grid](
            inputs,
            out,
            n_elements,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        packed = (int(out[0].item()) & 0xFFFFFFFFFFFFFFFF) ^ 0x8000000000000000
        index = packed & 0xFFFFFFFF
        return torch.tensor(index, device=inputs.device, dtype=torch.int64), kernel

    def forward_torch(self, inputs):
        return torch.argmin(inputs, dim=0)
