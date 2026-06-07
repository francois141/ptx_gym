import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

_ptx_kernel = {
    "ptx": """.version 8.7
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 x_ptr,
    .param .u64 output_ptr,
    .param .u32 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
{
    .reg .pred %p1;
    .reg .u32 %r_tid, %r_cta, %r_idx, %r_nelem;
    .reg .u64 %xptr, %optr, %off, %xaddr, %oaddr;
    .reg .f32 %x, %absx, %den, %out, %one;

    ld.param.u64 %xptr, [x_ptr];
    ld.param.u64 %optr, [output_ptr];
    ld.param.u32 %r_nelem, [n_elements];

    cvta.to.global.u64 %xptr, %xptr;
    cvta.to.global.u64 %optr, %optr;

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;
    mad.lo.u32 %r_idx, %r_cta, 256, %r_tid;

    setp.ge.u32 %p1, %r_idx, %r_nelem;

    mul.wide.u32 %off, %r_idx, 4;
    add.u64 %xaddr, %xptr, %off;
    add.u64 %oaddr, %optr, %off;

    mov.b32 %one, 0x3f800000;

    @!%p1 ld.global.f32 %x, [%xaddr];
    @!%p1 abs.f32 %absx, %x;
    @!%p1 add.f32 %den, %absx, %one;
    @!%p1 div.rn.f32 %out, %x, %den;
    @!%p1 st.global.f32 [%oaddr], %out;

    ret;
}""",
    "BLOCK_SIZE": 256,
    "num_warps": 8,
}


class SoftsignKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=None):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        x32 = x.to(tl.float32)
        output = x32 / (1.0 + tl.abs(x32))
        tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)

    def get_random_input(self, size=4096):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        output = torch.empty_like(inputs)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                inputs, output, n_elements, BLOCK_SIZE=self.block_size,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                inputs,
                output,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                **self.ptx_launch_kwargs(),
            )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.nn.functional.softsign(inputs)
