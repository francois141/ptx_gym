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
    .reg .pred %p_bound, %pneg;
    .reg .u32 %rIdx, %rN, %rTID, %rBID, %rBDIM;
    .reg .u64 %xptr, %optr, %addr;
    .reg .f32 %fx, %fy, %fay, %ft, %tmp, %fpoly, %fe, %fy2, %farg, %ferf, %ferf_signed, %fout, %fone, %fhalf, %f_p, %fa1, %fa2, %fa3, %fa4, %fa5, %flog2e, %fneg;

    ld.param.u64 %xptr, [x_ptr];
    ld.param.u64 %optr, [output_ptr];
    ld.param.u32 %rN, [n_elements];

    mov.u32 %rTID, %tid.x;
    mov.u32 %rBID, %ctaid.x;
    mov.u32 %rBDIM, %ntid.x;
    mad.lo.u32 %rIdx, %rBID, %rBDIM, %rTID;

    setp.ge.u32 %p_bound, %rIdx, %rN;
    @%p_bound bra DONE;

    mad.wide.u32 %addr, %rIdx, 4, %xptr;
    ld.global.f32 %fx, [%addr];

    mov.b32 %fone, 0x3f800000;
    mov.b32 %fhalf, 0x3f000000;
    mov.b32 %f_p, 0x3ea0a1f3;
    mov.b32 %fa1, 0x3e81f635;
    mov.b32 %fa2, 0xbe91a98e;
    mov.b32 %fa3, 0x3fb5f0e3;
    mov.b32 %fa4, 0xbfb9f2a1;
    mov.b32 %fa5, 0x3f87dc22;
    mov.b32 %flog2e, 0x3fb8aa3b;

    mul.rn.f32 %fy, %fx, 0f3f3504f3;

    setp.lt.f32 %pneg, %fy, 0f00000000;
    abs.f32 %fay, %fy;

    fma.rn.f32 %tmp, %fay, %f_p, %fone;
    rcp.approx.f32 %ft, %tmp;

    fma.rn.f32 %fpoly, %fa5, %ft, %fa4;
    fma.rn.f32 %fpoly, %fpoly, %ft, %fa3;
    fma.rn.f32 %fpoly, %fpoly, %ft, %fa2;
    fma.rn.f32 %fpoly, %fpoly, %ft, %fa1;

    mul.rn.f32 %fy2, %fy, %fy;
    neg.f32 %farg, %fy2;
    mul.rn.f32 %farg, %farg, %flog2e;
    ex2.approx.f32 %fe, %farg;

    mul.rn.f32 %tmp, %ft, %fpoly;
    mul.rn.f32 %tmp, %tmp, %fe;
    sub.rn.f32 %ferf, %fone, %tmp;

    neg.f32 %fneg, %ferf;
    selp.f32 %ferf_signed, %fneg, %ferf, %pneg;

    add.rn.f32 %ferf_signed, %ferf_signed, %fone;
    mul.rn.f32 %fout, %fx, %ferf_signed;
    mul.rn.f32 %fout, %fout, %fhalf;

    mad.wide.u32 %addr, %rIdx, 4, %optr;
    st.global.f32 [%addr], %fout;

DONE:
    ret;
}""",
    "BLOCK_SIZE": 256,
    "num_warps": 8,
}


class GELUKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=_ptx_kernel):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)

        x32 = x.to(tl.float32)
        inv_sqrt2 = 0.7071067811865476
        output = 0.5 * x32 * (1.0 + tl.math.erf(x32 * inv_sqrt2))
        tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty), mask=mask)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, x, ptx=False):
        n_elements = x.numel()
        output = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                x, output, n_elements, BLOCK_SIZE=self.block_size, num_warps=32,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                x,
                output,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                num_warps=self.ptx["num_warps"],
            )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.nn.functional.gelu(inputs)
