import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

_ptx_kernel = {
    "ptx":""".version 8.7
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
    .reg .pred %p<12>;
    .reg .b32 %r<9>;
    .reg .b64 %rd<6>;
    .reg .f32 %f<8>;

    ld.param.u64 %rd0, [x_ptr];
    ld.param.u64 %rd1, [output_ptr];
    ld.param.u32 %r0, [n_elements];
    cvta.to.global.u64 %rd0, %rd0;
    cvta.to.global.u64 %rd1, %rd1;

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;
    mov.u32 %r3, %ntid.x;

    and.b64 %rd2, %rd0, 15;
    setp.eq.u64 %p0, %rd2, 0;
    and.b64 %rd3, %rd1, 15;
    setp.eq.u64 %p1, %rd3, 0;
    and.pred %p2, %p0, %p1;

    setp.ge.u32 %p3, %r2, 64;
    @%p3 bra L_done;

L_loop:
    shl.b32 %r4, %r2, 2;
    mad.lo.u32 %r5, %r1, 256, %r4;
    setp.ge.u32 %p4, %r5, %r0;
    @%p4 bra L_done;

    add.u32 %r6, %r5, 1;
    add.u32 %r7, %r5, 2;
    add.u32 %r8, %r5, 3;

    setp.lt.u32 %p5, %r8, %r0;
    and.pred %p6, %p2, %p5;
    @%p6 bra L_vec4;

    mad.wide.u32 %rd4, %r5, 4, %rd0;
    ld.global.f32 %f0, [%rd4];
    max.f32 %f4, %f0, 0f00000000;
    mad.wide.u32 %rd5, %r5, 4, %rd1;
    st.global.f32 [%rd5], %f4;

    setp.lt.u32 %p7, %r6, %r0;
    @%p7 mad.wide.u32 %rd4, %r6, 4, %rd0;
    @%p7 ld.global.f32 %f1, [%rd4];
    @%p7 max.f32 %f5, %f1, 0f00000000;
    @%p7 mad.wide.u32 %rd5, %r6, 4, %rd1;
    @%p7 st.global.f32 [%rd5], %f5;

    setp.lt.u32 %p8, %r7, %r0;
    @%p8 mad.wide.u32 %rd4, %r7, 4, %rd0;
    @%p8 ld.global.f32 %f2, [%rd4];
    @%p8 max.f32 %f6, %f2, 0f00000000;
    @%p8 mad.wide.u32 %rd5, %r7, 4, %rd1;
    @%p8 st.global.f32 [%rd5], %f6;

    setp.lt.u32 %p9, %r8, %r0;
    @%p9 mad.wide.u32 %rd4, %r8, 4, %rd0;
    @%p9 ld.global.f32 %f3, [%rd4];
    @%p9 max.f32 %f7, %f3, 0f00000000;
    @%p9 mad.wide.u32 %rd5, %r8, 4, %rd1;
    @%p9 st.global.f32 [%rd5], %f7;
    bra L_cont;

L_vec4:
    mad.wide.u32 %rd4, %r5, 4, %rd0;
    ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%rd4];
    max.f32 %f4, %f0, 0f00000000;
    max.f32 %f5, %f1, 0f00000000;
    max.f32 %f6, %f2, 0f00000000;
    max.f32 %f7, %f3, 0f00000000;
    mad.wide.u32 %rd5, %r5, 4, %rd1;
    st.global.v4.f32 [%rd5], {%f4, %f5, %f6, %f7};

L_cont:
    add.u32 %r2, %r2, %r3;
    setp.lt.u32 %p10, %r2, 64;
    @%p10 bra L_loop;

L_done:
    ret;
}
""",
    "BLOCK_SIZE": 256,
    "num_warps": 4,
}

class ReLUKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=_ptx_kernel):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        tl.store(output_ptr + offsets, tl.maximum(x, 0.0), mask=mask)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, x, ptx=False):
        output = torch.empty_like(x)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](x, output, n_elements, BLOCK_SIZE=self.block_size)
        else:
            kernel = self.require_compiled_ptx()[grid](
                x,
                output,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                num_warps=self.ptx["num_warps"],
            )
        return output, kernel

    def forward_torch(self, x):
        return torch.relu(x)
