import torch
import triton
import triton.language as tl

from triton_ptx.helpers import get_ptx_constexpr
from triton_ptx.kernels.base import TritonPTXOperator

_ptx_kernel = {
    "ptx": """.version 8.7
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 x_ptr,
    .param .u64 output_ptr,
    .param .u64 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
.reqntid 128, 1, 1
{
    .reg .pred %p<16>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<16>;
    .shared .align 4 .b8 smem[16];

    ld.param.u64 %rd1, [x_ptr];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u64 %rd3, [output_ptr];
    cvta.to.global.u64 %rd4, %rd3;
    ld.param.u64 %rd5, [n_elements];

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %laneid;
    shr.u32 %r3, %r1, 5;
    mov.u32 %r4, %ctaid.x;

    cvt.u64.u32 %rd6, %r4;
    shl.b64 %rd7, %rd6, 9;
    setp.ge.u64 %p0, %rd7, %rd5;
    @%p0 bra K2_DONE;

    cvt.u64.u32 %rd8, %r1;
    add.u64 %rd9, %rd7, %rd8;
    shl.b64 %rd10, %rd9, 2;
    add.u64 %rd11, %rd2, %rd10;

    mov.f32 %f0, 0f00000000;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;

    setp.lt.u64 %p1, %rd9, %rd5;
    @%p1 ld.global.f32 %f0, [%rd11];

    add.u64 %rd12, %rd9, 128;
    add.u64 %rd13, %rd11, 512;
    setp.lt.u64 %p2, %rd12, %rd5;
    @%p2 ld.global.f32 %f1, [%rd13];

    add.u64 %rd14, %rd9, 256;
    add.u64 %rd15, %rd11, 1024;
    setp.lt.u64 %p3, %rd14, %rd5;
    @%p3 ld.global.f32 %f2, [%rd15];

    add.u64 %rd16, %rd9, 384;
    add.u64 %rd17, %rd11, 1536;
    setp.lt.u64 %p4, %rd16, %rd5;
    @%p4 ld.global.f32 %f3, [%rd17];

    max.f32 %f0, %f0, 0f00000000;
    max.f32 %f1, %f1, 0f00000000;
    max.f32 %f2, %f2, 0f00000000;
    max.f32 %f3, %f3, 0f00000000;

    add.f32 %f8, %f0, %f1;
    add.f32 %f9, %f2, %f3;
    add.f32 %f8, %f8, %f9;

    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 16, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;
    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 8, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;
    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 4, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;
    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 2, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;
    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 1, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;

    setp.eq.u32 %p5, %r2, 0;
    @!%p5 bra K2_NOSTORE;
    setp.eq.u32 %p6, %r3, 0;
    @%p6 st.shared.f32 [smem+0], %f8;
    setp.eq.u32 %p7, %r3, 1;
    @%p7 st.shared.f32 [smem+4], %f8;
    setp.eq.u32 %p8, %r3, 2;
    @%p8 st.shared.f32 [smem+8], %f8;
    setp.eq.u32 %p9, %r3, 3;
    @%p9 st.shared.f32 [smem+12], %f8;

K2_NOSTORE:
    bar.sync 0;

    setp.ne.u32 %p10, %r3, 0;
    @%p10 bra K2_DONE;

    mov.f32 %f8, 0f00000000;
    setp.eq.u32 %p11, %r2, 0;
    @%p11 ld.shared.f32 %f8, [smem+0];
    setp.eq.u32 %p12, %r2, 1;
    @%p12 ld.shared.f32 %f8, [smem+4];
    setp.eq.u32 %p13, %r2, 2;
    @%p13 ld.shared.f32 %f8, [smem+8];
    setp.eq.u32 %p14, %r2, 3;
    @%p14 ld.shared.f32 %f8, [smem+12];

    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 16, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;
    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 8, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;
    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 4, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;
    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 2, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;
    mov.b32 %r5, %f8;
    shfl.sync.down.b32 %r6, %r5, 1, 0x1f, 0xffffffff;
    mov.b32 %f9, %r6;
    add.f32 %f8, %f8, %f9;

    setp.eq.u32 %p15, %r2, 0;
    @!%p15 bra K2_DONE;
    atom.global.add.f32 %f10, [%rd4], %f8;

K2_DONE:
    ret;
}""",
    "BLOCK_SIZE": 512,
}




class ReLUReductionOperator(TritonPTXOperator):
    def __init__(self, block_size=1024, ptx=_ptx_kernel):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        relu_sum = tl.sum(tl.maximum(x, 0.0), axis=0)
        tl.atomic_add(output_ptr, relu_sum, sem="relaxed")

    def get_random_input(self, size=10_000_000):
        return torch.randn(size, device="cuda", dtype=torch.float32)

    def forward_triton(self, x, ptx=False):
        output = torch.zeros((), device=x.device, dtype=x.dtype)
        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](x, output, n_elements, BLOCK_SIZE=self.block_size)
        else:
            kernel = self.require_compiled_ptx()[grid](
                x,
                output,
                n_elements,
                BLOCK_SIZE=(get_ptx_constexpr(self.ptx, "BLOCK_SIZE") or self.block_size),
            )
        return output, kernel

    def forward_torch(self, x):
        return torch.sum(torch.relu(x))
