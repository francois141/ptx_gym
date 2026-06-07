import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

_ptx_kernel = {"ptx": 
               """.version 8.7
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 x_ptr,
    .param .u64 out_ptr,
    .param .u32 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
{
    .reg .pred %p<12>;
    .reg .b32 %r<40>;
    .reg .b64 %rd<20>;
    .reg .f32 %f<6>;

    ld.param.u64 %rd0, [x_ptr];
    ld.param.u64 %rd1, [out_ptr];
    ld.param.u32 %r0, [n_elements];

    cvta.to.global.u64 %rd2, %rd0;
    cvta.to.global.u64 %rd3, %rd1;

    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;

    mul.lo.u32 %r3, %r2, 128;
    add.u32 %r4, %r3, %r1;

    setp.lt.u32 %p0, %r4, %r0;
    mov.b32 %f0, 0xFF800000;
    mad.wide.u32 %rd4, %r4, 4, %rd2;
    @%p0 ld.global.f32 %f0, [%rd4];

    mov.u32 %r6, %r4;

    add.u32 %r5, %r4, 32;
    setp.lt.u32 %p1, %r5, %r0;
    mov.b32 %f1, 0xFF800000;
    mad.wide.u32 %rd5, %r5, 4, %rd2;
    @%p1 ld.global.f32 %f1, [%rd5];
    setp.gt.f32 %p2, %f1, %f0;
    setp.eq.f32 %p3, %f1, %f0;
    setp.lt.u32 %p4, %r5, %r6;
    and.pred %p5, %p3, %p4;
    or.pred %p6, %p2, %p5;
    @%p6 mov.b32 %f0, %f1;
    @%p6 mov.u32 %r6, %r5;

    add.u32 %r7, %r4, 64;
    setp.lt.u32 %p1, %r7, %r0;
    mov.b32 %f1, 0xFF800000;
    mad.wide.u32 %rd6, %r7, 4, %rd2;
    @%p1 ld.global.f32 %f1, [%rd6];
    setp.gt.f32 %p2, %f1, %f0;
    setp.eq.f32 %p3, %f1, %f0;
    setp.lt.u32 %p4, %r7, %r6;
    and.pred %p5, %p3, %p4;
    or.pred %p6, %p2, %p5;
    @%p6 mov.b32 %f0, %f1;
    @%p6 mov.u32 %r6, %r7;

    add.u32 %r8, %r4, 96;
    setp.lt.u32 %p1, %r8, %r0;
    mov.b32 %f1, 0xFF800000;
    mad.wide.u32 %rd7, %r8, 4, %rd2;
    @%p1 ld.global.f32 %f1, [%rd7];
    setp.gt.f32 %p2, %f1, %f0;
    setp.eq.f32 %p3, %f1, %f0;
    setp.lt.u32 %p4, %r8, %r6;
    and.pred %p5, %p3, %p4;
    or.pred %p6, %p2, %p5;
    @%p6 mov.b32 %f0, %f1;
    @%p6 mov.u32 %r6, %r8;

    shfl.sync.down.b32 %f2, %f0, 16, 31, 0xffffffff;
    shfl.sync.down.b32 %r20, %r6, 16, 31, 0xffffffff;
    setp.gt.f32 %p2, %f2, %f0;
    setp.eq.f32 %p3, %f2, %f0;
    setp.lt.u32 %p4, %r20, %r6;
    and.pred %p5, %p3, %p4;
    or.pred %p6, %p2, %p5;
    @%p6 mov.b32 %f0, %f2;
    @%p6 mov.u32 %r6, %r20;

    shfl.sync.down.b32 %f2, %f0, 8, 31, 0xffffffff;
    shfl.sync.down.b32 %r20, %r6, 8, 31, 0xffffffff;
    setp.gt.f32 %p2, %f2, %f0;
    setp.eq.f32 %p3, %f2, %f0;
    setp.lt.u32 %p4, %r20, %r6;
    and.pred %p5, %p3, %p4;
    or.pred %p6, %p2, %p5;
    @%p6 mov.b32 %f0, %f2;
    @%p6 mov.u32 %r6, %r20;

    shfl.sync.down.b32 %f2, %f0, 4, 31, 0xffffffff;
    shfl.sync.down.b32 %r20, %r6, 4, 31, 0xffffffff;
    setp.gt.f32 %p2, %f2, %f0;
    setp.eq.f32 %p3, %f2, %f0;
    setp.lt.u32 %p4, %r20, %r6;
    and.pred %p5, %p3, %p4;
    or.pred %p6, %p2, %p5;
    @%p6 mov.b32 %f0, %f2;
    @%p6 mov.u32 %r6, %r20;

    shfl.sync.down.b32 %f2, %f0, 2, 31, 0xffffffff;
    shfl.sync.down.b32 %r20, %r6, 2, 31, 0xffffffff;
    setp.gt.f32 %p2, %f2, %f0;
    setp.eq.f32 %p3, %f2, %f0;
    setp.lt.u32 %p4, %r20, %r6;
    and.pred %p5, %p3, %p4;
    or.pred %p6, %p2, %p5;
    @%p6 mov.b32 %f0, %f2;
    @%p6 mov.u32 %r6, %r20;

    shfl.sync.down.b32 %f2, %f0, 1, 31, 0xffffffff;
    shfl.sync.down.b32 %r20, %r6, 1, 31, 0xffffffff;
    setp.gt.f32 %p2, %f2, %f0;
    setp.eq.f32 %p3, %f2, %f0;
    setp.lt.u32 %p4, %r20, %r6;
    and.pred %p5, %p3, %p4;
    or.pred %p6, %p2, %p5;
    @%p6 mov.b32 %f0, %f2;
    @%p6 mov.u32 %r6, %r20;

    setp.eq.u32 %p7, %r1, 0;
    @%p7 mov.u32 %r8, %r6;

    @%p7 mov.b32 %r9, %f0;
    @%p7 shr.u32 %r10, %r9, 31;
    @%p7 setp.ne.u32 %p8, %r10, 0;
    @%p7 selp.u32 %r11, 0xffffffff, 0x80000000, %p8;
    @%p7 xor.b32 %r12, %r9, %r11;

    @%p7 not.b32 %r14, %r8;

    @%p7 cvt.u64.u32 %rd10, %r12;
    @%p7 shl.b64 %rd11, %rd10, 32;
    @%p7 cvt.u64.u32 %rd12, %r14;
    @%p7 or.b64 %rd13, %rd11, %rd12;

    @%p7 mov.u64 %rd14, 0x8000000000000000;
    @%p7 xor.b64 %rd15, %rd13, %rd14;

    @%p7 atom.global.max.s64 %rd16, [%rd3], %rd15;

    ret;
}""", 

"BLOCK_SIZE": 128,
"num_warps": 1}


class ArgmaxKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=None):
        self.block_size = block_size
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

        x = tl.load(x_ptr + offs, mask=mask, other=-float("inf")).to(tl.float32)

        local_val = tl.max(x, axis=0)
        local_idx = tl.argmax(x, axis=0)
        global_idx = pid * BLOCK_SIZE + local_idx

        val_bits = local_val.to(tl.uint32, bitcast=True)
        float_key = val_bits ^ tl.where((val_bits >> 31) != 0, 0xFFFFFFFF, 0x80000000)
        inv_idx = 0xFFFFFFFF - global_idx.to(tl.uint32)
        packed = (float_key.to(tl.uint64) << 32) | inv_idx.to(tl.uint64)
        # Shift the unsigned ordering into the signed int64 domain used by atomic_max.
        packed = (packed ^ 0x8000000000000000).to(tl.int64, bitcast=True)

        tl.atomic_max(out_ptr, packed, sem="relaxed")

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        out = torch.full((1,), torch.iinfo(torch.int64).min, device=inputs.device, dtype=torch.int64)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                inputs,
                out,
                n_elements,
                BLOCK_SIZE=self.block_size,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                inputs,
                out,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                num_warps=self.ptx["num_warps"],
            )
        packed = (int(out[0].item()) & 0xFFFFFFFFFFFFFFFF) ^ 0x8000000000000000
        index = 0xFFFFFFFF - (packed & 0xFFFFFFFF)
        return torch.tensor(index, device=inputs.device, dtype=torch.int64), kernel

    def forward_torch(self, inputs):
        return torch.argmax(inputs, dim=0)
