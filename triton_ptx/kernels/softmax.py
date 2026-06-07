import torch
import triton
import triton.language as tl

from triton_ptx.kernels.base import TritonPTXKernel

_ptx_kernel = {
    "ptx": """.version 8.7
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 output_ptr,
    .param .u64 input_ptr,
    .param .u32 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
{
    .reg .pred  %p0, %p1, %p2, %p3;
    .reg .b32   %r0, %r1, %r2, %r3, %r4;
    .reg .u32   %r_tid, %lane, %warp, %n, %i;
    .reg .u64   %out_ptr, %in_ptr, %addr, %addr2, %off64, %off64b, %smax_ptr, %ssum_ptr;
    .reg .f32   %f_max, %f_val, %f_tmp, %f_shfl, %f_sum, %f_exp, %f_sub, %f_scale, %f_den, %f_rcp, %f_out, %f_one;

    .shared .align 4 .f32 smax[32];
    .shared .align 4 .f32 ssum[32];

    mov.u32 %r0, %ctaid.x;
    setp.ne.u32 %p0, %r0, 0;
    @%p0 ret;

    ld.param.u64 %out_ptr, [output_ptr];
    ld.param.u64 %in_ptr, [input_ptr];
    ld.param.u32 %n, [n_elements];

    mov.u32 %r_tid, %tid.x;
    and.b32 %lane, %r_tid, 31;
    shr.u32 %warp, %r_tid, 5;

    mov.u64 %smax_ptr, smax;
    mov.u64 %ssum_ptr, ssum;

    mov.b32 %f_max, 0xff800000;
    mov.f32 %f_scale, 1.44269504;

    mov.u32 %i, %r_tid;
MAX_LOOP:
    setp.ge.u32 %p1, %i, %n;
    @%p1 bra MAX_DONE;
    mad.wide.u32 %addr, %i, 4, %in_ptr;
    ld.global.f32 %f_val, [%addr];
    max.f32 %f_max, %f_max, %f_val;
    add.u32 %i, %i, 1024;
    bra MAX_LOOP;
MAX_DONE:

    {
        .reg .b32 %t0, %t1;
        mov.b32 %t0, %f_max;
        shfl.sync.down.b32 %t1, %t0, 16, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t1;
        max.f32 %f_max, %f_max, %f_shfl;
        mov.b32 %t0, %f_max;
        shfl.sync.down.b32 %t1, %t0, 8, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t1;
        max.f32 %f_max, %f_max, %f_shfl;
        mov.b32 %t0, %f_max;
        shfl.sync.down.b32 %t1, %t0, 4, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t1;
        max.f32 %f_max, %f_max, %f_shfl;
        mov.b32 %t0, %f_max;
        shfl.sync.down.b32 %t1, %t0, 2, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t1;
        max.f32 %f_max, %f_max, %f_shfl;
        mov.b32 %t0, %f_max;
        shfl.sync.down.b32 %t1, %t0, 1, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t1;
        max.f32 %f_max, %f_max, %f_shfl;
    }

    setp.eq.u32 %p2, %lane, 0;
    @!%p2 bra SKIP_STORE_MAX;
    mul.wide.u32 %off64, %warp, 4;
    add.u64 %addr2, %smax_ptr, %off64;
    st.shared.f32 [%addr2], %f_max;
SKIP_STORE_MAX:

    bar.sync 0;

    setp.eq.u32 %p3, %warp, 0;
    @!%p3 bra SKIP_WARP0_MAX;
    mul.wide.u32 %off64b, %lane, 4;
    add.u64 %addr2, %smax_ptr, %off64b;
    ld.shared.f32 %f_max, [%addr2];

    {
        .reg .b32 %t2, %t3;
        mov.b32 %t2, %f_max;
        shfl.sync.down.b32 %t3, %t2, 16, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t3;
        max.f32 %f_max, %f_max, %f_shfl;
        mov.b32 %t2, %f_max;
        shfl.sync.down.b32 %t3, %t2, 8, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t3;
        max.f32 %f_max, %f_max, %f_shfl;
        mov.b32 %t2, %f_max;
        shfl.sync.down.b32 %t3, %t2, 4, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t3;
        max.f32 %f_max, %f_max, %f_shfl;
        mov.b32 %t2, %f_max;
        shfl.sync.down.b32 %t3, %t2, 2, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t3;
        max.f32 %f_max, %f_max, %f_shfl;
        mov.b32 %t2, %f_max;
        shfl.sync.down.b32 %t3, %t2, 1, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t3;
        max.f32 %f_max, %f_max, %f_shfl;
    }

    setp.eq.u32 %p2, %lane, 0;
    @!%p2 bra SKIP_WARP0_STORE_MAX;
    st.shared.f32 [%smax_ptr], %f_max;
SKIP_WARP0_STORE_MAX:
SKIP_WARP0_MAX:

    bar.sync 0;

    ld.shared.f32 %f_max, [%smax_ptr];

    mov.f32 %f_sum, 0f00000000;

    mov.u32 %i, %r_tid;
SUM_LOOP:
    setp.ge.u32 %p1, %i, %n;
    @%p1 bra SUM_DONE;
    mad.wide.u32 %addr, %i, 4, %in_ptr;
    ld.global.f32 %f_val, [%addr];
    sub.f32 %f_sub, %f_val, %f_max;
    mul.f32 %f_tmp, %f_sub, %f_scale;
    ex2.approx.f32 %f_exp, %f_tmp;
    add.f32 %f_sum, %f_sum, %f_exp;
    add.u32 %i, %i, 1024;
    bra SUM_LOOP;
SUM_DONE:

    {
        .reg .b32 %t4, %t5;
        mov.b32 %t4, %f_sum;
        shfl.sync.down.b32 %t5, %t4, 16, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t5;
        add.f32 %f_sum, %f_sum, %f_shfl;
        mov.b32 %t4, %f_sum;
        shfl.sync.down.b32 %t5, %t4, 8, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t5;
        add.f32 %f_sum, %f_sum, %f_shfl;
        mov.b32 %t4, %f_sum;
        shfl.sync.down.b32 %t5, %t4, 4, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t5;
        add.f32 %f_sum, %f_sum, %f_shfl;
        mov.b32 %t4, %f_sum;
        shfl.sync.down.b32 %t5, %t4, 2, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t5;
        add.f32 %f_sum, %f_sum, %f_shfl;
        mov.b32 %t4, %f_sum;
        shfl.sync.down.b32 %t5, %t4, 1, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t5;
        add.f32 %f_sum, %f_sum, %f_shfl;
    }

    setp.eq.u32 %p2, %lane, 0;
    @!%p2 bra SKIP_STORE_SUM;
    mul.wide.u32 %off64, %warp, 4;
    add.u64 %addr2, %ssum_ptr, %off64;
    st.shared.f32 [%addr2], %f_sum;
SKIP_STORE_SUM:

    bar.sync 0;

    setp.eq.u32 %p3, %warp, 0;
    @!%p3 bra SKIP_WARP0_SUM;
    mul.wide.u32 %off64b, %lane, 4;
    add.u64 %addr2, %ssum_ptr, %off64b;
    ld.shared.f32 %f_sum, [%addr2];

    {
        .reg .b32 %t6, %t7;
        mov.b32 %t6, %f_sum;
        shfl.sync.down.b32 %t7, %t6, 16, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t7;
        add.f32 %f_sum, %f_sum, %f_shfl;
        mov.b32 %t6, %f_sum;
        shfl.sync.down.b32 %t7, %t6, 8, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t7;
        add.f32 %f_sum, %f_sum, %f_shfl;
        mov.b32 %t6, %f_sum;
        shfl.sync.down.b32 %t7, %t6, 4, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t7;
        add.f32 %f_sum, %f_sum, %f_shfl;
        mov.b32 %t6, %f_sum;
        shfl.sync.down.b32 %t7, %t6, 2, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t7;
        add.f32 %f_sum, %f_sum, %f_shfl;
        mov.b32 %t6, %f_sum;
        shfl.sync.down.b32 %t7, %t6, 1, 0x1f, 0xffffffff;
        mov.b32 %f_shfl, %t7;
        add.f32 %f_sum, %f_sum, %f_shfl;
    }

    setp.eq.u32 %p2, %lane, 0;
    @!%p2 bra SKIP_WARP0_STORE_SUM;
    st.shared.f32 [%ssum_ptr], %f_sum;
SKIP_WARP0_STORE_SUM:
SKIP_WARP0_SUM:

    bar.sync 0;

    ld.shared.f32 %f_den, [%ssum_ptr];
    mov.f32 %f_one, 1.0;
    div.rn.f32 %f_rcp, %f_one, %f_den;

    mov.u32 %i, %r_tid;
WRITE_LOOP:
    setp.ge.u32 %p1, %i, %n;
    @%p1 bra WRITE_DONE;
    mad.wide.u32 %addr, %i, 4, %in_ptr;
    ld.global.f32 %f_val, [%addr];
    sub.f32 %f_sub, %f_val, %f_max;
    mul.f32 %f_tmp, %f_sub, %f_scale;
    ex2.approx.f32 %f_exp, %f_tmp;
    mul.f32 %f_out, %f_exp, %f_rcp;
    mad.wide.u32 %addr2, %i, 4, %out_ptr;
    st.global.f32 [%addr2], %f_out;
    add.u32 %i, %i, 1024;
    bra WRITE_LOOP;
WRITE_DONE:

    ret;
}""",
    "BLOCK_SIZE": 1024,
    "num_warps": 32,
}


class SoftmaxKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=None):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        output_ptr,
        input_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        if tl.program_id(0) != 0:
            return
        maximum = -float("inf")
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            maximum = tl.maximum(maximum, tl.max(inputs, axis=0))

        denominator = 0.0
        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            denominator += tl.sum(tl.exp(inputs - maximum), axis=0)

        for start in tl.range(0, n_elements, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            inputs = tl.load(input_ptr + offsets, mask=mask, other=-float("inf"))
            tl.store(output_ptr + offsets, tl.exp(inputs - maximum) / denominator, mask=mask)

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        output = torch.empty_like(inputs)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                output,
                inputs,
                n_elements,
                BLOCK_SIZE=self.block_size,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                output,
                inputs,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                **self.ptx_launch_kwargs(),
            )
        return output, kernel

    def forward_torch(self, inputs):
        return torch.softmax(inputs, dim=0)
