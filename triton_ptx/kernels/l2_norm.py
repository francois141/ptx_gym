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
    .reg .pred %p_active, %p_lane0, %p_warp0, %p_ld, %p_atomic;
    .reg .b32 %r_tid, %r_cta, %r_idx, %r_n, %r_lane, %r_warp;
    .reg .b64 %rd_x, %rd_out, %rd_addr, %rd_sh, %rd_tmp;
    .reg .f32 %f_val, %f_tmp;
    .shared .align 4 .b8 s_sums[128];

    ld.param.u64 %rd_x, [x_ptr];
    ld.param.u64 %rd_out, [output_ptr];
    ld.param.u32 %r_n, [n_elements];

    mov.u32 %r_tid, %tid.x;
    mov.u32 %r_cta, %ctaid.x;

    mul.lo.u32 %r_idx, %r_cta, 1024;
    add.u32 %r_idx, %r_idx, %r_tid;

    setp.lt.u32 %p_active, %r_idx, %r_n;

    mad.wide.u32 %rd_addr, %r_idx, 4, %rd_x;
    mov.f32 %f_val, 0f00000000;
    @%p_active ld.global.f32 %f_val, [%rd_addr];
    mul.f32 %f_val, %f_val, %f_val;

    shfl.sync.down.b32 %f_tmp, %f_val, 16, 0x1f, 0xffffffff;
    add.f32 %f_val, %f_val, %f_tmp;
    shfl.sync.down.b32 %f_tmp, %f_val, 8, 0x1f, 0xffffffff;
    add.f32 %f_val, %f_val, %f_tmp;
    shfl.sync.down.b32 %f_tmp, %f_val, 4, 0x1f, 0xffffffff;
    add.f32 %f_val, %f_val, %f_tmp;
    shfl.sync.down.b32 %f_tmp, %f_val, 2, 0x1f, 0xffffffff;
    add.f32 %f_val, %f_val, %f_tmp;
    shfl.sync.down.b32 %f_tmp, %f_val, 1, 0x1f, 0xffffffff;
    add.f32 %f_val, %f_val, %f_tmp;

    mov.u32 %r_lane, %laneid;
    and.b32 %r_lane, %r_lane, 31;
    setp.eq.u32 %p_lane0, %r_lane, 0;

    shr.u32 %r_warp, %r_tid, 5;

    mov.u64 %rd_sh, s_sums;
    mad.wide.u32 %rd_tmp, %r_warp, 4, %rd_sh;
    @%p_lane0 st.shared.f32 [%rd_tmp], %f_val;

    bar.sync 0;

    setp.eq.u32 %p_warp0, %r_warp, 0;

    mov.f32 %f_val, 0f00000000;
    setp.lt.u32 %p_ld, %r_lane, 32;
    and.pred %p_ld, %p_ld, %p_warp0;
    mad.wide.u32 %rd_tmp, %r_lane, 4, %rd_sh;
    @%p_ld ld.shared.f32 %f_val, [%rd_tmp];

    @%p_warp0 shfl.sync.down.b32 %f_tmp, %f_val, 16, 0x1f, 0xffffffff;
    @%p_warp0 add.f32 %f_val, %f_val, %f_tmp;
    @%p_warp0 shfl.sync.down.b32 %f_tmp, %f_val, 8, 0x1f, 0xffffffff;
    @%p_warp0 add.f32 %f_val, %f_val, %f_tmp;
    @%p_warp0 shfl.sync.down.b32 %f_tmp, %f_val, 4, 0x1f, 0xffffffff;
    @%p_warp0 add.f32 %f_val, %f_val, %f_tmp;
    @%p_warp0 shfl.sync.down.b32 %f_tmp, %f_val, 2, 0x1f, 0xffffffff;
    @%p_warp0 add.f32 %f_val, %f_val, %f_tmp;
    @%p_warp0 shfl.sync.down.b32 %f_tmp, %f_val, 1, 0x1f, 0xffffffff;
    @%p_warp0 add.f32 %f_val, %f_val, %f_tmp;

    and.pred %p_atomic, %p_warp0, %p_lane0;
    @%p_atomic atom.global.add.f32 %f_tmp, [%rd_out], %f_val;

    ret;
}""",
    "BLOCK_SIZE": 1024,
    "num_warps": 32,
}


class L2NormKernel(TritonPTXKernel):
    def __init__(self, block_size=1024, ptx=None):
        self.block_size = block_size
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        x32 = x.to(tl.float32)
        partial_sum = tl.sum(x32 * x32, axis=0)
        tl.atomic_add(output_ptr, partial_sum, sem="relaxed")

    def get_random_input(self, size=10_000_000):
        return self._rand_1d(size)

    def forward_triton(self, inputs, ptx=False):
        n_elements = inputs.numel()
        accum = torch.zeros((), device=inputs.device, dtype=torch.float32)
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

        if not ptx:
            kernel = self.compiled_kernel[grid](
                inputs, accum, n_elements, BLOCK_SIZE=self.block_size,
            )
        else:
            kernel = self.require_compiled_ptx()[grid](
                inputs,
                accum,
                n_elements,
                BLOCK_SIZE=self.ptx["BLOCK_SIZE"],
                num_warps=self.ptx["num_warps"],
            )

        output = torch.sqrt(accum).to(inputs.dtype)
        return output, kernel

    def forward_torch(self, inputs):
        return torch.norm(inputs, p=2)
