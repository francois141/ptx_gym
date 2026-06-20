from __future__ import annotations

from .base import LLMEndpoint


ADD_KERNEL_PTX = """.version 8.7
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u64 output_ptr,
    .param .u32 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
{
    // Registers
    .reg .pred  p;
    .reg .b32   r_tid, r_cta, r_n, r_idx;
    .reg .b64   rx, ry, ro, ax, ay, ao;
    .reg .f32   fx, fy, fo;

    // Load kernel parameters
    ld.param.u64 rx, [x_ptr];
    ld.param.u64 ry, [y_ptr];
    ld.param.u64 ro, [output_ptr];
    ld.param.u32 r_n, [n_elements];

    // Thread and block indices
    mov.u32 r_tid, %tid.x;
    mov.u32 r_cta, %ctaid.x;

    // Compute starting element index for this CTA: block_start = ctaid.x * 1024
    shl.b32 r_idx, r_cta, 10;

    // Add thread offset within the 1024-tile
    add.u32 r_idx, r_idx, r_tid;

    // Iterate 8 times to cover BLOCK_SIZE=1024 with 128 threads (stride = 128)
    // Iteration 0
    setp.lt.u32 p, r_idx, r_n;
    mad.wide.u32 ax, r_idx, 4, rx;    // addr = x_ptr + idx*4
    mad.wide.u32 ay, r_idx, 4, ry;    // addr = y_ptr + idx*4
    mad.wide.u32 ao, r_idx, 4, ro;    // addr = out_ptr + idx*4
    @p ld.global.ca.f32 fx, [ax];
    @p ld.global.ca.f32 fy, [ay];
    @p add.f32 fo, fx, fy;
    @p st.global.f32 [ao], fo;
    add.u32 r_idx, r_idx, 128;

    // Iteration 1
    setp.lt.u32 p, r_idx, r_n;
    mad.wide.u32 ax, r_idx, 4, rx;
    mad.wide.u32 ay, r_idx, 4, ry;
    mad.wide.u32 ao, r_idx, 4, ro;
    @p ld.global.ca.f32 fx, [ax];
    @p ld.global.ca.f32 fy, [ay];
    @p add.f32 fo, fx, fy;
    @p st.global.f32 [ao], fo;
    add.u32 r_idx, r_idx, 128;

    // Iteration 2
    setp.lt.u32 p, r_idx, r_n;
    mad.wide.u32 ax, r_idx, 4, rx;
    mad.wide.u32 ay, r_idx, 4, ry;
    mad.wide.u32 ao, r_idx, 4, ro;
    @p ld.global.ca.f32 fx, [ax];
    @p ld.global.ca.f32 fy, [ay];
    @p add.f32 fo, fx, fy;
    @p st.global.f32 [ao], fo;
    add.u32 r_idx, r_idx, 128;

    // Iteration 3
    setp.lt.u32 p, r_idx, r_n;
    mad.wide.u32 ax, r_idx, 4, rx;
    mad.wide.u32 ay, r_idx, 4, ry;
    mad.wide.u32 ao, r_idx, 4, ro;
    @p ld.global.ca.f32 fx, [ax];
    @p ld.global.ca.f32 fy, [ay];
    @p add.f32 fo, fx, fy;
    @p st.global.f32 [ao], fo;
    add.u32 r_idx, r_idx, 128;

    // Iteration 4
    setp.lt.u32 p, r_idx, r_n;
    mad.wide.u32 ax, r_idx, 4, rx;
    mad.wide.u32 ay, r_idx, 4, ry;
    mad.wide.u32 ao, r_idx, 4, ro;
    @p ld.global.ca.f32 fx, [ax];
    @p ld.global.ca.f32 fy, [ay];
    @p add.f32 fo, fx, fy;
    @p st.global.f32 [ao], fo;
    add.u32 r_idx, r_idx, 128;

    // Iteration 5
    setp.lt.u32 p, r_idx, r_n;
    mad.wide.u32 ax, r_idx, 4, rx;
    mad.wide.u32 ay, r_idx, 4, ry;
    mad.wide.u32 ao, r_idx, 4, ro;
    @p ld.global.ca.f32 fx, [ax];
    @p ld.global.ca.f32 fy, [ay];
    @p add.f32 fo, fx, fy;
    @p st.global.f32 [ao], fo;
    add.u32 r_idx, r_idx, 128;

    // Iteration 6
    setp.lt.u32 p, r_idx, r_n;
    mad.wide.u32 ax, r_idx, 4, rx;
    mad.wide.u32 ay, r_idx, 4, ry;
    mad.wide.u32 ao, r_idx, 4, ro;
    @p ld.global.ca.f32 fx, [ax];
    @p ld.global.ca.f32 fy, [ay];
    @p add.f32 fo, fx, fy;
    @p st.global.f32 [ao], fo;
    add.u32 r_idx, r_idx, 128;

    // Iteration 7
    setp.lt.u32 p, r_idx, r_n;
    mad.wide.u32 ax, r_idx, 4, rx;
    mad.wide.u32 ay, r_idx, 4, ry;
    mad.wide.u32 ao, r_idx, 4, ro;
    @p ld.global.ca.f32 fx, [ax];
    @p ld.global.ca.f32 fy, [ay];
    @p add.f32 fo, fx, fy;
    @p st.global.f32 [ao], fo;

    ret;
}
"""


ADD_KERNEL_PAYLOAD = {
    "ptx": ADD_KERNEL_PTX,
    "num_threads_x": 128,
}


class FakeAddPrompt(LLMEndpoint):
    """Deterministic fake endpoint for the AddKernel test-time-scaling path."""

    def __init__(self, model: str | None = None, kernel_name: str = "AddKernel"):
        if kernel_name != "AddKernel":
            raise ValueError("fake_add only supports AddKernel")
        self.model = model

    def generate_response(self, prompt: str, *, num_answers: int | None = None) -> list[dict]:
        count = 1 if num_answers is None else num_answers
        return [dict(ADD_KERNEL_PAYLOAD) for _ in range(count)]
