ptx_add = """
.version 8.0
.target sm_89
.address_size 64

.visible .entry kernel(
.param .u64 x_ptr,
.param .u64 y_ptr,
.param .u64 output_ptr,
.param .u64 n_elements,
.param .u64 dummy_ptr1,
.param .u64 dummy_ptr2
)
.maxntid 256, 1, 1
{
.reg .pred %p<5>;
.reg .b32 %r<12>;
.reg .b64 %rd<14>;
.reg .f32 %f<3>;

```
ld.param.u64 %rd1, [x_ptr];
ld.param.u64 %rd2, [y_ptr];
ld.param.u64 %rd3, [output_ptr];
ld.param.u64 %rd4, [n_elements];

mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %tid.x;
mov.u32 %r3, %ntid.x;
shl.b32 %r4, %r1, 10;
mov.u32 %r5, %r2;
```

L0:
setp.ge.u32 %p1, %r5, 1024;
@%p1 bra L1;

```
add.u32 %r6, %r4, %r5;
cvt.u64.u32 %rd5, %r6;
setp.lt.u64 %p2, %rd5, %rd4;
@!%p2 bra L2;

shl.b64 %rd6, %rd5, 2;
add.u64 %rd7, %rd1, %rd6;
add.u64 %rd8, %rd2, %rd6;
add.u64 %rd9, %rd3, %rd6;

ld.global.f32 %f1, [%rd7];
ld.global.f32 %f2, [%rd8];
add.rn.f32 %f1, %f1, %f2;
st.global.f32 [%rd9], %f1;
```

L2:
add.u32 %r5, %r5, %r3;
bra L0;

L1:
ret;
}
"""

ptx_fused = """
.version 8.0
.target sm_89
.address_size 64

.visible .entry fancy_math_kernel(
    .param .u64 x_ptr,
    .param .u64 y_ptr,
    .param .u32 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
{
    .reg .pred      %p_mask;
    .reg .b32       %r_tid, %r_ctaid, %r_ntid, %r_idx, %r_N;
    .reg .b64       %rd_x, %rd_y, %rd_idx, %rd_offset, %rd_x_addr, %rd_y_addr;
    .reg .f32       %f_zero, %f_log2e, %f_one, %f_x, %f_sx, %f_rsx, %f_cx, %f_cx_log2e, %f_ecx, %f_xsq, %f_neg_xsq, %f_neg_xsq_log2e, %f_exp_neg_xsq, %f_sig_den, %f_sig, %f_res;

    ld.param.u32    %r_N, [n_elements];
    ld.param.u64    %rd_x, [x_ptr];
    ld.param.u64    %rd_y, [y_ptr];

    mov.u32         %r_tid, %tid.x;
    mov.u32         %r_ctaid, %ctaid.x;
    mov.u32         %r_ntid, %ntid.x;
    mad.lo.s32      %r_idx, %r_ctaid, %r_ntid, %r_tid;

    setp.lt.u32     %p_mask, %r_idx, %r_N;
    @!%p_mask bra   L_DONE;

    mov.f32         %f_zero, 0f00000000;
    mov.f32         %f_log2e, 0f3FB8AA3B;
    mov.f32         %f_one, 0f3F800000;

    cvt.u64.u32     %rd_idx, %r_idx;
    shl.b64         %rd_offset, %rd_idx, 2;
    add.s64         %rd_x_addr, %rd_x, %rd_offset;
    add.s64         %rd_y_addr, %rd_y, %rd_offset;

    ld.global.f32   %f_x, [%rd_x_addr];

    sin.approx.ftz.f32  %f_sx, %f_x;
    max.f32             %f_rsx, %f_sx, %f_zero;
    cos.approx.ftz.f32  %f_cx, %f_rsx;
    mul.ftz.f32         %f_cx_log2e, %f_cx, %f_log2e;
    ex2.approx.ftz.f32  %f_ecx, %f_cx_log2e;

    mul.ftz.f32         %f_xsq, %f_x, %f_x;
    neg.f32             %f_neg_xsq, %f_xsq;
    mul.ftz.f32         %f_neg_xsq_log2e, %f_neg_xsq, %f_log2e;
    ex2.approx.ftz.f32  %f_exp_neg_xsq, %f_neg_xsq_log2e;
    add.ftz.f32         %f_sig_den, %f_one, %f_exp_neg_xsq;
    rcp.approx.ftz.f32  %f_sig, %f_sig_den;

    mul.ftz.f32         %f_res, %f_ecx, %f_sig;
    st.global.f32       [%rd_y_addr], %f_res;

L_DONE:
    ret;
}"""

ptx_sigmoid = """
.version 8.0
.target sm_89
.address_size 64

.visible .entry kernel(
.param .u64 x_ptr,
.param .u64 output_ptr,
.param .u64 n_elements,
.param .u64 dummy_ptr1,
.param .u64 dummy_ptr2
)
.reqntid 256, 1, 1
{
.reg .pred %p<5>;
.reg .b32 %r<8>;
.reg .b64 %rd<12>;
.reg .f32 %f<12>;

ld.param.u64 %rd1, [x_ptr];
ld.param.u64 %rd2, [output_ptr];
ld.param.u64 %rd3, [n_elements];

mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %tid.x;

shl.b32 %r3, %r1, 8;
add.u32 %r4, %r3, %r2;

cvt.u64.u32 %rd4, %r4;
setp.ge.u64 %p1, %rd4, %rd3;
@%p1 bra DONE;

shl.b64 %rd5, %rd4, 2;
add.u64 %rd6, %rd1, %rd5;
add.u64 %rd7, %rd2, %rd5;

ld.global.f32 %f1, [%rd6];
neg.f32 %f2, %f1;
mul.rn.f32 %f3, %f2, 0f3FB8AA3B;
ex2.approx.ftz.f32 %f4, %f3;
add.rn.f32 %f5, %f4, 0f3F800000;
rcp.approx.ftz.f32 %f6, %f5;
store.global.f32 [%rd7], %f6;

DONE:
ret;
}
"""

ptx_sigmoid2 = """
.version 8.0
.target sm_89
.address_size 64

.visible .entry kernel(
    .param .u64 x_ptr,
    .param .u64 output_ptr,
    .param .u64 n_elements,
    .param .u64 dummy_ptr1,
    .param .u64 dummy_ptr2
)
.reqntid 256, 1, 1
{
    .reg .pred %p<6>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<25>;
    .reg .f32 %f<25>;

    ld.param.u64 %rd1, [x_ptr];
    ld.param.u64 %rd2, [output_ptr];
    ld.param.u64 %rd3, [n_elements];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %tid.x;

    cvt.u64.u32 %rd4, %r1;
    shl.b64 %rd5, %rd4, 10;
    add.u64 %rd6, %rd5, 1023;
    setp.lt.u64 %p1, %rd6, %rd3;

    cvt.u64.u32 %rd7, %r2;
    add.u64 %rd8, %rd5, %rd7;

    shl.b64 %rd9, %rd8, 2;
    add.u64 %rd10, %rd1, %rd9;
    add.u64 %rd11, %rd2, %rd9;

    @!%p1 bra TAIL;

    ld.global.ca.f32 %f1, [%rd10];
    add.u64 %rd12, %rd10, 1024;
    ld.global.ca.f32 %f2, [%rd12];
    add.u64 %rd13, %rd10, 2048;
    ld.global.ca.f32 %f3, [%rd13];
    add.u64 %rd14, %rd10, 3072;
    ld.global.ca.f32 %f4, [%rd14];

    neg.ftz.f32 %f5, %f1;
    neg.ftz.f32 %f6, %f2;
    neg.ftz.f32 %f7, %f3;
    neg.ftz.f32 %f8, %f4;

    mul.rn.ftz.f32 %f9, %f5, 0f3FB8AA3B;
    mul.rn.ftz.f32 %f10, %f6, 0f3FB8AA3B;
    mul.rn.ftz.f32 %f11, %f7, 0f3FB8AA3B;
    mul.rn.ftz.f32 %f12, %f8, 0f3FB8AA3B;

    ex2.approx.ftz.f32 %f13, %f9;
    ex2.approx.ftz.f32 %f14, %f10;
    ex2.approx.ftz.f32 %f15, %f11;
    ex2.approx.ftz.f32 %f16, %f12;

    add.rn.ftz.f32 %f17, %f13, 0f3F800000;
    add.rn.ftz.f32 %f18, %f14, 0f3F800000;
    add.rn.ftz.f32 %f19, %f15, 0f3F800000;
    add.rn.ftz.f32 %f20, %f16, 0f3F800000;

    rcp.approx.ftz.f32 %f21, %f17;
    rcp.approx.ftz.f32 %f22, %f18;
    rcp.approx.ftz.f32 %f23, %f19;
    rcp.approx.ftz.f32 %f24, %f20;

    st.global.wb.f32 [%rd11], %f21;
    add.u64 %rd15, %rd11, 1024;
    st.global.wb.f32 [%rd15], %f22;
    add.u64 %rd16, %rd11, 2048;
    st.global.wb.f32 [%rd16], %f23;
    add.u64 %rd17, %rd11, 3072;
    st.global.wb.f32 [%rd17], %f24;

    bra DONE;

TAIL:
    add.u64 %rd12, %rd8, 256;
    add.u64 %rd13, %rd8, 512;
    add.u64 %rd14, %rd8, 768;

    setp.lt.u64 %p2, %rd8, %rd3;
    setp.lt.u64 %p3, %rd12, %rd3;
    setp.lt.u64 %p4, %rd13, %rd3;
    setp.lt.u64 %p5, %rd14, %rd3;

    @%p2 ld.global.ca.f32 %f1, [%rd10];
    add.u64 %rd15, %rd10, 1024;
    @%p3 ld.global.ca.f32 %f2, [%rd15];
    add.u64 %rd16, %rd10, 2048;
    @%p4 ld.global.ca.f32 %f3, [%rd16];
    add.u64 %rd17, %rd10, 3072;
    @%p5 ld.global.ca.f32 %f4, [%rd17];

    @%p2 neg.ftz.f32 %f5, %f1;
    @%p3 neg.ftz.f32 %f6, %f2;
    @%p4 neg.ftz.f32 %f7, %f3;
    @%p5 neg.ftz.f32 %f8, %f4;

    @%p2 mul.rn.ftz.f32 %f9, %f5, 0f3FB8AA3B;
    @%p3 mul.rn.ftz.f32 %f10, %f6, 0f3FB8AA3B;
    @%p4 mul.rn.ftz.f32 %f11, %f7, 0f3FB8AA3B;
    @%p5 mul.rn.ftz.f32 %f12, %f8, 0f3FB8AA3B;

    @%p2 ex2.approx.ftz.f32 %f13, %f9;
    @%p3 ex2.approx.ftz.f32 %f14, %f10;
    @%p4 ex2.approx.ftz.f32 %f15, %f11;
    @%p5 ex2.approx.ftz.f32 %f16, %f12;

    @%p2 add.rn.ftz.f32 %f17, %f13, 0f3F800000;
    @%p3 add.rn.ftz.f32 %f18, %f14, 0f3F800000;
    @%p4 add.rn.ftz.f32 %f19, %f15, 0f3F800000;
    @%p5 add.rn.ftz.f32 %f20, %f16, 0f3F800000;

    @%p2 rcp.approx.ftz.f32 %f21, %f17;
    @%p3 rcp.approx.ftz.f32 %f22, %f18;
    @%p4 rcp.approx.ftz.f32 %f23, %f19;
    @%p5 rcp.approx.ftz.f32 %f24, %f20;

    @%p2 st.global.wb.f32 [%rd11], %f21;
    add.u64 %rd18, %rd11, 1024;
    @%p3 st.global.wb.f32 [%rd18], %f22;
    add.u64 %rd19, %rd11, 2048;
    @%p4 st.global.wb.f32 [%rd19], %f23;
    add.u64 %rd20, %rd11, 3072;
    @%p5 st.global.wb.f32 [%rd20], %f24;

DONE:
    ret;
}
"""

ptx_pooling = """
.version 8.0
.target sm_89
.address_size 64

.visible .entry kernel(
.param .u64 input_ptr,
.param .u64 output_ptr,
.param .u64 dummy_ptr1,
.param .u64 dummy_ptr2
)
{
.reg .pred %p<17>;
.reg .b32 %r<64>;
.reg .b64 %rd<32>;
.reg .f32 %f<20>;
}
ld.param.u64 %rd1, [input_ptr];
ld.param.u64 %rd2, [output_ptr];

mov.u32 %r1, %ctaid.x;
mov.u32 %r2, %ctaid.y;
mov.u32 %r3, %tid.x;
mov.u32 %r4, %nctaid.y;

and.b32 %r5, %r3, 15;
shr.u32 %r6, %r3, 4;

setp.ge.u32 %p1, %r3, 256;
@%p1 bra DONE;

shl.b32 %r7, %r1, 4;
add.u32 %r8, %r7, %r6;
shl.b32 %r9, %r2, 4;
add.u32 %r10, %r9, %r5;

shl.b32 %r11, %r4, 4;

mul.lo.u32 %r12, %r8, %r11;
add.u32 %r13, %r12, %r10;
cvt.u64.u32 %rd3, %r13;
shl.b64 %rd4, %rd3, 2;
add.u64 %rd5, %rd2, %rd4;

shl.b32 %r14, %r11, 1;
shl.b32 %r15, %r8, 1;
shl.b32 %r16, %r10, 1;

mul.lo.u32 %r17, %r15, %r14;
add.u32 %r18, %r17, %r16;
cvt.u64.u32 %rd6, %r18;
shl.b64 %rd7, %rd6, 2;
add.u64 %rd8, %rd1, %rd7;

ld.global.f32 %f1, [%rd8];
add.u64 %rd9, %rd8, 4;
ld.global.f32 %f2, [%rd9];
cvt.u64.u32 %rd10, %r14;
shl.b64 %rd11, %rd10, 2;
add.u64 %rd12, %rd8, %rd11;
ld.global.f32 %f3, [%rd12];
add.u64 %rd13, %rd12, 4;
ld.global.f32 %f4, [%rd13];

max.f32 %f5, %f1, %f2;
max.f32 %f6, %f3, %f4;
max.f32 %f7, %f5, %f6;

st.global.f32 [%rd5], %f7;
DONE:
ret;
}"""


ptx_values = {
    "AddOperator": ptx_add,
    "FancyFusedOperator": ptx_fused,
    "SigmoidOperator": ptx_sigmoid2,
    "MaxPooling2DOperator": ptx_pooling
}