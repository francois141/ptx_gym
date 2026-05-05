import torch
import triton
import triton.language as tl

from triton_ptx.helpers import get_ptx_constexpr
from triton_ptx.kernels.base import TritonPTXOperator

_ptx_kernel = {
    "ptx": None,
    "H": None,
    "W": None,
    "kernel_h": None,
    "kernel_w": None,
    "stride_h": None,
    "stride_w": None,
    "input_stride_h": None,
    "input_stride_w": None,
    "output_stride_h": None,
    "output_stride_w": None,
    "BLOCK_SIZE_H": None,
    "BLOCK_SIZE_W": None,
}




class MaxPooling2DOperator(TritonPTXOperator):
    def __init__(self, ptx=_ptx_kernel):
        self.BLOCK_SIZE_H = 8
        self.BLOCK_SIZE_W = 32
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        input_ptr,
        output_ptr,
        H: tl.constexpr,
        W: tl.constexpr,
        kernel_h: tl.constexpr,
        kernel_w: tl.constexpr,
        stride_h: tl.constexpr,
        stride_w: tl.constexpr,
        input_stride_h: tl.constexpr,
        input_stride_w: tl.constexpr,
        output_stride_h: tl.constexpr,
        output_stride_w: tl.constexpr,
        BLOCK_SIZE_H: tl.constexpr,
        BLOCK_SIZE_W: tl.constexpr,
    ):
        pid_h = tl.program_id(axis=0)
        pid_w = tl.program_id(axis=1)
        offs_oh = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
        offs_ow = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
        output_H = (H - kernel_h) // stride_h + 1
        output_W = (W - kernel_w) // stride_w + 1
        output_mask = (offs_oh[:, None] < output_H) & (offs_ow[None, :] < output_W)
        base_ih = offs_oh * stride_h
        base_iw = offs_ow * stride_w
        base_ptrs = input_ptr + base_ih[:, None] * input_stride_h + base_iw[None, :] * input_stride_w

        output_block = tl.full((BLOCK_SIZE_H, BLOCK_SIZE_W), float("-inf"), dtype=input_ptr.dtype.element_ty)
        for kh_idx in tl.static_range(0, kernel_h):
            for kw_idx in tl.static_range(0, kernel_w):
                input_data = tl.load(
                    base_ptrs + kh_idx * input_stride_h + kw_idx * input_stride_w,
                    mask=output_mask,
                    other=float("-inf"),
                )
                output_block = tl.maximum(output_block, input_data)

        output_ptrs = output_ptr + offs_oh[:, None] * output_stride_h + offs_ow[None, :] * output_stride_w
        tl.store(output_ptrs, output_block, mask=output_mask)

    def get_random_input(self, H=128, W=128, kernel_size=2, stride=2):
        input_tensor = torch.randn((1, 1, H, W), device="cuda", dtype=torch.float16)
        return input_tensor, kernel_size, stride

    def forward_triton(self, inputs, ptx=False):
        input_tensor, kernel_size, stride = inputs
        n, c, h, w = input_tensor.shape
        if n != 1 or c != 1:
            raise ValueError("MaxPooling2DOperator currently supports only N=C=1 inputs.")
        output_h = (h - kernel_size) // stride + 1
        output_w = (w - kernel_size) // stride + 1
        output_tensor = torch.empty((n, c, output_h, output_w), device=input_tensor.device, dtype=input_tensor.dtype)
        grid = lambda meta: (
            triton.cdiv(output_h, meta["BLOCK_SIZE_H"]),
            triton.cdiv(output_w, meta["BLOCK_SIZE_W"]),
        )

        if not ptx:
            kernel = self.compiled_kernel[grid](
                input_tensor.reshape(h, w),
                output_tensor.reshape(output_h, output_w),
                h,
                w,
                kernel_size,
                kernel_size,
                stride,
                stride,
                input_tensor.stride(-2),
                input_tensor.stride(-1),
                output_tensor.stride(-2),
                output_tensor.stride(-1),
                BLOCK_SIZE_H=self.BLOCK_SIZE_H,
                BLOCK_SIZE_W=self.BLOCK_SIZE_W,
                num_warps=4,
            )
        else:
            ptx_h = (get_ptx_constexpr(self.ptx, "H") or h)
            ptx_w = (get_ptx_constexpr(self.ptx, "W") or w)
            ptx_kernel_h = (get_ptx_constexpr(self.ptx, "kernel_h") or kernel_size)
            ptx_kernel_w = (get_ptx_constexpr(self.ptx, "kernel_w") or kernel_size)
            ptx_stride_h = (get_ptx_constexpr(self.ptx, "stride_h") or stride)
            ptx_stride_w = (get_ptx_constexpr(self.ptx, "stride_w") or stride)
            ptx_input_stride_h = (get_ptx_constexpr(self.ptx, "input_stride_h") or input_tensor.stride(-2))
            ptx_input_stride_w = (get_ptx_constexpr(self.ptx, "input_stride_w") or input_tensor.stride(-1))
            ptx_output_stride_h = (get_ptx_constexpr(self.ptx, "output_stride_h") or output_tensor.stride(-2))
            ptx_output_stride_w = (get_ptx_constexpr(self.ptx, "output_stride_w") or output_tensor.stride(-1))
            kernel = self.require_compiled_ptx()[grid](
                input_tensor.reshape(h, w),
                output_tensor.reshape(output_h, output_w),
                ptx_h,
                ptx_w,
                ptx_kernel_h,
                ptx_kernel_w,
                ptx_stride_h,
                ptx_stride_w,
                ptx_input_stride_h,
                ptx_input_stride_w,
                ptx_output_stride_h,
                ptx_output_stride_w,
                BLOCK_SIZE_H=(get_ptx_constexpr(self.ptx, "BLOCK_SIZE_H") or self.BLOCK_SIZE_H),
                BLOCK_SIZE_W=(get_ptx_constexpr(self.ptx, "BLOCK_SIZE_W") or self.BLOCK_SIZE_W),
                num_warps=4,
            )
        return output_tensor, kernel

    def forward_torch(self, inputs):
        input_tensor, kernel_size, stride = inputs
        return torch.nn.functional.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride)
