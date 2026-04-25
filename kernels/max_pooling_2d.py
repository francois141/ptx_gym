import torch
import triton
import triton.language as tl


class MaxPooling2DOperator:
    def __init__(self, H=128, W=128, kernel_size=2, stride=2, ptx=None):
        self.H = H
        self.W = W
        self.kernel_size = kernel_size
        self.stride = stride
        self.BLOCK_SIZE_H = 8
        self.BLOCK_SIZE_W = 32
        self.compiled_kernel = triton.jit(self.kernel)
        self.ptx = ptx
        if ptx is not None:
            self.compiled_kernel_ptx = triton.jit(self.kernel, ptx=ptx)

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

    def get_random_input(self):
        input_tensor = torch.randn((1, 1, self.H, self.W), device="cuda", dtype=torch.float16)
        return input_tensor, self.kernel_size, self.stride

    def forward_triton(self, inputs, ptx=False, use_ptx=None):
        input_tensor, kernel_size, stride = inputs
        n, c, h, w = input_tensor.shape
        output_h = (h - kernel_size) // stride + 1
        output_w = (w - kernel_size) // stride + 1
        output_tensor = torch.empty((n, c, output_h, output_w), device=input_tensor.device, dtype=input_tensor.dtype)
        grid = lambda meta: (
            triton.cdiv(output_h, meta["BLOCK_SIZE_H"]),
            triton.cdiv(output_w, meta["BLOCK_SIZE_W"]),
        )

        if use_ptx is not None:
            ptx = use_ptx

        if not ptx:
            kernel = self.compiled_kernel[grid](
                input_tensor.squeeze(),
                output_tensor.squeeze(),
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
            assert self.ptx is not None
            kernel = self.compiled_kernel_ptx[grid](
                input_tensor.squeeze(),
                output_tensor.squeeze(),
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
        return output_tensor, kernel

    def forward_torch(self, inputs):
        input_tensor, kernel_size, stride = inputs
        return torch.nn.functional.max_pool2d(input_tensor, kernel_size=kernel_size, stride=stride)
