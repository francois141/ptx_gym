import torch
import triton
import triton.language as tl


class MatrixMultiplicationOperator:
    def __init__(
        self,
        M=1024,
        N=1024,
        K=1024,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ptx=None,
    ):
        self.M, self.N, self.K = M, N, K
        self.BLOCK_SIZE_M = BLOCK_SIZE_M
        self.BLOCK_SIZE_N = BLOCK_SIZE_N
        self.BLOCK_SIZE_K = BLOCK_SIZE_K
        self.GROUP_SIZE_M = GROUP_SIZE_M
        self.compiled_kernel = triton.jit(self.kernel)
        self.ptx = ptx
        if ptx is not None:
            self.compiled_kernel_ptx = triton.jit(self.kernel, ptx=ptx)

    @staticmethod
    def kernel(
        A_ptr,
        B_ptr,
        C_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        A_block_ptr = tl.make_block_ptr(
            base=A_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            offsets=(pid_m * BLOCK_SIZE_M, 0),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            order=(0, 1),
        )
        B_block_ptr = tl.make_block_ptr(
            base=B_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            offsets=(0, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
            order=(1, 0),
        )
        C_block_ptr = tl.make_block_ptr(
            base=C_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
            order=(0, 1),
        )

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            a = tl.load(A_block_ptr)
            b = tl.load(B_block_ptr)
            accumulator += tl.dot(a.to(tl.float32), b.to(tl.float32))
            A_block_ptr = tl.advance(A_block_ptr, (0, BLOCK_SIZE_K))
            B_block_ptr = tl.advance(B_block_ptr, (BLOCK_SIZE_K, 0))

        tl.store(C_block_ptr, accumulator.to(C_ptr.dtype.element_ty))

    def get_random_input(self):
        a = torch.randn((self.M, self.K), device="cuda", dtype=torch.float32)
        b = torch.randn((self.K, self.N), device="cuda", dtype=torch.float32)
        return a, b

    def forward_triton(self, inputs, ptx=False, use_ptx=None):
        a, b = inputs
        M, K = a.shape
        _, N = b.shape
        c = torch.empty((M, N), device=a.device, dtype=torch.float32)
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        if use_ptx is not None:
            ptx = use_ptx

        if not ptx:
            kernel = self.compiled_kernel[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                BLOCK_SIZE_K=self.BLOCK_SIZE_K,
                GROUP_SIZE_M=self.GROUP_SIZE_M,
            )
        else:
            assert self.ptx is not None
            kernel = self.compiled_kernel_ptx[grid](
                a,
                b,
                c,
                M,
                N,
                K,
                a.stride(0),
                a.stride(1),
                b.stride(0),
                b.stride(1),
                c.stride(0),
                c.stride(1),
                BLOCK_SIZE_M=self.BLOCK_SIZE_M,
                BLOCK_SIZE_N=self.BLOCK_SIZE_N,
                BLOCK_SIZE_K=self.BLOCK_SIZE_K,
                GROUP_SIZE_M=self.GROUP_SIZE_M,
            )
        return c, kernel

    def forward_torch(self, inputs):
        a, b = inputs
        return torch.matmul(a, b)
