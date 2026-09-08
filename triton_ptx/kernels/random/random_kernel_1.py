"""An integer-only, random-data vector kernel."""

import torch
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class RandomKernel1(TritonPTXKernel):
    """Mix three integer vectors through a branchy integer expression."""

    def __init__(self, *, ptx=None):
        self.size = 4096
        self.batch_size = 256
        self.block_size = 256
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.num_warps = 4
        self.init_compiled_kernels(ptx=ptx, autotune=False)

    @staticmethod
    def kernel(
        first_ptr,
        second_ptr,
        third_ptr,
        output_ptr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        lane = tl.arange(0, BLOCK_SIZE)
        offsets = pid * BLOCK_SIZE + lane
        offsets_hint = tl.max_contiguous(
            tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE
        )

        a = tl.load(first_ptr + offsets_hint).to(tl.int32)
        b = tl.load(second_ptr + offsets_hint).to(tl.int32)
        c = tl.load(third_ptr + offsets_hint).to(tl.int32)

        h = offsets.to(tl.uint32)
        h = h ^ (h >> 16)
        h = h * 0x45D9F3B
        h = h ^ (h >> 16)
        h = h * 0x45D9F3B
        h = h ^ (h >> 16)

        noise = (h & 0xFF).to(tl.int32) - 128
        wave = ((h >> 8) & 0xFF).to(tl.int32) - 128
        x = a * 1618 + b * 707 - c * 421
        cross = (a - b) * (b - c) * (c - a)
        x = x + cross // 8 + noise * 17
        x = tl.maximum(tl.minimum(x, 1_000_000), -1_000_000)

        block_mean = tl.sum(x, axis=0) // BLOCK_SIZE
        deviation = x - block_mean
        block_deviation = tl.sum(tl.abs(deviation), axis=0) // BLOCK_SIZE
        normalized = deviation // (block_deviation + 1)
        x = (x * 5 + normalized * 3 + block_mean) // 8

        for round_id in tl.static_range(0, 4):
            divisor = round_id + 2
            phase = x * (round_id + 1) + noise * (round_id + 3) + pid
            oscillation = (phase % 257) - 128
            branch0 = x + oscillation + (wave * (round_id + 1)) // 2
            branch1 = x - oscillation + normalized * divisor
            selector = ((offsets + round_id * 17) % (5 + divisor)) < (
                2 + (round_id & 1)
            )
            x = tl.where(selector, branch0, branch1)
            x = x + cross // (16 * divisor)
            x = tl.maximum(tl.minimum(x, 1_000_000), -1_000_000)

        abs_x = tl.abs(x)
        block_max = tl.max(abs_x, axis=0)
        energy = tl.sum(abs_x, axis=0) // BLOCK_SIZE
        global_mood = block_mean // 4 + block_max // 32 + energy // 8

        path_a = x + global_mood + wave
        path_b = x - normalized * 3 + global_mood // 2
        path_c = x + (global_mood - x) // 3 + noise
        path_d = x - global_mood
        pid_selector = pid % 4
        x = tl.where(
            pid_selector == 0,
            path_a,
            tl.where(
                pid_selector == 1,
                path_b,
                tl.where(pid_selector == 2, path_c, path_d),
            ),
        )

        cult_id = offsets % 13
        cult_0 = x + (a * b) // 32
        cult_1 = -x // 2 + (c * 11) // 16
        cult_2 = x + ((a - b) * (b - c)) // 16
        cult_3 = (x * (3 + ((h >> 16) & 1).to(tl.int32))) // 4
        x = tl.where(
            cult_id < 3,
            cult_0,
            tl.where(cult_id < 6, cult_1, tl.where(cult_id < 10, cult_2, cult_3)),
        )

        output = x * 11 // 16 + ((a - b) * (b - c) * (c - a)) // 64 + noise
        output = tl.where((offsets % 11) == 0, -output, output)
        output = tl.where((offsets % 17) == 0, output + block_mean, output)
        output = tl.where((offsets % 23) == 0, output + wave * 7, output)
        output = tl.where(lane == 0, output + energy // 24, output)
        output = tl.maximum(tl.minimum(output, 2_000_000_000), -2_000_000_000)
        tl.store(output_ptr + offsets_hint, output)

    def get_random_input(self, fixed: bool = False):
        shape = (self.batch_size, self.size)
        return tuple(
            torch.randint(0, 256, shape, device="cuda", dtype=torch.int32)
            for _ in range(3)
        )

    def get_shape_information(self) -> str:
        shape = f"({self.batch_size}, {self.size})"
        return (
            f"- first_ptr: int32 tensor with shape {shape}\n"
            f"- second_ptr: int32 tensor with shape {shape}\n"
            f"- third_ptr: int32 tensor with shape {shape}\n"
            f"- output_ptr: int32 tensor with shape {shape}"
        )

    def forward_triton(self, inputs, ptx: bool = False):
        first, second, third = inputs
        output = torch.empty_like(first)
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs() if ptx else {"num_warps": self.num_warps}
        )
        grid = lambda meta: (triton.cdiv(first.numel(), meta["BLOCK_SIZE"]),)
        launched_kernel = launch_kernel[grid](
            first,
            second,
            third,
            output,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return output, launched_kernel

    def forward_torch(self, inputs):
        first, second, third = inputs
        mixed = torch.where(
            first > second,
            first * second + third // 8,
            (second - first) * (third + 1),
        )
        return mixed - (first * second) // 4 + third
