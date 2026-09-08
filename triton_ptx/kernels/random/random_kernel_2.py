"""A needlessly ceremonial integer vector kernel."""

import torch
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class RandomKernel2(TritonPTXKernel):
    """Send three vectors through a tiny, hostile bureaucracy."""

    def __init__(self, *, ptx=None):
        self.size = 4096
        self.batch_size = 256
        self.block_size = 256
        self.constexpr_values = {"BLOCK_SIZE": self.block_size}
        self.num_warps = 8
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
        aligned_offsets = tl.max_contiguous(
            tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE
        )

        a = tl.load(first_ptr + aligned_offsets).to(tl.int32)
        b = tl.load(second_ptr + aligned_offsets).to(tl.int32)
        c = tl.load(third_ptr + aligned_offsets).to(tl.int32)

        passport = offsets.to(tl.uint32) + 0x6D2B79F5
        passport = (passport ^ (passport >> 15)) * 0x2C1B3C6D
        passport = (passport ^ (passport >> 12)) * 0x297A2D39
        passport = passport ^ (passport >> 15)
        stamp = (passport & 0xFF).to(tl.int32) - 127
        queue = ((passport >> 8) & 0xFF).to(tl.int32) - 113
        clerk = ((passport >> 16) & 0x7F).to(tl.int32) - 61

        form_a = a * 31 - b * 17 + c * 13 + stamp
        form_b = (a - c) * (b + 3) - queue * 7
        form_c = (a + b - c) * (clerk + 65)
        x = tl.where((passport & 1) == 0, form_a, form_b)
        x = tl.where((offsets % 7) < 3, x + form_c // 8, x - form_c // 11)
        x = tl.maximum(tl.minimum(x, 4_000_000), -4_000_000)

        office_average = tl.sum(x, axis=0) // BLOCK_SIZE
        office_panic = tl.sum(tl.abs(x - office_average), axis=0) // BLOCK_SIZE
        office_ceiling = tl.max(tl.abs(x), axis=0)
        office_floor = tl.min(x, axis=0)
        rumor = office_average + office_panic // 4 - office_floor // 32

        for hearing in tl.static_range(0, 6):
            hearing_number = hearing + 1
            rotating_door = (
                x * (hearing_number * 2 + 1)
                + stamp * (hearing_number + 5)
                - queue * (hearing_number + 2)
            )
            rotating_door = (rotating_door % 509) - 254
            approved = x + rotating_door + rumor // (hearing_number + 1)
            denied = x - rotating_door + clerk * hearing_number
            misplaced = -x // 2 + office_panic // (hearing_number + 2)
            selector = (offsets + hearing * 19 + pid * 3) % 11
            x = tl.where(
                selector < 4,
                approved,
                tl.where(selector < 8, denied, misplaced),
            )
            contradictory_stamp = tl.where(
                ((passport >> hearing) & 1) == 0,
                form_a // (hearing_number + 2),
                -form_b // (hearing_number + 3),
            )
            x = x + contradictory_stamp
            x = tl.maximum(tl.minimum(x, 8_000_000), -8_000_000)

        committee_sum = tl.sum(x, axis=0) // BLOCK_SIZE
        committee_abs = tl.sum(tl.abs(x), axis=0) // BLOCK_SIZE
        committee_max = tl.max(x, axis=0)
        committee_min = tl.min(x, axis=0)
        committee_range = committee_max - committee_min

        department = offsets % 17
        agriculture = x + a * 9 - committee_sum
        lunar_affairs = -x + b * 7 + committee_abs // 3
        wet_paper = x // 3 + c * 11 - committee_range // 16
        forbidden_staples = x + (a - b) * (b - c) // 8 + office_ceiling // 64
        x = tl.where(
            department < 4,
            agriculture,
            tl.where(
                department < 8,
                lunar_affairs,
                tl.where(department < 13, wet_paper, forbidden_staples),
            ),
        )

        appeal = (x ^ stamp) + queue * 23 - clerk * 19
        appeal = tl.where((lane % 5) == 0, appeal + rumor, appeal)
        appeal = tl.where((lane % 9) == 0, -appeal, appeal)
        appeal = tl.where((lane % 13) == 0, appeal // 2 + office_panic, appeal)
        appeal = tl.maximum(tl.minimum(appeal, 12_000_000), -12_000_000)

        elevator = (lane + pid * 23) % 8
        basement = appeal - committee_abs + form_a // 7
        lobby = appeal + committee_sum + form_b // 9
        mezzanine = -appeal // 3 + committee_range // 11
        roof = appeal + office_ceiling // 17 - office_floor // 19
        appeal = tl.where(
            elevator < 2,
            basement,
            tl.where(
                elevator < 4,
                lobby,
                tl.where(elevator < 6, mezzanine, roof),
            ),
        )

        final_form = appeal * 5 // 7 + x * 2 // 9
        final_form = final_form + (a * b - b * c + c * a) // 32
        final_form = final_form + stamp * 17 + queue * 5 - clerk * 3
        final_form = tl.where(
            ((passport >> 27) & 1) == 0,
            final_form,
            -final_form + rumor,
        )
        final_form = tl.where(
            lane == (passport & 0xFF),
            final_form + committee_range,
            final_form,
        )
        final_form = tl.maximum(tl.minimum(final_form, 2_000_000_000), -2_000_000_000)
        tl.store(output_ptr + aligned_offsets, final_form)

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
        verdict = torch.where(
            (first + third) % 3 == 0,
            first * 31 - second * 17 + third * 13,
            (first - third) * (second + 3),
        )
        return verdict + (first * second - second * third + third * first) // 32
