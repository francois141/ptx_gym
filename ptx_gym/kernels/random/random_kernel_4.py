"""A haunted-laundromat integer vector kernel."""

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class RandomKernel4(TritonPTXKernel):
    tuning_options = {}
    """Wash three vectors until their arithmetic becomes paranormal."""

    def __init__(self, *, ptx=None):
        self.size = 4096
        self.batch_size = 256
        self.block_size = 256
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

        lint = offsets.to(tl.uint32) + 0x165667B1
        lint = (lint ^ (lint >> 16)) * 0x27D4EB2D
        lint = (lint ^ (lint >> 15)) * 0x0D3A2645
        lint = lint ^ (lint >> 16)
        sock = (lint & 0xFF).to(tl.int32) - 128
        coin = ((lint >> 8) & 0xFF).to(tl.int32) - 128
        ghost = ((lint >> 16) & 0xFF).to(tl.int32) - 128
        detergent = ((lint >> 24) & 0x7F).to(tl.int32) + 1

        whites = a * 73 - b * 41 + c * 19
        colors = b * 67 + c * 23 - a * 31
        delicates = (a - b) * (b - c) * (c - a) // 64
        abandoned_towel = sock * 11 + coin * 7 - ghost * 5
        drum = whites + colors // 2 + delicates + abandoned_towel
        drum = tl.where((lint & 1) == 0, drum, -drum + detergent * 37)
        drum = tl.maximum(tl.minimum(drum, 6_000_000), -6_000_000)

        water_level = tl.sum(drum, axis=0) // BLOCK_SIZE
        foam = tl.sum(tl.abs(drum - water_level), axis=0) // BLOCK_SIZE
        lost_quarter = tl.max(tl.abs(drum), axis=0)
        suspicious_puddle = tl.min(drum, axis=0)

        for cycle in tl.static_range(0, 8):
            cycle_number = cycle + 1
            agitation = (
                drum * (cycle_number + 1)
                + sock * (cycle_number * 5)
                - coin * (cycle_number + 9)
                + ghost * 13
            )
            agitation = (agitation % 769) - 384
            warm = drum + agitation + water_level // (cycle_number + 1)
            cold = drum - agitation + foam // (cycle_number + 2)
            cursed = -drum // 2 + delicates // (cycle_number + 3)
            inexplicably_dry = drum // 3 + abandoned_towel * cycle_number - detergent
            dial = (lane + cycle * 7 + pid * 11 + sock) % 20
            drum = tl.where(
                dial < 5,
                warm,
                tl.where(
                    dial < 10,
                    cold,
                    tl.where(dial < 15, cursed, inexplicably_dry),
                ),
            )
            ghost_tax = tl.where(
                ((lint >> cycle) & 1) == 0,
                colors // (cycle_number + 4),
                -whites // (cycle_number + 5),
            )
            drum = drum + ghost_tax
            drum = tl.maximum(tl.minimum(drum, 10_000_000), -10_000_000)

        spin_mean = tl.sum(drum, axis=0) // BLOCK_SIZE
        spin_violence = tl.sum(tl.abs(drum), axis=0) // BLOCK_SIZE
        spin_peak = tl.max(drum, axis=0)
        spin_abyss = tl.min(drum, axis=0)
        machine_opinion = (
            spin_mean + spin_violence // 7 + spin_peak // 31 - spin_abyss // 37
        )

        machine = offsets % 23
        normal_machine = drum + machine_opinion + a * 5
        screaming_machine = -drum + spin_violence - b * 13
        portal_machine = drum // 4 + water_level * 3 + c * 17
        vending_machine = drum + delicates // 11 - lost_quarter // 17 + coin * 19
        drum = tl.where(
            machine < 6,
            normal_machine,
            tl.where(
                machine < 12,
                screaming_machine,
                tl.where(machine < 18, portal_machine, vending_machine),
            ),
        )

        dryer = drum ^ (sock * 257 + coin)
        dryer = dryer + ghost * detergent - machine_opinion // 3
        dryer = tl.where((offsets % 7) == 0, -dryer, dryer)
        dryer = tl.where((offsets % 11) == 0, dryer + foam, dryer)
        dryer = tl.where(
            (offsets % 17) == 0,
            dryer - suspicious_puddle,
            dryer,
        )

        for tumble in tl.static_range(0, 3):
            tumble_number = tumble + 2
            static = (dryer + sock * tumble_number) % (347 + tumble * 64)
            prophecy = (dryer - ghost * tumble_number) % (599 + tumble * 32)
            dryer = dryer + static - prophecy + coin * tumble_number
            dryer = tl.where(
                ((lane + tumble * 13) % 9) < 4,
                dryer + machine_opinion // (tumble_number + 1),
                -dryer // 2 + spin_mean,
            )

        basket = (offsets + detergent) % 12
        folded = dryer * 5 // 8 + whites // 13
        unfolded = -dryer * 7 // 9 + colors // 17
        stolen = dryer // 3 + delicates // 19 + lost_quarter
        sentient = dryer + abandoned_towel - spin_violence // 5
        dryer = tl.where(
            basket < 3,
            folded,
            tl.where(
                basket < 6,
                unfolded,
                tl.where(basket < 9, stolen, sentient),
            ),
        )

        output = dryer + sock * 31 - coin * 17 + ghost * 13
        output = output + (a * b - b * c + c * a) // 29
        output = tl.where(
            ((lint >> 29) & 1) == 0,
            output,
            -output + machine_opinion,
        )
        output = tl.where(lane == 0, output + spin_peak - spin_abyss, output)
        output = tl.maximum(tl.minimum(output, 2_000_000_000), -2_000_000_000)
        tl.store(output_ptr + aligned_offsets, output)

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
        whites = first * 73 - second * 41 + third * 19
        colors = second * 67 + third * 23 - first * 31
        delicates = (first - second) * (second - third) * (third - first)
        return whites + colors // 2 + delicates // 64
