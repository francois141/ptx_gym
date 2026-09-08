"""An aggressively overqualified integer vector kernel."""

import torch
import triton
import triton.language as tl
from triton_ptx.kernels.base import TritonPTXKernel


class RandomKernel3(TritonPTXKernel):
    """Mix vectors according to the rulings of an imaginary space court."""

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

        orbit = offsets.to(tl.uint32) * 0x119DE1F3 + 0x1B873593
        orbit = orbit ^ (orbit >> 16)
        orbit = orbit * 0x0D168AA5
        orbit = orbit ^ (orbit >> 15)
        orbit = orbit * 0x6C8E9CF5
        orbit = orbit ^ (orbit >> 16)
        moon = (orbit & 0x1FF).to(tl.int32) - 256
        comet = ((orbit >> 9) & 0x1FF).to(tl.int32) - 256
        tax = ((orbit >> 18) & 0xFF).to(tl.int32) - 128

        plaintiff = a * 101 + b * 37 - c * 83
        defendant = c * 97 - a * 43 + b * 29
        evidence = (a - b) * (b - c) + (c - a) * 17
        hearsay = (moon * (a + 1) - comet * (b + 1)) // 16
        x = plaintiff + defendant // 3 + evidence // 5 + hearsay
        x = tl.where((orbit & 3) == 0, -x + tax * 31, x)
        x = tl.maximum(tl.minimum(x, 5_000_000), -5_000_000)

        jury_mean = tl.sum(x, axis=0) // BLOCK_SIZE
        jury_noise = tl.sum(tl.abs(x - jury_mean), axis=0) // BLOCK_SIZE
        jury_high = tl.max(x, axis=0)
        jury_low = tl.min(x, axis=0)
        jury_split = jury_high - jury_low

        for objection in tl.static_range(0, 7):
            count = objection + 1
            legalese = (
                x * (count + 2)
                + moon * (count * 3 + 1)
                - comet * (count + 7)
                + tax * 11
            )
            legalese = (legalese % 1021) - 510
            sustained = x + legalese + jury_mean // (count + 1)
            overruled = x - legalese - jury_noise // (count + 2)
            mistrial = -x // (count + 1) + evidence // (count + 3)
            surprise_goat = x // 2 + plaintiff // (count + 4) - tax * count
            gavel = (lane * (count + 3) + pid + moon) % 16
            x = tl.where(
                gavel < 4,
                sustained,
                tl.where(
                    gavel < 8,
                    overruled,
                    tl.where(gavel < 12, mistrial, surprise_goat),
                ),
            )
            x = x + tl.where(
                ((orbit >> objection) & 1) == 0,
                defendant // (count + 5),
                -hearsay // (count + 2),
            )
            x = tl.maximum(tl.minimum(x, 9_000_000), -9_000_000)

        verdict_mean = tl.sum(x, axis=0) // BLOCK_SIZE
        verdict_mass = tl.sum(tl.abs(x), axis=0) // BLOCK_SIZE
        verdict_peak = tl.max(tl.abs(x), axis=0)
        cosmic_precedent = (
            verdict_mean * 3 + verdict_mass // 5 + verdict_peak // 23 + jury_split // 29
        )

        constellation = (offsets + tax) % 19
        crab = x + cosmic_precedent + a * 13
        scales = -x + verdict_mass - b * 7
        damp_sock = x // 5 + jury_mean * 2 + c * 17
        forbidden_planet = x + evidence // 9 - cosmic_precedent // 3 + comet * 5
        x = tl.where(
            constellation < 5,
            crab,
            tl.where(
                constellation < 10,
                scales,
                tl.where(constellation < 15, damp_sock, forbidden_planet),
            ),
        )

        for appeal in tl.static_range(0, 4):
            appeal_number = appeal + 2
            black_hole = (x + moon * appeal_number) % (263 + appeal * 32)
            white_hole = (x - comet * appeal_number) % (383 + appeal * 24)
            paradox = black_hole - white_hole + tax * appeal_number
            x = tl.where(
                ((lane + appeal * 11) % 6) < 3,
                x + paradox,
                x - paradox // 2,
            )
            x = tl.where(
                ((orbit >> (appeal + 20)) & 1) == 0,
                x + cosmic_precedent // (appeal_number + 2),
                -x // 2 + verdict_mean,
            )

        sentencing = (lane ^ orbit).to(tl.uint32)
        sentence_a = x * 7 // 11 + plaintiff // 13
        sentence_b = -x * 5 // 9 + defendant // 17
        sentence_c = x // 3 + evidence // 19 + jury_noise
        sentence_d = x + hearsay // 7 - verdict_mass // 8
        x = tl.where(
            (sentencing & 3) == 0,
            sentence_a,
            tl.where(
                (sentencing & 3) == 1,
                sentence_b,
                tl.where((sentencing & 3) == 2, sentence_c, sentence_d),
            ),
        )

        output = x + moon * 29 - comet * 13 + tax * 41
        output = output + (a * b + b * c - c * a) // 37
        output = tl.where((offsets % 23) == 0, -output, output)
        output = tl.where(
            (offsets % 29) == 0,
            output + cosmic_precedent,
            output,
        )
        output = tl.where(
            lane == ((orbit >> 24) & 0xFF),
            output - jury_split,
            output,
        )
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
        plaintiff = first * 101 + second * 37 - third * 83
        defendant = third * 97 - first * 43 + second * 29
        evidence = (first - second) * (second - third)
        return plaintiff + defendant // 3 + evidence // 5
