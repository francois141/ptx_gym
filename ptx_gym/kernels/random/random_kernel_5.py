"""An eldritch spreadsheet disguised as an integer vector kernel."""

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


class RandomKernel5(TritonPTXKernel):
    tuning_options = {}
    """Reconcile three vectors with accounting practices from the abyss."""

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

        ledger = offsets.to(tl.uint32) + 0x7F4A7C15
        ledger = (ledger ^ (ledger >> 15)) * 0x2C9277B5
        ledger = (ledger ^ (ledger >> 13)) * 0x592B84A9
        ledger = ledger ^ (ledger >> 16)
        debit = (ledger & 0xFF).to(tl.int32) - 128
        credit = ((ledger >> 8) & 0xFF).to(tl.int32) - 128
        squid = ((ledger >> 16) & 0xFF).to(tl.int32) - 128
        auditor = ((ledger >> 24) & 0x7F).to(tl.int32) + 1

        assets = a * 127 + b * 61 - c * 89
        liabilities = c * 109 - b * 47 + a * 23
        goodwill = (a - b) * (b - c) + (c - a) * (a + b + c)
        cursed_receipt = debit * 17 - credit * 13 + squid * 11
        balance = assets - liabilities // 3 + goodwill // 7 + cursed_receipt
        balance = tl.where(
            (ledger & 3) == 3,
            -balance + auditor * 43,
            balance,
        )
        balance = tl.maximum(tl.minimum(balance, 7_000_000), -7_000_000)

        market_mean = tl.sum(balance, axis=0) // BLOCK_SIZE
        market_fear = tl.sum(tl.abs(balance - market_mean), axis=0) // BLOCK_SIZE
        market_delusion = tl.max(balance, axis=0)
        market_basement = tl.min(balance, axis=0)
        market_range = market_delusion - market_basement

        for quarter in tl.static_range(0, 9):
            quarter_number = quarter + 1
            projection = (
                balance * (quarter_number + 2)
                + debit * (quarter_number * 7)
                - credit * (quarter_number + 11)
                + squid * (quarter_number + 3)
            )
            projection = (projection % 1151) - 575
            profit = balance + projection + market_mean // (quarter_number + 1)
            loss = balance - projection - market_fear // (quarter_number + 2)
            fraud = -balance // 2 + goodwill // (quarter_number + 3)
            tentacle = balance // 3 + cursed_receipt * quarter_number - auditor * 17
            forecast = (lane * 3 + quarter * 11 + pid * 7 + debit) % 24
            balance = tl.where(
                forecast < 6,
                profit,
                tl.where(
                    forecast < 12,
                    loss,
                    tl.where(forecast < 18, fraud, tentacle),
                ),
            )
            accountant = tl.where(
                ((ledger >> quarter) & 1) == 0,
                liabilities // (quarter_number + 4),
                -assets // (quarter_number + 5),
            )
            balance = balance + accountant
            balance = tl.maximum(tl.minimum(balance, 12_000_000), -12_000_000)

        annual_mean = tl.sum(balance, axis=0) // BLOCK_SIZE
        annual_panic = tl.sum(tl.abs(balance), axis=0) // BLOCK_SIZE
        annual_peak = tl.max(tl.abs(balance), axis=0)
        annual_prophecy = (
            annual_mean * 2 + annual_panic // 5 + annual_peak // 29 + market_range // 31
        )

        column = (offsets + auditor) % 29
        column_a = balance + annual_prophecy + a * 19
        column_b = -balance + annual_panic - b * 23
        column_c = balance // 5 + market_mean * 3 + c * 29
        column_d = balance + goodwill // 13 - annual_peak // 17 + credit * 31
        balance = tl.where(
            column < 7,
            column_a,
            tl.where(
                column < 14,
                column_b,
                tl.where(column < 22, column_c, column_d),
            ),
        )

        pivot_table = balance ^ (debit * 257 + squid)
        pivot_table = pivot_table + credit * auditor - annual_prophecy // 3
        pivot_table = tl.where(
            (offsets % 11) == 0,
            -pivot_table,
            pivot_table,
        )
        pivot_table = tl.where(
            (offsets % 17) == 0,
            pivot_table + market_fear,
            pivot_table,
        )
        pivot_table = tl.where(
            (offsets % 31) == 0,
            pivot_table - market_basement,
            pivot_table,
        )

        for revision in tl.static_range(0, 5):
            revision_number = revision + 2
            circular_reference = (pivot_table + debit * revision_number) % (
                419 + revision * 48
            )
            broken_macro = (pivot_table - squid * revision_number) % (
                647 + revision * 36
            )
            existential_error = (
                circular_reference - broken_macro + credit * revision_number
            )
            pivot_table = tl.where(
                ((lane + revision * 17) % 10) < 5,
                pivot_table + existential_error,
                pivot_table - existential_error // 2,
            )
            pivot_table = tl.where(
                ((ledger >> (revision + 19)) & 1) == 0,
                pivot_table + annual_prophecy // (revision_number + 1),
                -pivot_table // 2 + annual_mean,
            )

        department = (offsets * 3 + auditor) % 16
        payroll = pivot_table * 7 // 11 + assets // 17
        compliance = -pivot_table * 5 // 9 + liabilities // 19
        rituals = pivot_table // 3 + goodwill // 23 + annual_peak
        catering = pivot_table + cursed_receipt - annual_panic // 7
        pivot_table = tl.where(
            department < 4,
            payroll,
            tl.where(
                department < 8,
                compliance,
                tl.where(department < 12, rituals, catering),
            ),
        )

        output = pivot_table + debit * 37 - credit * 29 + squid * 19
        output = output + (a * b + b * c - c * a) // 41
        output = tl.where(
            ((ledger >> 28) & 1) == 0,
            output,
            -output + annual_prophecy,
        )
        output = tl.where(
            lane == ((ledger >> 23) & 0xFF),
            output + market_range,
            output,
        )
        output = tl.where(
            lane == 255,
            output - market_fear + annual_mean,
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
        assets = first * 127 + second * 61 - third * 89
        liabilities = third * 109 - second * 47 + first * 23
        goodwill = (first - second) * (second - third)
        return assets - liabilities // 3 + goodwill // 7
