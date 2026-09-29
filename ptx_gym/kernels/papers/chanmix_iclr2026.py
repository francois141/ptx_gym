"""
Mixed-bit KV-cache packing kernels used by ChanMix (ICLR 2026).
https://github.com/cxiliao/ChanMix
"""

from __future__ import annotations

from typing import ClassVar

import torch
import triton
import triton.language as tl
from ptx_gym.kernels.base import TritonPTXKernel


def _cache_layout(channel_store_bits, device):
    """Build ChanMix's packed-byte metadata for channels in quantized order."""
    bits = torch.as_tensor(channel_store_bits, device=device, dtype=torch.int32)
    if bits.ndim != 1 or bits.numel() == 0:
        raise ValueError(
            "channel_store_bits must be a non-empty one-dimensional tensor."
        )
    if not torch.all((bits == 1) | (bits == 2) | (bits == 4) | (bits == 8)):
        raise ValueError(
            "ChanMix cache storage supports only 1-, 2-, 4-, or 8-bit channels."
        )

    cumulative_bits = bits.cumsum(0)
    if cumulative_bits[-1].item() % 8:
        raise ValueError(
            "The total number of channel storage bits must be divisible by 8."
        )

    offsets = 8 - cumulative_bits.remainder(8)
    offsets[offsets == 8] = 0
    packed_indices = torch.div(cumulative_bits - 1, 8, rounding_mode="floor")

    def groups(bit_width):
        indices = torch.where(bits == bit_width)[0]
        group_size = 8 // bit_width
        if indices.numel() % group_size:
            raise ValueError(
                f"The {bit_width}-bit channels must occur in complete {group_size}-channel "
                "packing groups."
            )
        starts = indices[::group_size]
        if indices.numel() and not torch.equal(
            indices,
            (starts[:, None] + torch.arange(group_size, device=device)).reshape(-1),
        ):
            raise ValueError(
                f"The {bit_width}-bit channels must be contiguous packing groups."
            )
        return starts, packed_indices[starts]

    one_starts, one_packed = groups(1)
    two_starts, two_packed = groups(2)
    four_starts, four_packed = groups(4)
    eight_starts, eight_packed = groups(8)
    return {
        "bits": bits,
        "offsets": offsets,
        "packed_indices": packed_indices,
        "packed_dim": int(cumulative_bits[-1].item() // 8),
        "one_starts": one_starts,
        "one_packed": one_packed,
        "two_starts": two_starts,
        "two_packed": two_packed,
        "four_starts": four_starts,
        "four_packed": four_packed,
        "eight_starts": eight_starts,
        "eight_packed": eight_packed,
    }


class ChanMixICLR2026SetCache(TritonPTXKernel):
    """Pack ChanMix's already-quantized mixed-bit KV-cache values into bytes."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "num_warps": (1, 2, 4, 8),
        "num_stages": (1, 2, 3, 4),
    }

    def __init__(
        self,
        *,
        channel_store_bits=None,
        batch_size=2,
        sequence_length=128,
        block_size=64,
        ptx=None,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.block_size = block_size
        self.num_warps = 4
        self.num_stages = 2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if channel_store_bits is None:
            channel_store_bits = torch.full((512,), 4, dtype=torch.int32)
        self.layout = _cache_layout(channel_store_bits, device)
        self.channel_dim = self.layout["bits"].numel()
        if self.channel_dim % block_size:
            raise ValueError("channel_dim must be divisible by block_size.")
        self.packed_channel_dim = self.layout["packed_dim"]
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        packed_cache_ptr,
        quantized_cache_ptr,
        channel_offset_bit_ptr,
        channel_2bit_ptr,
        channel_4bit_ptr,
        channel_8bit_ptr,
        channel_2bit_q2s_idx_ptr,
        channel_4bit_q2s_idx_ptr,
        channel_8bit_q2s_idx_ptr,
        channel_2bit_num,
        channel_4bit_num,
        channel_8bit_num,
        n_cache_ele,
        channel_blocks: tl.constexpr,
        channel_dim: tl.constexpr,
        packed_channel_dim: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(0)

        # Get token_id and block_id
        token_id = pid // channel_blocks
        block_id = pid % channel_blocks

        # The offset may exceed pack_dim, so we need mask
        channel_offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

        if channel_2bit_num != 0:
            # 1. Original idx
            quantized_2bit_channel_offsets = channel_offsets % channel_2bit_num
            quantized_2bit_channel = tl.load(
                channel_2bit_ptr + quantized_2bit_channel_offsets
            )

            # 2. Set 2d offsets and calculate shifted cache
            group_2bit_offsets = (
                quantized_2bit_channel[:, None] + tl.arange(0, 4)[None, :]
            )
            ele_2bit_offsets = token_id * channel_dim + group_2bit_offsets
            quantized_2bit_cache = tl.load(
                quantized_cache_ptr + ele_2bit_offsets,
                mask=ele_2bit_offsets < n_cache_ele,
            )
            offsets_2bit = tl.load(
                channel_offset_bit_ptr + group_2bit_offsets,
                mask=group_2bit_offsets < channel_dim,
            )
            shifted_cache_2bit = quantized_2bit_cache << offsets_2bit

            # 3. Packed idx
            q2s_2bit_idx = tl.load(
                channel_2bit_q2s_idx_ptr + quantized_2bit_channel_offsets
            )
            offsets_2bit_packed = token_id * packed_channel_dim + q2s_2bit_idx
            tl.store(
                packed_cache_ptr + offsets_2bit_packed,
                tl.sum(shifted_cache_2bit, axis=1),
            )

        if channel_4bit_num != 0:
            # 1. Original idx
            quantized_4bit_channel_offsets = channel_offsets % channel_4bit_num
            quantized_4bit_channel = tl.load(
                channel_4bit_ptr + quantized_4bit_channel_offsets
            )

            # 2. Set 2d offsets and calculate shifted cache
            group_4bit_offsets = (
                quantized_4bit_channel[:, None] + tl.arange(0, 2)[None, :]
            )
            ele_4bit_offsets = token_id * channel_dim + group_4bit_offsets
            quantized_4bit_cache = tl.load(
                quantized_cache_ptr + ele_4bit_offsets,
                mask=ele_4bit_offsets < n_cache_ele,
            )
            offsets_4bit = tl.load(
                channel_offset_bit_ptr + group_4bit_offsets,
                mask=group_4bit_offsets < channel_dim,
            )
            shifted_cache_4bit = quantized_4bit_cache << offsets_4bit

            # 3. Packed idx
            q2s_4bit_idx = tl.load(
                channel_4bit_q2s_idx_ptr + quantized_4bit_channel_offsets
            )
            offsets_4bit_packed = token_id * packed_channel_dim + q2s_4bit_idx
            tl.store(
                packed_cache_ptr + offsets_4bit_packed,
                tl.sum(shifted_cache_4bit, axis=1),
            )

        if channel_8bit_num != 0:
            # 1. Original idx
            quantized_8bit_channel_offsets = channel_offsets % channel_8bit_num
            quantized_8bit_channel = tl.load(
                channel_8bit_ptr + quantized_8bit_channel_offsets
            )

            # 2. Set 2d offsets and calculate shifted cache
            ele_8bit_offsets = token_id * channel_dim + quantized_8bit_channel
            quantized_8bit_cache = tl.load(
                quantized_cache_ptr + ele_8bit_offsets,
                mask=ele_8bit_offsets < n_cache_ele,
            )

            # 3. Packed idx
            q2s_8bit_idx = tl.load(
                channel_8bit_q2s_idx_ptr + quantized_8bit_channel_offsets
            )
            offsets_8bit_packed = token_id * packed_channel_dim + q2s_8bit_idx
            tl.store(packed_cache_ptr + offsets_8bit_packed, quantized_8bit_cache)

    def get_random_input(self, fixed=False):
        del fixed
        maximum = (1 << self.layout["bits"]) - 1
        values = torch.rand(
            self.batch_size,
            self.sequence_length,
            self.channel_dim,
            device=self.layout["bits"].device,
        )
        return (values.mul(maximum).to(torch.uint8),)

    def get_shape_information(self):
        return (
            f"- quantized_cache_ptr: uint8 tensor with shape ({self.batch_size}, "
            f"{self.sequence_length}, {self.channel_dim})\n"
            f"- packed_cache_ptr: uint8 tensor with shape ({self.batch_size}, "
            f"{self.sequence_length}, {self.packed_channel_dim})\n"
            "- channel metadata: int32 tensors describing 1/2/4/8-bit groups"
        )

    def forward_triton(self, inputs, ptx=False):
        (quantized_cache,) = inputs
        if quantized_cache.dtype != torch.uint8 or quantized_cache.ndim != 3:
            raise ValueError("quantized_cache must be a rank-3 uint8 tensor.")
        if quantized_cache.shape[-1] != self.channel_dim:
            raise ValueError("quantized_cache has an unexpected channel dimension.")
        packed = torch.empty(
            *quantized_cache.shape[:2],
            self.packed_channel_dim,
            dtype=torch.uint8,
            device=quantized_cache.device,
        )
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs()
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        channel_blocks = self.channel_dim // self.block_size
        kernel = launch_kernel[
            (quantized_cache.shape[0] * quantized_cache.shape[1] * channel_blocks,)
        ](
            packed,
            quantized_cache,
            self.layout["offsets"],
            self.layout["two_starts"],
            self.layout["four_starts"],
            self.layout["eight_starts"],
            self.layout["two_packed"],
            self.layout["four_packed"],
            self.layout["eight_packed"],
            self.layout["two_starts"].numel(),
            self.layout["four_starts"].numel(),
            self.layout["eight_starts"].numel(),
            quantized_cache.numel(),
            channel_blocks=channel_blocks,
            channel_dim=self.channel_dim,
            packed_channel_dim=self.packed_channel_dim,
            BLOCK_SIZE=self.block_size,
            **launch_kwargs,
        )
        return packed, kernel

    def forward_torch(self, inputs):
        (quantized_cache,) = inputs
        packed = torch.zeros(
            *quantized_cache.shape[:2],
            self.packed_channel_dim,
            dtype=torch.uint8,
            device=quantized_cache.device,
        )
        for width, label in ((1, "one"), (2, "two"), (4, "four"), (8, "eight")):
            starts = self.layout[f"{label}_starts"]
            if not starts.numel():
                continue
            channels = starts[:, None] + torch.arange(8 // width, device=starts.device)
            values = quantized_cache[..., channels]
            shifts = self.layout["offsets"][channels]
            packed[..., self.layout[f"{label}_packed"]] = (
                (values.to(torch.int32) << shifts).sum(dim=-1).to(torch.uint8)
            )
        return packed


class ChanMixICLR2026GetCache(TritonPTXKernel):
    """Unpack and dequantize ChanMix's mixed-bit KV cache."""

    tuning_options: ClassVar[dict[str, tuple[int, ...]]] = {
        "num_warps": (1, 2, 4, 8),
        "num_stages": (1, 2, 3, 4),
    }

    def __init__(
        self,
        *,
        channel_store_bits=None,
        original_to_quantized_index=None,
        batch_size=2,
        sequence_length=128,
        token_group_size=128,
        channel_group_size=128,
        block_size=64,
        ptx=None,
    ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.token_group_size = token_group_size
        self.channel_group_size = channel_group_size
        self.block_size = block_size
        self.num_warps = 4
        self.num_stages = 2
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if channel_store_bits is None:
            channel_store_bits = torch.full((512,), 4, dtype=torch.int32)
        self.layout = _cache_layout(channel_store_bits, device)
        self.channel_dim = self.layout["bits"].numel()
        if self.channel_dim % block_size:
            raise ValueError("channel_dim must be divisible by block_size.")
        if self.channel_dim % channel_group_size:
            raise ValueError("channel_dim must be divisible by channel_group_size.")
        if original_to_quantized_index is None:
            original_to_quantized_index = torch.arange(
                self.channel_dim, device=device, dtype=torch.int32
            )
        self.original_to_quantized_index = torch.as_tensor(
            original_to_quantized_index, device=device, dtype=torch.int32
        )
        if not torch.equal(
            torch.sort(self.original_to_quantized_index).values,
            torch.arange(self.channel_dim, device=device, dtype=torch.int32),
        ):
            raise ValueError(
                "original_to_quantized_index must be a channel permutation."
            )
        self.init_compiled_kernels(ptx=ptx)

    @staticmethod
    def kernel(
        packed_cache_ptr,
        cache_ptr,
        scale_ptr,
        zero_point_ptr,
        channel_bit_ptr,
        channel_offset_bit_ptr,
        channel_o2q_map_ptr,
        channel_q2s_idx_ptr,
        token_group_size: tl.constexpr,
        channel_group_size: tl.constexpr,
        channel_group_num: tl.constexpr,
        channel_blocks: tl.constexpr,
        channel_dim: tl.constexpr,
        packed_channel_dim: tl.constexpr,
        BLOCK_SIZE_CHANNEL: tl.constexpr,
    ):
        pid = tl.program_id(0)

        # Get token_id and block_id
        token_id = pid // channel_blocks
        block_id = pid % channel_blocks

        channel_offsets = block_id * BLOCK_SIZE_CHANNEL + tl.arange(
            0, BLOCK_SIZE_CHANNEL
        )
        channel_mask = channel_offsets < channel_dim

        # Get real value from 8bit variable
        bit_width = tl.load(channel_bit_ptr + channel_offsets, mask=channel_mask)
        bit_offset = tl.load(
            channel_offset_bit_ptr + channel_offsets, mask=channel_mask
        )

        # Load packed cache
        q2s_idx = tl.load(channel_q2s_idx_ptr + channel_offsets, mask=channel_mask)
        packed_cache = tl.load(
            packed_cache_ptr + token_id * packed_channel_dim + q2s_idx
        )

        quantized_cache = (packed_cache >> bit_offset) & ((1 << bit_width) - 1)

        # Get scale and zero point value
        channel_group_offsets = channel_offsets // channel_group_size
        token_group_offsets = token_id // token_group_size
        scale = tl.load(
            scale_ptr + token_group_offsets * channel_group_num + channel_group_offsets
        )
        zero_point = tl.load(
            zero_point_ptr
            + token_group_offsets * channel_group_num
            + channel_group_offsets
        )

        # Load q2o_idx_map
        o2q_map = tl.load(channel_o2q_map_ptr + channel_offsets, mask=channel_mask)

        # Calculate and write back to cache
        tl.store(
            cache_ptr + token_id * channel_dim + o2q_map,
            quantized_cache * scale + zero_point,
        )

    def get_random_input(self, fixed=False):
        del fixed
        num_groups = (
            self.batch_size
            * triton.cdiv(self.sequence_length, self.token_group_size)
            * (self.channel_dim // self.channel_group_size)
        )
        packed = torch.randint(
            0,
            256,
            (self.batch_size, self.sequence_length, self.layout["packed_dim"]),
            dtype=torch.uint8,
            device=self.layout["bits"].device,
        )
        scale = torch.rand(num_groups, device=packed.device, dtype=torch.float32) + 0.01
        zero_point = torch.randn(num_groups, device=packed.device, dtype=torch.float32)
        return packed, scale, zero_point

    def get_shape_information(self):
        return (
            f"- packed_cache_ptr: uint8 tensor with shape ({self.batch_size}, "
            f"{self.sequence_length}, {self.layout['packed_dim']})\n"
            f"- cache_ptr: float32 tensor with shape ({self.batch_size}, "
            f"{self.sequence_length}, {self.channel_dim})\n"
            "- scale_ptr and zero_point_ptr: float32 tensors per token/channel group"
        )

    def forward_triton(self, inputs, ptx=False):
        packed, scale, zero_point = inputs
        if packed.dtype != torch.uint8 or packed.ndim != 3:
            raise ValueError("packed must be a rank-3 uint8 tensor.")
        if packed.shape[-1] != self.layout["packed_dim"]:
            raise ValueError("packed has an unexpected packed channel dimension.")
        cache = torch.empty(
            *packed.shape[:2], self.channel_dim, dtype=scale.dtype, device=packed.device
        )
        channel_blocks = self.channel_dim // self.block_size
        launch_kernel = self.compiled_kernel_ptx if ptx else self.compiled_kernel
        launch_kwargs = (
            self.ptx_launch_kwargs()
            if ptx
            else {"num_warps": self.num_warps, "num_stages": self.num_stages}
        )
        kernel = launch_kernel[(packed.shape[0] * packed.shape[1] * channel_blocks,)](
            packed,
            cache,
            scale,
            zero_point,
            self.layout["bits"],
            self.layout["offsets"],
            self.original_to_quantized_index,
            self.layout["packed_indices"],
            token_group_size=self.token_group_size,
            channel_group_size=self.channel_group_size,
            channel_group_num=self.channel_dim // self.channel_group_size,
            channel_blocks=channel_blocks,
            channel_dim=self.channel_dim,
            packed_channel_dim=self.layout["packed_dim"],
            BLOCK_SIZE_CHANNEL=self.block_size,
            **launch_kwargs,
        )
        return cache, kernel

    def forward_torch(self, inputs):
        packed, scale, zero_point = inputs
        channels = torch.arange(self.channel_dim, device=packed.device)
        quantized = (
            packed[..., self.layout["packed_indices"]].to(torch.int32)
            >> self.layout["offsets"]
        ) & ((1 << self.layout["bits"]) - 1)
        group = (
            torch.arange(packed.shape[1], device=packed.device)[:, None]
            // self.token_group_size
            * (self.channel_dim // self.channel_group_size)
            + channels[None, :] // self.channel_group_size
        )
        values = (
            quantized.to(scale.dtype) * scale.view(packed.shape[0], -1)[:, group]
            + zero_point.view(packed.shape[0], -1)[:, group]
        )
        cache = torch.empty_like(values)
        cache[..., self.original_to_quantized_index] = values
        return cache
