"""Apertus v1.5 model implementation backed by Triton kernels."""

import json
import math
from contextlib import ExitStack
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from ..llm import LLM
from .causal_attention import CausalAttentionKernel
from .linear import LinearKernel
from .rms_norm import ApertusRMSNormKernel
from .rope import RoPEKernel
from .xielu import XIELUKernel

MODEL_ID = "swiss-ai/Apertus-v1.5-8B"
MODEL_CACHE_DIR = "/matx/u/franc141/huggingface-cache"
MODEL_WEIGHTS_DIR = Path(
    "/matx/u/franc141/huggingface-cache/models--swiss-ai--Apertus-v1.5-8B/"
    "snapshots/a411d838600baf0e3635a3daf66fb7c55fc97bb6"
)
VOCAB_SIZE = 266_752
OUTPUT_VOCAB_SIZE = 131_072
HIDDEN_SIZE = 4_096
INTERMEDIATE_SIZE = 21_504
NUM_LAYERS = 32
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
RMS_NORM_EPS = 1e-5
ROPE_THETA = 4_000_000
ROPE_FACTOR = 32.0
ROPE_LOW_FREQ_FACTOR = 1.0
ROPE_HIGH_FREQ_FACTOR = 4.0
ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS = 8_192
EOS_TOKEN_IDS = (2, 68, 72)
RMS_NORM_BLOCK_SIZE = 128
XIELU_BLOCK_SIZE = 256
ROPE_BLOCK_SIZE = 128
ATTENTION_BLOCK_M = 32
ATTENTION_BLOCK_N = 64
APERTUS_KERNEL_CLASSES = {
    "causal_attention": CausalAttentionKernel,
    "linear": LinearKernel,
    "rms_norm": ApertusRMSNormKernel,
    "rope": RoPEKernel,
    "xielu": XIELUKernel,
}


class ApertusKernelSet:
    """Apertus operators, optionally backed by custom PTX payloads."""

    def __init__(
        self,
        *,
        causal_attention_payload=None,
        linear_payload=None,
        rms_norm_payload=None,
        rope_payload=None,
        xielu_payload=None,
    ):
        self.causal_attention_payload = self.normalize_payload(
            causal_attention_payload
        )
        self.linear_payload = self.normalize_payload(linear_payload)
        self.rms_norm_payload = self.normalize_payload(rms_norm_payload)
        self.rope_payload = self.normalize_payload(rope_payload)
        self.xielu_payload = self.normalize_payload(xielu_payload)

        self.causal_attention = CausalAttentionKernel(
            ptx=self.causal_attention_payload
        )
        self.linear = LinearKernel(ptx=self.linear_payload)
        self.rms_norm = ApertusRMSNormKernel(ptx=self.rms_norm_payload)
        self.rope = RoPEKernel(ptx=self.rope_payload)
        self.xielu = XIELUKernel(ptx=self.xielu_payload)

    @staticmethod
    def normalize_payload(payload):
        if hasattr(payload, "model_dump"):
            return payload.model_dump()
        return payload

    @staticmethod
    def uses_custom_ptx(kernel):
        return kernel.compiled_kernel_ptx is not None


class Apertus1p5TextRMSNorm(torch.nn.Module):
    def __init__(
        self, hidden_size, eps=RMS_NORM_EPS, device=None, dtype=None, kernels=None
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.eps = eps
        self.kernel = (kernels or ApertusKernelSet()).rms_norm

    def forward(self, hidden_states):
        output, _ = self.kernel.forward_triton(
            (hidden_states, self.weight, self.eps),
            ptx=ApertusKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class XIELUActivation(torch.nn.Module):
    def __init__(self, device=None, dtype=None, kernels=None):
        super().__init__()
        self.alpha_p = torch.nn.Parameter(
            torch.log(
                torch.expm1(torch.tensor(0.8, device=device, dtype=dtype))
            ).unsqueeze(0)
        )
        self.alpha_n = torch.nn.Parameter(
            torch.log(
                torch.expm1(torch.tensor(0.3, device=device, dtype=dtype))
            ).unsqueeze(0)
        )
        self.register_buffer("beta", torch.tensor(0.5, device=device, dtype=dtype))
        self.register_buffer("eps", torch.tensor(-1e-6, device=device, dtype=dtype))
        self.kernel = (kernels or ApertusKernelSet()).xielu

    def forward(self, hidden_states):
        output, _ = self.kernel.forward_triton(
            (hidden_states, self.alpha_p, self.alpha_n, self.beta, self.eps),
            ptx=ApertusKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class Apertus1p5TextRotaryEmbedding(torch.nn.Module):
    def __init__(self, device=None):
        super().__init__()
        inverse_frequencies = 1.0 / (
            ROPE_THETA
            ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM)
        )
        wavelengths = 2 * math.pi / inverse_frequencies
        low_frequency_wavelength = (
            ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS / ROPE_LOW_FREQ_FACTOR
        )
        high_frequency_wavelength = (
            ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS / ROPE_HIGH_FREQ_FACTOR
        )
        scaled_frequencies = torch.where(
            wavelengths > low_frequency_wavelength,
            inverse_frequencies / ROPE_FACTOR,
            inverse_frequencies,
        )
        smooth_factor = (
            ROPE_ORIGINAL_MAX_POSITION_EMBEDDINGS / wavelengths - ROPE_LOW_FREQ_FACTOR
        ) / (ROPE_HIGH_FREQ_FACTOR - ROPE_LOW_FREQ_FACTOR)
        smoothed_frequencies = (
            1 - smooth_factor
        ) * scaled_frequencies / ROPE_FACTOR + smooth_factor * scaled_frequencies
        is_medium_frequency = (wavelengths >= high_frequency_wavelength) & (
            wavelengths <= low_frequency_wavelength
        )
        self.register_buffer(
            "inverse_frequencies",
            torch.where(is_medium_frequency, smoothed_frequencies, scaled_frequencies),
            persistent=False,
        )

    def forward(self, hidden_states, position_ids):
        frequencies = torch.outer(
            position_ids.float(), self.inverse_frequencies.float()
        )
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return (
            angles.cos().to(hidden_states.dtype),
            angles.sin().to(hidden_states.dtype),
        )


class Apertus1p5TextAttention(torch.nn.Module):
    def __init__(self, device, dtype, kernels=None):
        super().__init__()
        self.kernels = kernels or ApertusKernelSet()
        self.q_proj = torch.nn.Linear(
            HIDDEN_SIZE, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )
        self.k_proj = torch.nn.Linear(
            HIDDEN_SIZE,
            NUM_KEY_VALUE_HEADS * HEAD_DIM,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.v_proj = torch.nn.Linear(
            HIDDEN_SIZE,
            NUM_KEY_VALUE_HEADS * HEAD_DIM,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.o_proj = torch.nn.Linear(
            HIDDEN_SIZE, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )
        self.q_norm = Apertus1p5TextRMSNorm(
            HEAD_DIM, device=device, dtype=dtype, kernels=self.kernels
        )
        self.k_norm = Apertus1p5TextRMSNorm(
            HEAD_DIM, device=device, dtype=dtype, kernels=self.kernels
        )

    def forward(self, hidden_states, cos, sin):
        batch_size, sequence_length, _ = hidden_states.shape
        query_states = (
            self.q_proj(hidden_states)
            .view(batch_size, sequence_length, NUM_ATTENTION_HEADS, HEAD_DIM)
            .transpose(1, 2)
            .contiguous()
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(batch_size, sequence_length, NUM_KEY_VALUE_HEADS, HEAD_DIM)
            .transpose(1, 2)
            .contiguous()
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(batch_size, sequence_length, NUM_KEY_VALUE_HEADS, HEAD_DIM)
            .transpose(1, 2)
            .contiguous()
        )
        query_states, _ = self.kernels.rope.forward_triton(
            (self.q_norm(query_states), cos, sin),
            ptx=ApertusKernelSet.uses_custom_ptx(self.kernels.rope),
        )
        key_states, _ = self.kernels.rope.forward_triton(
            (self.k_norm(key_states), cos, sin),
            ptx=ApertusKernelSet.uses_custom_ptx(self.kernels.rope),
        )
        repeat_factor = NUM_ATTENTION_HEADS // NUM_KEY_VALUE_HEADS
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
        value_states = value_states.repeat_interleave(repeat_factor, dim=1)
        attention_output, _ = self.kernels.causal_attention.forward_triton(
            (query_states, key_states, value_states),
            ptx=ApertusKernelSet.uses_custom_ptx(self.kernels.causal_attention),
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, HIDDEN_SIZE
        )
        return self.o_proj(attention_output)


class Apertus1p5TextMLP(torch.nn.Module):
    def __init__(self, device, dtype, kernels=None):
        super().__init__()
        self.kernels = kernels or ApertusKernelSet()
        self.up_proj = torch.nn.Linear(
            HIDDEN_SIZE, INTERMEDIATE_SIZE, bias=False, device=device, dtype=dtype
        )
        self.down_proj = torch.nn.Linear(
            INTERMEDIATE_SIZE, HIDDEN_SIZE, bias=False, device=device, dtype=dtype
        )
        self.act_fn = XIELUActivation(device=device, dtype=dtype, kernels=self.kernels)

    def forward(self, hidden_states):
        hidden_states, _ = self.kernels.linear.forward_triton(
            (hidden_states, self.up_proj.weight),
            ptx=ApertusKernelSet.uses_custom_ptx(self.kernels.linear),
        )
        hidden_states = self.act_fn(hidden_states)
        output, _ = self.kernels.linear.forward_triton(
            (hidden_states, self.down_proj.weight),
            ptx=ApertusKernelSet.uses_custom_ptx(self.kernels.linear),
        )
        return output


class Apertus1p5TextDecoderLayer(torch.nn.Module):
    def __init__(self, device, dtype, kernels=None):
        super().__init__()
        self.kernels = kernels or ApertusKernelSet()
        self.self_attn = Apertus1p5TextAttention(device, dtype, self.kernels)
        self.mlp = Apertus1p5TextMLP(device, dtype, self.kernels)
        self.attention_layernorm = Apertus1p5TextRMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=self.kernels
        )
        self.feedforward_layernorm = Apertus1p5TextRMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=self.kernels
        )

    def forward(self, hidden_states, cos, sin):
        hidden_states = hidden_states + self.self_attn(
            self.attention_layernorm(hidden_states), cos, sin
        )
        return hidden_states + self.mlp(self.feedforward_layernorm(hidden_states))


class Apertus1p5TextModel(torch.nn.Module):
    def __init__(self, device, dtype, kernels=None):
        super().__init__()
        self.kernels = kernels or ApertusKernelSet()
        self.embed_tokens = torch.nn.Embedding(
            VOCAB_SIZE,
            HIDDEN_SIZE,
            padding_idx=3,
            device=device,
            dtype=dtype,
        )
        self.layers = torch.nn.ModuleList(
            Apertus1p5TextDecoderLayer(device, dtype, self.kernels)
            for _ in range(NUM_LAYERS)
        )
        self.norm = Apertus1p5TextRMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=self.kernels
        )
        self.rotary_emb = Apertus1p5TextRotaryEmbedding(device=device)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        position_ids = torch.arange(input_ids.shape[-1], device=input_ids.device)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin)
        return self.norm(hidden_states)


class Apertus1p5TextForCausalLM(torch.nn.Module):
    def __init__(self, device, dtype=torch.bfloat16, kernels=None):
        super().__init__()
        self.kernels = kernels or ApertusKernelSet()
        self.model = Apertus1p5TextModel(device, dtype, self.kernels)
        self.lm_head = torch.nn.Linear(
            HIDDEN_SIZE,
            OUTPUT_VOCAB_SIZE,
            bias=False,
            device=device,
            dtype=dtype,
        )

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=32):
        end_token_ids = torch.tensor(EOS_TOKEN_IDS, device=input_ids.device)
        for _ in range(max_new_tokens):
            next_token = self(input_ids)[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if torch.isin(next_token, end_token_ids).all():
                break
        return input_ids


class ApertusLLM(LLM):
    @classmethod
    def get_kernel_classes(cls):
        return APERTUS_KERNEL_CLASSES

    def __init__(
        self,
        *,
        causal_attention_payload=None,
        linear_payload=None,
        rms_norm_payload=None,
        rope_payload=None,
        xielu_payload=None,
    ):
        assert torch.cuda.is_available()
        self.kernels = ApertusKernelSet(
            causal_attention_payload=causal_attention_payload,
            linear_payload=linear_payload,
            rms_norm_payload=rms_norm_payload,
            rope_payload=rope_payload,
            xielu_payload=xielu_payload,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
        )
        self.model = self.load_pretrained_text_model(self.kernels)

    @classmethod
    def from_custom_ptx(
        cls,
        *,
        causal_attention_payload=None,
        linear_payload=None,
        rms_norm_payload=None,
        rope_payload=None,
        xielu_payload=None,
    ):
        """Load Apertus with explicitly named custom PTX payloads."""
        return cls(
            causal_attention_payload=causal_attention_payload,
            linear_payload=linear_payload,
            rms_norm_payload=rms_norm_payload,
            rope_payload=rope_payload,
            xielu_payload=xielu_payload,
        )

    def load_pretrained_text_model(self, kernels):
        model = Apertus1p5TextForCausalLM("meta", kernels=kernels)
        model.to_empty(device="cuda")
        model.model.rotary_emb = Apertus1p5TextRotaryEmbedding("cuda")
        index_path = MODEL_WEIGHTS_DIR / "model.safetensors.index.json"
        weight_map = json.loads(index_path.read_text())["weight_map"]
        model_state = model.state_dict()
        loaded_keys = 0

        with torch.no_grad(), ExitStack() as stack:
            checkpoints = {
                shard: stack.enter_context(
                    safe_open(MODEL_WEIGHTS_DIR / shard, framework="pt", device="cpu")
                )
                for shard in set(weight_map.values())
            }
            for model_key, parameter in model_state.items():
                checkpoint_key = "model.language_model." + model_key.removeprefix(
                    "model."
                )
                if model_key == "lm_head.weight":
                    checkpoint_key = model_key
                checkpoint_shard = weight_map.get(checkpoint_key)
                if checkpoint_shard is None:
                    raise KeyError(f"Missing checkpoint tensor: {checkpoint_key}")
                tensor = checkpoints[checkpoint_shard].get_tensor(checkpoint_key)
                if tensor.shape != parameter.shape:
                    raise ValueError(
                        f"Shape mismatch for {checkpoint_key}: {tensor.shape} != "
                        f"{parameter.shape}"
                    )
                parameter.copy_(tensor.to("cuda", dtype=parameter.dtype))
                loaded_keys += 1

        print(f"Loaded {loaded_keys} text-model tensors from the checkpoint.")
        return model.eval()

    def tokenize_messages(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        return input_ids.to("cuda")

    @torch.inference_mode()
    def generate_tokens(self, input_ids, max_new_tokens):
        return self.model.generate(input_ids, max_new_tokens=max_new_tokens)

    def decode_response(self, output_ids, prompt_length):
        return self.tokenizer.decode(
            output_ids[0, prompt_length:],
            skip_special_tokens=True,
        )
