import math

import torch
from safetensors import safe_open
from transformers import GemmaTokenizerFast

from ..llm import LLM
from .attention import GemmaAttentionKernel
from .gelu import GemmaGELUKernel
from .linear import GemmaLinearKernel
from .rms_norm import GemmaRMSNormKernel
from .rope import GemmaRoPEKernel

MODEL_ID = "google/gemma-4-E4B-it"
MODEL_CACHE_DIR = "/matx/u/franc141/huggingface-cache"
VOCAB_SIZE = 262_144
HIDDEN_SIZE = 2_560
INTERMEDIATE_SIZE = 10_240
NUM_LAYERS = 42
HEAD_DIM = 256
LOCAL_ATTENTION_WINDOW = 512
MODEL_WEIGHTS_PATH = (
    "/matx/u/franc141/huggingface-cache/models--google--gemma-4-E4B-it/"
    "snapshots/ee0ef6023621cff504d758262d4e04895a5af4a2/model.safetensors"
)
GEMMA_KERNEL_CLASSES = {
    "attention": GemmaAttentionKernel,
    "gelu": GemmaGELUKernel,
    "linear": GemmaLinearKernel,
    "rms_norm": GemmaRMSNormKernel,
    "rope": GemmaRoPEKernel,
}


class GemmaKernelSet:
    def __init__(
        self,
        *,
        attention_payload=None,
        gelu_payload=None,
        linear_payload=None,
        rms_norm_payload=None,
        rope_payload=None,
    ):
        payloads = {
            "attention": attention_payload,
            "gelu": gelu_payload,
            "linear": linear_payload,
            "rms_norm": rms_norm_payload,
            "rope": rope_payload,
        }
        for name, kernel_class in GEMMA_KERNEL_CLASSES.items():
            payload = payloads[name]
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            setattr(self, name, kernel_class(ptx=payload))

    @staticmethod
    def uses_custom_ptx(kernel):
        return kernel.compiled_kernel_ptx is not None


class Gemma4GELU(torch.nn.Module):
    """Gemma's GELU activation backed by the custom Triton kernel."""

    def __init__(self, kernels):
        super().__init__()
        self.kernel = kernels.gelu

    def forward(self, hidden_states):
        output, _ = self.kernel.forward_triton(
            hidden_states,
            ptx=GemmaKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class Gemma4Attention(torch.nn.Module):
    """Gemma attention backed by the custom Triton kernel."""

    def __init__(self, kernels):
        super().__init__()
        self.kernel = kernels.attention

    def forward(self, query_states, key_states, value_states, window_size):
        output, _ = self.kernel.forward_triton(
            (
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                window_size,
            ),
            ptx=GemmaKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class Gemma4Linear(torch.nn.Module):
    """Bias-free Gemma projection backed by the custom Triton kernel."""

    def __init__(
        self,
        input_features,
        output_features,
        device=None,
        dtype=None,
        kernels=None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(output_features, input_features, device=device, dtype=dtype)
        )
        self.kernel = (kernels or GemmaKernelSet()).linear

    def forward(self, hidden_states):
        output, _ = self.kernel.forward_triton(
            (hidden_states, self.weight),
            ptx=GemmaKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class Gemma4RMSNorm(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        eps=1e-6,
        with_scale=True,
        device=None,
        dtype=None,
        kernels=None,
    ):
        super().__init__()
        self.weight = (
            torch.nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
            if with_scale
            else None
        )
        self.eps = eps
        self.kernel = (kernels or GemmaKernelSet()).rms_norm

    def forward(self, hidden_states):
        if self.weight is not None:
            hidden_states = hidden_states.contiguous()
            output, _ = self.kernel.forward_triton(
                (hidden_states, self.weight, self.eps),
                ptx=GemmaKernelSet.uses_custom_ptx(self.kernel),
            )
            return output
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().square().mean(dim=-1, keepdim=True)
        hidden_states = (hidden_states * torch.rsqrt(variance + self.eps)).to(
            input_dtype
        )
        return hidden_states


class Gemma4TextScaledWordEmbedding(torch.nn.Embedding):
    def __init__(
        self, num_embeddings, embedding_dim, embed_scale, device=None, dtype=None
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=0,
            device=device,
            dtype=dtype,
        )
        self.scale = embed_scale

    def forward(self, input_ids):
        return super().forward(input_ids) * self.scale


class Gemma4TextRotaryEmbedding(torch.nn.Module):
    def __init__(self, device=None, kernels=None):
        super().__init__()
        self.device = device
        self.kernel = (kernels or GemmaKernelSet()).rope

    def forward(
        self, sequence_length, head_dim, is_local, device, dtype, position_offset=0
    ):
        base = 10_000 if is_local else 1_000_000
        inverse_frequencies = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
        )
        positions = torch.arange(
            position_offset, position_offset + sequence_length, device=device
        )
        frequencies = torch.outer(positions, inverse_frequencies)
        angles = torch.cat((frequencies, frequencies), dim=-1)
        return angles.cos().to(dtype), angles.sin().to(dtype)

    def apply(self, hidden_states, cos, sin):
        output, _ = self.kernel.forward_triton(
            (hidden_states.contiguous(), cos, sin),
            ptx=GemmaKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class Gemma4TextAttention(torch.nn.Module):
    def __init__(self, layer_index, device, dtype, kernels):
        super().__init__()
        self.layer_index = layer_index
        self.is_local = (layer_index + 1) % 6 != 0
        self.is_kv_shared_layer = layer_index >= NUM_LAYERS - 18
        self.stores_shared_kv = layer_index in (22, 23)
        self.num_heads = 8
        self.num_key_value_heads = 2
        self.head_dim = HEAD_DIM if self.is_local else 512
        self.q_proj = Gemma4Linear(
            HIDDEN_SIZE,
            self.num_heads * self.head_dim,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )
        self.q_norm = Gemma4RMSNorm(
            self.head_dim,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )
        if not self.is_kv_shared_layer:
            self.k_norm = Gemma4RMSNorm(
                self.head_dim,
                device=device,
                dtype=dtype,
                kernels=kernels,
            )
            self.v_norm = Gemma4RMSNorm(
                self.head_dim,
                with_scale=False,
                device=device,
                dtype=dtype,
                kernels=kernels,
            )
            self.k_proj = Gemma4Linear(
                HIDDEN_SIZE,
                self.num_key_value_heads * self.head_dim,
                device=device,
                dtype=dtype,
                kernels=kernels,
            )
            self.v_proj = Gemma4Linear(
                HIDDEN_SIZE,
                self.num_key_value_heads * self.head_dim,
                device=device,
                dtype=dtype,
                kernels=kernels,
            )
        self.o_proj = Gemma4Linear(
            self.num_heads * self.head_dim,
            HIDDEN_SIZE,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )
        self.attention = Gemma4Attention(kernels)

    def forward(
        self, hidden_states, rotary_embedding, shared_kv_states, past_key_value=None
    ):
        batch_size, sequence_length, _ = hidden_states.shape
        past_length = 0 if past_key_value is None else past_key_value[0].shape[2]
        query_states = (
            self.q_proj(hidden_states)
            .view(batch_size, sequence_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        query_states = self.q_norm(query_states)

        cos, sin = rotary_embedding(
            sequence_length,
            self.head_dim,
            self.is_local,
            hidden_states.device,
            hidden_states.dtype,
            past_length,
        )
        query_states = rotary_embedding.apply(query_states, cos, sin)
        if self.is_kv_shared_layer:
            key_states, value_states = shared_kv_states[self.is_local]
        else:
            key_states = (
                self.k_proj(hidden_states)
                .view(
                    batch_size, sequence_length, self.num_key_value_heads, self.head_dim
                )
                .transpose(1, 2)
            )
            value_states = (
                self.v_proj(hidden_states)
                .view(
                    batch_size, sequence_length, self.num_key_value_heads, self.head_dim
                )
                .transpose(1, 2)
            )
            key_states = rotary_embedding.apply(self.k_norm(key_states), cos, sin)
            value_states = self.v_norm(value_states)
            if past_key_value is not None:
                past_key_states, past_value_states = past_key_value
                key_states = torch.cat((past_key_states, key_states), dim=2)
                value_states = torch.cat((past_value_states, value_states), dim=2)
            if self.stores_shared_kv:
                shared_kv_states[self.is_local] = (key_states, value_states)
        key_value = (key_states, value_states)
        repeat_factor = self.num_heads // self.num_key_value_heads
        key_states = key_states.repeat_interleave(repeat_factor, dim=1)
        value_states = value_states.repeat_interleave(repeat_factor, dim=1)
        attention_output = self.attention(
            query_states,
            key_states,
            value_states,
            LOCAL_ATTENTION_WINDOW if self.is_local else 0,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size, sequence_length, -1
        )
        return self.o_proj(attention_output), key_value


class Gemma4TextMLP(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.gate_proj = Gemma4Linear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )
        self.up_proj = Gemma4Linear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )
        self.down_proj = Gemma4Linear(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )
        self.act_fn = Gemma4GELU(kernels)

    def forward(self, hidden_states):
        gate = self.act_fn(self.gate_proj(hidden_states))
        return self.down_proj(gate * self.up_proj(hidden_states))


class Gemma4TextDecoderLayer(torch.nn.Module):
    def __init__(self, layer_index, device, dtype, kernels):
        super().__init__()
        self.layer_index = layer_index
        self.self_attn = Gemma4TextAttention(layer_index, device, dtype, kernels)
        self.mlp = Gemma4TextMLP(device, dtype, kernels)
        self.input_layernorm = Gemma4RMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=kernels
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=kernels
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=kernels
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=kernels
        )
        self.per_layer_input_gate = Gemma4Linear(
            HIDDEN_SIZE, 256, device=device, dtype=dtype, kernels=kernels
        )
        self.per_layer_projection = Gemma4Linear(
            256, HIDDEN_SIZE, device=device, dtype=dtype, kernels=kernels
        )
        self.post_per_layer_input_norm = Gemma4RMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=kernels
        )
        self.act_fn = Gemma4GELU(kernels)
        self.register_buffer("layer_scalar", torch.ones(1, device=device, dtype=dtype))

    def forward(
        self,
        hidden_states,
        layer_embedding,
        rotary_embedding,
        shared_kv_states,
        past_key_value=None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, key_value = self.self_attn(
            hidden_states, rotary_embedding, shared_kv_states, past_key_value
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        layer_input = self.act_fn(self.per_layer_input_gate(hidden_states))
        layer_input = layer_input * layer_embedding
        hidden_states = hidden_states + self.post_per_layer_input_norm(
            self.per_layer_projection(layer_input)
        )
        return hidden_states * self.layer_scalar, key_value


class Gemma4TextModel(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            VOCAB_SIZE, HIDDEN_SIZE, math.sqrt(HIDDEN_SIZE), device=device, dtype=dtype
        )
        self.layers = torch.nn.ModuleList(
            Gemma4TextDecoderLayer(layer_index, device, dtype, kernels)
            for layer_index in range(NUM_LAYERS)
        )
        self.norm = Gemma4RMSNorm(
            HIDDEN_SIZE, device=device, dtype=dtype, kernels=kernels
        )
        self.rotary_emb = Gemma4TextRotaryEmbedding(device=device, kernels=kernels)
        self.embed_tokens_per_layer = Gemma4TextScaledWordEmbedding(
            VOCAB_SIZE, NUM_LAYERS * 256, math.sqrt(256), device=device, dtype=dtype
        )
        self.per_layer_model_projection = Gemma4Linear(
            HIDDEN_SIZE,
            NUM_LAYERS * 256,
            device=device,
            dtype=dtype,
            kernels=kernels,
        )
        self.per_layer_projection_norm = Gemma4RMSNorm(
            256, device=device, dtype=dtype, kernels=kernels
        )

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        if past_key_values is not None and len(past_key_values) != NUM_LAYERS:
            raise ValueError("past_key_values must contain one entry per layer.")
        hidden_states = self.embed_tokens(input_ids)
        per_layer_embeddings = self.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape, NUM_LAYERS, 256
        )
        per_layer_projection = self.per_layer_model_projection(hidden_states)
        per_layer_projection = per_layer_projection * HIDDEN_SIZE**-0.5
        per_layer_projection = self.per_layer_projection_norm(
            per_layer_projection.reshape(*input_ids.shape, NUM_LAYERS, 256)
        )
        per_layer_embeddings = (per_layer_embeddings + per_layer_projection) * 2**-0.5
        shared_kv_states = {}
        next_key_values = [] if use_cache else None
        for layer_index, layer in enumerate(self.layers):
            layer_embedding = per_layer_embeddings[:, :, layer_index, :]
            cache_index = (
                22
                if layer.self_attn.is_kv_shared_layer and layer.self_attn.is_local
                else 23
                if layer.self_attn.is_kv_shared_layer
                else layer_index
            )
            past_key_value = (
                None if past_key_values is None else past_key_values[cache_index]
            )
            hidden_states, key_value = layer(
                hidden_states,
                layer_embedding,
                self.rotary_emb,
                shared_kv_states,
                past_key_value,
            )
            if use_cache:
                next_key_values.append(key_value)
        hidden_states = self.norm(hidden_states)
        if use_cache:
            return hidden_states, tuple(next_key_values)
        return hidden_states


class Gemma4ForConditionalGeneration(torch.nn.Module):
    def __init__(self, device, dtype=torch.bfloat16, kernels=None):
        super().__init__()
        self.kernels = kernels or GemmaKernelSet()
        self.model = Gemma4TextModel(device, dtype, self.kernels)
        self.lm_head = Gemma4Linear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            device=device,
            dtype=dtype,
            kernels=self.kernels,
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        model_output = self.model(input_ids, past_key_values, use_cache)
        if use_cache:
            hidden_states, next_key_values = model_output
            logits = self.lm_head(hidden_states)
            return torch.tanh(logits / 30) * 30, next_key_values
        logits = self.lm_head(model_output)
        return torch.tanh(logits / 30) * 30

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=32):
        end_token_ids = torch.tensor((1, 106), device=input_ids.device)
        logits, past_key_values = self(input_ids, use_cache=True)
        generated_tokens = []
        for token_index in range(max_new_tokens):
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            if torch.isin(next_token, end_token_ids).all():
                break
            if token_index + 1 < max_new_tokens:
                logits, past_key_values = self(
                    next_token, past_key_values=past_key_values, use_cache=True
                )
        return torch.cat((input_ids, *generated_tokens), dim=-1)


def load_pretrained_text_model(device, kernels=None):
    model = Gemma4ForConditionalGeneration("meta", kernels=kernels)
    model.to_empty(device=device)
    model_state = model.state_dict()
    loaded_keys = 0
    with (
        torch.no_grad(),
        safe_open(MODEL_WEIGHTS_PATH, framework="pt", device="cpu") as checkpoint,
    ):
        checkpoint_keys = set(checkpoint.keys())
        for model_key, parameter in model_state.items():
            if model_key == "lm_head.weight":
                continue
            checkpoint_key = "model.language_model." + model_key.removeprefix("model.")
            if checkpoint_key not in checkpoint_keys:
                raise KeyError(f"Missing checkpoint tensor: {checkpoint_key}")
            tensor = checkpoint.get_tensor(checkpoint_key)
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"Shape mismatch for {checkpoint_key}: "
                    f"{tensor.shape} != {parameter.shape}"
                )
            parameter.copy_(tensor.to(device=device, dtype=parameter.dtype))
            loaded_keys += 1
    model.lm_head.weight = model.model.embed_tokens.weight
    print(f"Loaded {loaded_keys} text-model tensors from the checkpoint.")
    return model.eval()


class GemmaLLM(LLM):
    """Gemma 4 text-generation backend."""

    def __init__(
        self,
        *,
        attention_payload=None,
        gelu_payload=None,
        linear_payload=None,
        rms_norm_payload=None,
        rope_payload=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Gemma text inference requires a CUDA GPU.")
        self.kernels = GemmaKernelSet(
            attention_payload=attention_payload,
            gelu_payload=gelu_payload,
            linear_payload=linear_payload,
            rms_norm_payload=rms_norm_payload,
            rope_payload=rope_payload,
        )
        self.tokenizer = GemmaTokenizerFast.from_pretrained(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
        )
        self.model = load_pretrained_text_model("cuda", self.kernels)

    @classmethod
    def from_custom_ptx(cls, **payloads):
        return cls(**payloads)

    @classmethod
    def get_kernel_classes(cls):
        return GEMMA_KERNEL_CLASSES

    def tokenize_messages(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
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
