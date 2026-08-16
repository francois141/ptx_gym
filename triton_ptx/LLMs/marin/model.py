import json
import logging
from contextlib import ExitStack
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import PreTrainedTokenizerFast

from ..llm import LLM
from .add import MarinAddKernel
from .causal_attention import MarinCausalAttentionKernel
from .embedding import MarinEmbeddingKernel
from .linear import MarinLinearKernel
from .rms_norm import MarinRMSNormKernel
from .rope import MarinRoPEKernel
from .swiglu import MarinSwiGLUKernel

LOGGER = logging.getLogger(__name__)
MODEL_ID = "marin-community/marin-8b-base"
MODEL_CACHE_DIR = "/matx/u/franc141/huggingface-cache"
VOCAB_SIZE = 128_256
HIDDEN_SIZE = 4_096
INTERMEDIATE_SIZE = 14_336
NUM_LAYERS = 32
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
RMS_NORM_EPS = 1e-5
EOS_TOKEN_ID = 128_001
MARIN_KERNEL_CLASSES = {
    "add": MarinAddKernel,
    "causal_attention": MarinCausalAttentionKernel,
    "embedding": MarinEmbeddingKernel,
    "linear": MarinLinearKernel,
    "rms_norm": MarinRMSNormKernel,
    "rope": MarinRoPEKernel,
    "swiglu": MarinSwiGLUKernel,
}


class MarinKernelSet:
    def __init__(
        self,
        *,
        add_payload=None,
        causal_attention_payload=None,
        embedding_payload=None,
        linear_payload=None,
        rms_norm_payload=None,
        rope_payload=None,
        swiglu_payload=None,
    ):
        payloads = {
            "add": add_payload,
            "causal_attention": causal_attention_payload,
            "embedding": embedding_payload,
            "linear": linear_payload,
            "rms_norm": rms_norm_payload,
            "rope": rope_payload,
            "swiglu": swiglu_payload,
        }
        for name, kernel_class in MARIN_KERNEL_CLASSES.items():
            payload = payloads[name]
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            setattr(self, name, kernel_class(ptx=payload))

    @staticmethod
    def uses_custom_ptx(kernel):
        return kernel.compiled_kernel_ptx is not None


class MarinLinear(torch.nn.Module):
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
            torch.empty(
                output_features,
                input_features,
                device=device,
                dtype=dtype,
            )
        )
        self.kernel = (kernels or MarinKernelSet()).linear

    def forward(self, hidden_states):
        output, _ = self.kernel.forward_triton(
            (hidden_states, self.weight),
            ptx=MarinKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class MarinEmbedding(torch.nn.Module):
    def __init__(self, device=None, dtype=None, kernels=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(VOCAB_SIZE, HIDDEN_SIZE, device=device, dtype=dtype)
        )
        self.kernel = (kernels or MarinKernelSet()).embedding

    def forward(self, input_ids):
        output, _ = self.kernel.forward_triton(
            (input_ids, self.weight),
            ptx=MarinKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class MarinRMSNorm(torch.nn.Module):
    def __init__(self, device=None, dtype=None, kernels=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(HIDDEN_SIZE, device=device, dtype=dtype)
        )
        self.kernel = (kernels or MarinKernelSet()).rms_norm

    def forward(self, hidden_states):
        output, _ = self.kernel.forward_triton(
            (hidden_states, self.weight, RMS_NORM_EPS),
            ptx=MarinKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class MarinAttention(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.kernels = kernels
        self.q_proj = MarinLinear(
            HIDDEN_SIZE,
            NUM_ATTENTION_HEADS * HEAD_DIM,
            device,
            dtype,
            kernels,
        )
        self.k_proj = MarinLinear(
            HIDDEN_SIZE,
            NUM_KEY_VALUE_HEADS * HEAD_DIM,
            device,
            dtype,
            kernels,
        )
        self.v_proj = MarinLinear(
            HIDDEN_SIZE,
            NUM_KEY_VALUE_HEADS * HEAD_DIM,
            device,
            dtype,
            kernels,
        )
        self.o_proj = MarinLinear(
            NUM_ATTENTION_HEADS * HEAD_DIM,
            HIDDEN_SIZE,
            device,
            dtype,
            kernels,
        )

    def forward(self, hidden_states, past_key_value=None):
        batch_size, sequence_length, _ = hidden_states.shape
        query_states = (
            self.q_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                NUM_ATTENTION_HEADS,
                HEAD_DIM,
            )
            .transpose(1, 2)
            .contiguous()
        )
        key_states = (
            self.k_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                NUM_KEY_VALUE_HEADS,
                HEAD_DIM,
            )
            .transpose(1, 2)
            .contiguous()
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(
                batch_size,
                sequence_length,
                NUM_KEY_VALUE_HEADS,
                HEAD_DIM,
            )
            .transpose(1, 2)
            .contiguous()
        )
        query_states, _ = self.kernels.rope.forward_triton(
            query_states,
            ptx=MarinKernelSet.uses_custom_ptx(self.kernels.rope),
            position_offset=0 if past_key_value is None else past_key_value[0].shape[2],
        )
        key_states, _ = self.kernels.rope.forward_triton(
            key_states,
            ptx=MarinKernelSet.uses_custom_ptx(self.kernels.rope),
            position_offset=0 if past_key_value is None else past_key_value[0].shape[2],
        )
        if past_key_value is not None:
            past_key_states, past_value_states = past_key_value
            key_states = torch.cat((past_key_states, key_states), dim=2)
            value_states = torch.cat((past_value_states, value_states), dim=2)
        attention_output, _ = self.kernels.causal_attention.forward_triton(
            (query_states, key_states, value_states),
            ptx=MarinKernelSet.uses_custom_ptx(self.kernels.causal_attention),
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            NUM_ATTENTION_HEADS * HEAD_DIM,
        )
        return self.o_proj(attention_output.contiguous()), (key_states, value_states)


class MarinMLP(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.kernels = kernels
        self.gate_proj = MarinLinear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device,
            dtype,
            kernels,
        )
        self.up_proj = MarinLinear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device,
            dtype,
            kernels,
        )
        self.down_proj = MarinLinear(
            INTERMEDIATE_SIZE,
            HIDDEN_SIZE,
            device,
            dtype,
            kernels,
        )

    def forward(self, hidden_states):
        gate = self.gate_proj(hidden_states)
        up = self.up_proj(hidden_states)
        hidden_states, _ = self.kernels.swiglu.forward_triton(
            (gate, up),
            ptx=MarinKernelSet.uses_custom_ptx(self.kernels.swiglu),
        )
        return self.down_proj(hidden_states)


class MarinDecoderLayer(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.kernels = kernels
        self.self_attn = MarinAttention(device, dtype, kernels)
        self.mlp = MarinMLP(device, dtype, kernels)
        self.input_layernorm = MarinRMSNorm(device, dtype, kernels)
        self.post_attention_layernorm = MarinRMSNorm(device, dtype, kernels)

    def add_residual(self, residual, hidden_states):
        output, _ = self.kernels.add.forward_triton(
            (residual, hidden_states),
            ptx=MarinKernelSet.uses_custom_ptx(self.kernels.add),
        )
        return output

    def forward(self, hidden_states, past_key_value=None):
        residual = hidden_states
        hidden_states, key_value = self.self_attn(
            self.input_layernorm(hidden_states), past_key_value
        )
        hidden_states = self.add_residual(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return self.add_residual(residual, hidden_states), key_value


class MarinModel(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.embed_tokens = MarinEmbedding(device, dtype, kernels)
        self.layers = torch.nn.ModuleList(
            MarinDecoderLayer(device, dtype, kernels) for _ in range(NUM_LAYERS)
        )
        self.norm = MarinRMSNorm(device, dtype, kernels)

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        if past_key_values is not None and len(past_key_values) != NUM_LAYERS:
            raise ValueError("past_key_values must contain one entry per layer.")
        hidden_states = self.embed_tokens(input_ids)
        next_key_values = [] if use_cache else None
        for index, layer in enumerate(self.layers):
            past_key_value = None if past_key_values is None else past_key_values[index]
            hidden_states, key_value = layer(hidden_states, past_key_value)
            if use_cache:
                next_key_values.append(key_value)
        hidden_states = self.norm(hidden_states[:, -1:, :].contiguous())
        if use_cache:
            return hidden_states, tuple(next_key_values)
        return hidden_states


def sample_next_token(logits):
    top_values, top_indices = torch.topk(logits[:, -1, :].float(), k=50, dim=-1)
    probabilities = torch.softmax(top_values, dim=-1)
    cumulative_probabilities = probabilities.cumsum(dim=-1)
    remove_mask = cumulative_probabilities - probabilities > 0.95
    probabilities = probabilities.masked_fill(remove_mask, 0.0)
    sampled_index = torch.multinomial(probabilities, num_samples=1)
    return top_indices.gather(-1, sampled_index)


class MarinForCausalLM(torch.nn.Module):
    def __init__(self, device, dtype=torch.bfloat16, kernels=None):
        super().__init__()
        self.kernels = kernels or MarinKernelSet()
        self.model = MarinModel(device, dtype, self.kernels)
        self.lm_head = MarinLinear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            device,
            dtype,
            self.kernels,
        )

    def forward(self, input_ids, past_key_values=None, use_cache=False):
        model_output = self.model(input_ids, past_key_values, use_cache)
        if not use_cache:
            return self.lm_head(model_output)
        hidden_states, next_key_values = model_output
        return self.lm_head(hidden_states), next_key_values

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=100):
        logits, past_key_values = self(input_ids, use_cache=True)
        generated_tokens = []
        for token_index in range(max_new_tokens):
            next_token = sample_next_token(logits)
            generated_tokens.append(next_token)
            if next_token.item() == EOS_TOKEN_ID:
                break
            if token_index + 1 < max_new_tokens:
                logits, past_key_values = self(
                    next_token, past_key_values=past_key_values, use_cache=True
                )
        return torch.cat((input_ids, *generated_tokens), dim=-1)


def resolve_model_directory():
    return Path(
        snapshot_download(
            MODEL_ID,
            cache_dir=MODEL_CACHE_DIR,
            allow_patterns=(
                "*.json",
                "*.jinja",
                "*.safetensors",
                "tokenizer*",
                "vocab*",
                "merges*",
            ),
        )
    )


def load_pretrained_model(model_directory, kernels):
    model = MarinForCausalLM("meta", kernels=kernels)
    model.to_empty(device="cuda")
    index_path = model_directory / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    else:
        weight_path = model_directory / "model.safetensors"
        with safe_open(weight_path, framework="pt", device="cpu") as checkpoint:
            weight_map = {name: weight_path.name for name in checkpoint}

    loaded_keys = 0
    with torch.no_grad(), ExitStack() as stack:
        checkpoints = {
            shard: stack.enter_context(
                safe_open(model_directory / shard, framework="pt", device="cpu")
            )
            for shard in set(weight_map.values())
        }
        for model_key, parameter in model.state_dict().items():
            checkpoint_shard = weight_map.get(model_key)
            if checkpoint_shard is None:
                raise KeyError(f"Missing checkpoint tensor: {model_key}")
            tensor = checkpoints[checkpoint_shard].get_tensor(model_key)
            if tensor.shape != parameter.shape:
                raise ValueError(
                    f"Shape mismatch for {model_key}: "
                    f"{tensor.shape} != {parameter.shape}"
                )
            parameter.copy_(tensor.to("cuda", dtype=parameter.dtype))
            loaded_keys += 1
    LOGGER.info("Loaded %d Marin checkpoint tensors.", loaded_keys)
    return model.eval()


class MarinLLM(LLM):
    @classmethod
    def get_kernel_classes(cls):
        return MARIN_KERNEL_CLASSES

    def __init__(
        self,
        *,
        add_payload=None,
        causal_attention_payload=None,
        embedding_payload=None,
        linear_payload=None,
        rms_norm_payload=None,
        rope_payload=None,
        swiglu_payload=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Marin inference requires a CUDA GPU.")
        self.kernels = MarinKernelSet(
            add_payload=add_payload,
            causal_attention_payload=causal_attention_payload,
            embedding_payload=embedding_payload,
            linear_payload=linear_payload,
            rms_norm_payload=rms_norm_payload,
            rope_payload=rope_payload,
            swiglu_payload=swiglu_payload,
        )
        model_directory = resolve_model_directory()
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(
            model_directory,
            local_files_only=True,
            clean_up_tokenization_spaces=False,
        )
        self.model = load_pretrained_model(model_directory, self.kernels)

    @classmethod
    def from_custom_ptx(cls, **payloads):
        return cls(**payloads)

    def tokenize_messages(self, messages):
        if len(messages) != 1:
            raise ValueError("Marin inference currently requires one prompt.")
        prompt = messages[0]
        if isinstance(prompt, dict):
            prompt = prompt["content"]
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        )
        return inputs["input_ids"].to("cuda")

    @torch.inference_mode()
    def generate_tokens(self, input_ids, max_new_tokens):
        return self.model.generate(input_ids, max_new_tokens=max_new_tokens)

    def decode_response(self, output_ids, prompt_length):
        return self.tokenizer.decode(
            output_ids[0, prompt_length:],
            skip_special_tokens=True,
        )

    def decode_full_response(self, output_ids):
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
