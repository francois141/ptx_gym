import json
import logging
from contextlib import ExitStack
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from ..llm import LLM
from .add import QwenAddKernel
from .causal_attention import QwenCausalAttentionKernel
from .embedding import QwenEmbeddingKernel
from .linear import QwenLinearKernel
from .rms_norm import QwenRMSNormKernel
from .rope import QwenRoPEKernel
from .swiglu import QwenSwiGLUKernel

LOGGER = logging.getLogger(__name__)
MODEL_ID = "Qwen/Qwen3-4B"
MODEL_CACHE_DIR = "/matx/u/franc141/huggingface-cache"
VOCAB_SIZE = 151_936
HIDDEN_SIZE = 2_560
INTERMEDIATE_SIZE = 9_728
NUM_LAYERS = 36
NUM_ATTENTION_HEADS = 32
NUM_KEY_VALUE_HEADS = 8
HEAD_DIM = 128
RMS_NORM_EPS = 1e-6
EOS_TOKEN_ID = 151_645
THINK_END_TOKEN_ID = 151_668
QWEN_KERNEL_CLASSES = {
    "add": QwenAddKernel,
    "causal_attention": QwenCausalAttentionKernel,
    "embedding": QwenEmbeddingKernel,
    "linear": QwenLinearKernel,
    "rms_norm": QwenRMSNormKernel,
    "rope": QwenRoPEKernel,
    "swiglu": QwenSwiGLUKernel,
}


class QwenKernelSet:
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
        for name, kernel_class in QWEN_KERNEL_CLASSES.items():
            payload = payloads[name]
            if hasattr(payload, "model_dump"):
                payload = payload.model_dump()
            setattr(self, name, kernel_class(ptx=payload))

    @staticmethod
    def uses_custom_ptx(kernel):
        return kernel.compiled_kernel_ptx is not None


class QwenLinear(torch.nn.Module):
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
        self.kernel = (kernels or QwenKernelSet()).linear

    def forward(self, hidden_states):
        output, _ = self.kernel.forward_triton(
            (hidden_states, self.weight),
            ptx=QwenKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class QwenEmbedding(torch.nn.Module):
    def __init__(self, device=None, dtype=None, kernels=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(VOCAB_SIZE, HIDDEN_SIZE, device=device, dtype=dtype)
        )
        self.kernel = (kernels or QwenKernelSet()).embedding

    def forward(self, input_ids):
        output, _ = self.kernel.forward_triton(
            (input_ids, self.weight),
            ptx=QwenKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class QwenRMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, device=None, dtype=None, kernels=None):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.ones(hidden_size, device=device, dtype=dtype)
        )
        self.kernel = (kernels or QwenKernelSet()).rms_norm

    def forward(self, hidden_states):
        output, _ = self.kernel.forward_triton(
            (hidden_states, self.weight, RMS_NORM_EPS),
            ptx=QwenKernelSet.uses_custom_ptx(self.kernel),
        )
        return output


class QwenAttention(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.kernels = kernels
        self.q_proj = QwenLinear(
            HIDDEN_SIZE,
            NUM_ATTENTION_HEADS * HEAD_DIM,
            device,
            dtype,
            kernels,
        )
        self.k_proj = QwenLinear(
            HIDDEN_SIZE,
            NUM_KEY_VALUE_HEADS * HEAD_DIM,
            device,
            dtype,
            kernels,
        )
        self.v_proj = QwenLinear(
            HIDDEN_SIZE,
            NUM_KEY_VALUE_HEADS * HEAD_DIM,
            device,
            dtype,
            kernels,
        )
        self.o_proj = QwenLinear(
            NUM_ATTENTION_HEADS * HEAD_DIM,
            HIDDEN_SIZE,
            device,
            dtype,
            kernels,
        )
        self.q_norm = QwenRMSNorm(HEAD_DIM, device, dtype, kernels)
        self.k_norm = QwenRMSNorm(HEAD_DIM, device, dtype, kernels)

    def forward(self, hidden_states):
        batch_size, sequence_length, _ = hidden_states.shape
        query_states = self.q_proj(hidden_states).view(
            batch_size,
            sequence_length,
            NUM_ATTENTION_HEADS,
            HEAD_DIM,
        )
        key_states = self.k_proj(hidden_states).view(
            batch_size,
            sequence_length,
            NUM_KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        value_states = self.v_proj(hidden_states).view(
            batch_size,
            sequence_length,
            NUM_KEY_VALUE_HEADS,
            HEAD_DIM,
        )
        query_states = self.q_norm(query_states).transpose(1, 2).contiguous()
        key_states = self.k_norm(key_states).transpose(1, 2).contiguous()
        value_states = value_states.transpose(1, 2).contiguous()
        query_states, _ = self.kernels.rope.forward_triton(
            query_states,
            ptx=QwenKernelSet.uses_custom_ptx(self.kernels.rope),
        )
        key_states, _ = self.kernels.rope.forward_triton(
            key_states,
            ptx=QwenKernelSet.uses_custom_ptx(self.kernels.rope),
        )
        attention_output, _ = self.kernels.causal_attention.forward_triton(
            (query_states, key_states, value_states),
            ptx=QwenKernelSet.uses_custom_ptx(self.kernels.causal_attention),
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            NUM_ATTENTION_HEADS * HEAD_DIM,
        )
        return self.o_proj(attention_output.contiguous())


class QwenMLP(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.kernels = kernels
        self.gate_proj = QwenLinear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device,
            dtype,
            kernels,
        )
        self.up_proj = QwenLinear(
            HIDDEN_SIZE,
            INTERMEDIATE_SIZE,
            device,
            dtype,
            kernels,
        )
        self.down_proj = QwenLinear(
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
            ptx=QwenKernelSet.uses_custom_ptx(self.kernels.swiglu),
        )
        return self.down_proj(hidden_states)


class QwenDecoderLayer(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.kernels = kernels
        self.self_attn = QwenAttention(device, dtype, kernels)
        self.mlp = QwenMLP(device, dtype, kernels)
        self.input_layernorm = QwenRMSNorm(
            HIDDEN_SIZE,
            device,
            dtype,
            kernels,
        )
        self.post_attention_layernorm = QwenRMSNorm(
            HIDDEN_SIZE,
            device,
            dtype,
            kernels,
        )

    def add_residual(self, residual, hidden_states):
        output, _ = self.kernels.add.forward_triton(
            (residual, hidden_states),
            ptx=QwenKernelSet.uses_custom_ptx(self.kernels.add),
        )
        return output

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = self.add_residual(residual, hidden_states)
        residual = hidden_states
        hidden_states = self.mlp(self.post_attention_layernorm(hidden_states))
        return self.add_residual(residual, hidden_states)


class QwenModel(torch.nn.Module):
    def __init__(self, device, dtype, kernels):
        super().__init__()
        self.embed_tokens = QwenEmbedding(device, dtype, kernels)
        self.layers = torch.nn.ModuleList(
            QwenDecoderLayer(device, dtype, kernels) for _ in range(NUM_LAYERS)
        )
        self.norm = QwenRMSNorm(HIDDEN_SIZE, device, dtype, kernels)

    def forward(self, input_ids):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm(hidden_states[:, -1:, :].contiguous())


class Qwen3ForCausalLM(torch.nn.Module):
    def __init__(self, device, dtype=torch.bfloat16, kernels=None):
        super().__init__()
        self.kernels = kernels or QwenKernelSet()
        self.model = QwenModel(device, dtype, self.kernels)
        self.lm_head = QwenLinear(
            HIDDEN_SIZE,
            VOCAB_SIZE,
            device,
            dtype,
            self.kernels,
        )
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids):
        return self.lm_head(self.model(input_ids))

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=32):
        for _ in range(max_new_tokens):
            next_token = self(input_ids).argmax(dim=-1)
            input_ids = torch.cat((input_ids, next_token), dim=-1)
            if (next_token == EOS_TOKEN_ID).all():
                break
        return input_ids


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
    model = Qwen3ForCausalLM("meta", kernels=kernels)
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
            if model_key == "lm_head.weight":
                continue
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
    model.lm_head.weight = model.model.embed_tokens.weight
    LOGGER.info("Loaded %d Qwen checkpoint tensors.", loaded_keys)
    return model.eval()


class QwenLLM(LLM):
    @classmethod
    def get_kernel_classes(cls):
        return QWEN_KERNEL_CLASSES

    def __init__(
        self,
        *,
        enable_thinking=True,
        add_payload=None,
        causal_attention_payload=None,
        embedding_payload=None,
        linear_payload=None,
        rms_norm_payload=None,
        rope_payload=None,
        swiglu_payload=None,
    ):
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen inference requires a CUDA GPU.")
        self.enable_thinking = enable_thinking
        self.kernels = QwenKernelSet(
            add_payload=add_payload,
            causal_attention_payload=causal_attention_payload,
            embedding_payload=embedding_payload,
            linear_payload=linear_payload,
            rms_norm_payload=rms_norm_payload,
            rope_payload=rope_payload,
            swiglu_payload=swiglu_payload,
        )
        model_directory = resolve_model_directory()
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_directory,
            local_files_only=True,
        )
        self.model = load_pretrained_model(model_directory, self.kernels)

    @classmethod
    def from_custom_ptx(cls, *, enable_thinking=True, **payloads):
        return cls(enable_thinking=enable_thinking, **payloads)

    def tokenize_messages(self, messages):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
        )
        if not isinstance(input_ids, torch.Tensor):
            input_ids = input_ids["input_ids"]
        return input_ids.to("cuda")

    @torch.inference_mode()
    def generate_tokens(self, input_ids, max_new_tokens):
        return self.model.generate(input_ids, max_new_tokens=max_new_tokens)

    def decode_parts(self, output_ids, prompt_length):
        generated_ids = output_ids[0, prompt_length:].tolist()
        try:
            split_index = len(generated_ids) - generated_ids[::-1].index(
                THINK_END_TOKEN_ID
            )
        except ValueError:
            split_index = 0
        thinking_content = self.tokenizer.decode(
            generated_ids[:split_index],
            skip_special_tokens=True,
        ).strip("\n")
        content = self.tokenizer.decode(
            generated_ids[split_index:],
            skip_special_tokens=True,
        ).strip("\n")
        return thinking_content, content

    def decode_response(self, output_ids, prompt_length):
        thinking_content, content = self.decode_parts(output_ids, prompt_length)
        return content or thinking_content
