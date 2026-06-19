from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI
from triton_ptx.evaluation import EvaluatedCandidate, TritonPTXCandidateEvaluator
from triton_ptx.evaluation.sandbox import compile_ptx
from triton_ptx.generator.base import parse_response_text
from triton_ptx.helpers.environment import get_ptx_system_config
from triton_ptx.helpers.ptx import parse_ptx_signature
from triton_ptx.helpers.triton import dump_kernel_ptx
from triton_ptx.kernels import resolve_kernel
from triton_ptx.prompts import (
    build_follow_up_prompt_for_operator,
    build_prompt_for_operator,
    build_repair_prompt_for_operator,
)

PACKAGE_DIR = Path(__file__).resolve().parent
TENSOR_CORE_SKILL_PATH = PACKAGE_DIR / "skills" / "use-tensor-cores-ptx-mma" / "SKILL.md"

TENSOR_CORE_FALLBACK_PROMPT = """You are optimizing NVIDIA PTX kernels.
Prefer Tensor Core paths for GEMM-like work when viable: FP16/BF16/TF32 inputs, FP32 accumulation,
ldmatrix/shared-memory staging, and mma.sync.aligned or newer WGMMA-family instructions. Use local
tools to compile, verify, benchmark, and repair candidates before finalizing."""

AGENT_PROMPTS_PATH = PACKAGE_DIR / "triton_ptx" / "prompts"
SYSTEM_PROMPT_PATH = AGENT_PROMPTS_PATH / "SYSTEM.md"

def log_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def log_subsection(title: str) -> None:
    print("\n" + "-" * 80)
    print(title)
    print("-" * 80)


def log_json(title: str, value: Any, limit: int = 300) -> None:
    log_subsection(title)
    text = json.dumps(_json_safe(value), indent=2, ensure_ascii=False)
    print(text)
    return
    if len(text) > limit:
        print(text[:limit])
        print(f"\n... truncated after {limit} characters ...")
    else:
        print(text)


def load_tensor_core_skill_prompt(skill_path: Path = TENSOR_CORE_SKILL_PATH) -> str:
    if not skill_path.exists():
        return TENSOR_CORE_FALLBACK_PROMPT

    skill_text = skill_path.read_text(encoding="utf-8").strip()
    return (
        "You are optimizing NVIDIA PTX kernels with the following skill loaded.\n\n"
        f"{skill_text}\n\n"
        "Use the available local tools to compile, verify, benchmark, and repair candidates before finalizing."
    )


def get_system_prompt() -> str:
    system_prompt = SYSTEM_PROMPT_PATH.read_text()
    skills = [load_tensor_core_skill_prompt()]
    skill_prompt = "\n=============\n".join(skills)
    return system_prompt + "\n\n<skills>" + skill_prompt + "\n</skills>"


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def _candidate_from_text_or_payload(candidate: Any) -> dict[str, Any]:
    if isinstance(candidate, dict):
        return candidate
    if isinstance(candidate, str):
        parsed = parse_response_text(candidate)
        if len(parsed) != 1:
            raise ValueError(f"Expected exactly one candidate, got {len(parsed)}.")
        return parsed[0]
    raise TypeError("candidate must be a dictionary or serialized candidate text.")


def _evaluated_candidate_from_dict(data: dict[str, Any]) -> EvaluatedCandidate:
    fields = set(EvaluatedCandidate.__dataclass_fields__) - {"sort_index"}
    payload = {key: value for key, value in data.items() if key in fields}

    defaults = {
        "kernel_name": "",
        "git_commit_hash": "unknown",
        "round_index": 1,
        "index": 1,
        "payload": {},
        "compiles": False,
        "correct": False,
        "message": "",
        "triton_p20": float("inf"),
        "triton_p50": float("inf"),
        "triton_p80": float("inf"),
        "p20": float("inf"),
        "p50": float("inf"),
        "p80": float("inf"),
        "speedup_vs_triton": 0,
        "compile_output": "",
        "compile_error": "",
        "timing_error": "",
        "verifier_report": {},
    }
    defaults.update(payload)
    return EvaluatedCandidate(**defaults)


def get_kernel_prompt(kernel_name: str, num_answers: int = 1) -> dict[str, Any]:
    kernel_cls = resolve_kernel(kernel_name)
    baseline_ptx = dump_kernel_ptx(kernel_cls())
    ptx_signature = parse_ptx_signature(baseline_ptx)
    version, target, address_size = get_ptx_system_config()

    prompt = build_prompt_for_operator(
        kernel_cls,
        num_answers=num_answers,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
    )

    result = {
        "kernel_name": kernel_cls.__name__,
        "version": version,
        "target": target,
        "address_size": address_size,
        #"ptx_signature": ptx_signature,
        #"baseline_ptx": baseline_ptx,
        "prompt": prompt,
    }

    log_json("KERNEL PROMPT CONTEXT", result)
    return result


def compile_candidate(kernel_name: str, candidate: dict[str, Any] | str) -> dict[str, Any]:
    kernel_cls = resolve_kernel(kernel_name)
    payload = _candidate_from_text_or_payload(candidate)

    log_json("COMPILING CANDIDATE PAYLOAD", payload)

    result = compile_ptx(payload)
    result = _json_safe(result)

    log_json("COMPILE RESULT", result)
    return result

def evaluate_candidate(
    kernel_name: str,
    candidate: dict[str, Any] | str,
    round_index: int = 1,
    candidate_index: int = 1,
    clear_cache: bool = False,
) -> dict[str, Any]:
    kernel_cls = resolve_kernel(kernel_name)
    payload = _candidate_from_text_or_payload(candidate)

    log_json("EVALUATING CANDIDATE PAYLOAD", payload)

    evaluator = TritonPTXCandidateEvaluator(kernel_cls, clear_cache=clear_cache)
    result = evaluator.evaluate(
        payload,
        round_index=round_index,
        candidate_index=candidate_index,
    )

    result_dict = _json_safe(result.to_dict())
    log_json("EVALUATION RESULT", result_dict)
    return result_dict


def get_repair_prompt(
    kernel_name: str,
    evaluation_result: dict[str, Any],
    retry_index: int = 1,
    max_retries: int = 3,
) -> dict[str, Any]:
    kernel_cls = resolve_kernel(kernel_name)
    version, target, address_size = get_ptx_system_config()
    baseline_ptx = dump_kernel_ptx(kernel_cls())
    ptx_signature = parse_ptx_signature(baseline_ptx)

    failed_candidate = _evaluated_candidate_from_dict(evaluation_result)

    prompt = build_repair_prompt_for_operator(
        failed_candidate,
        kernel_cls,
        retry_index=retry_index,
        max_retries=max_retries,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
    )

    result = {"prompt": prompt}
    log_json("REPAIR PROMPT", result)
    return result


def get_follow_up_prompt(
    kernel_name: str,
    evaluation_results: list[dict[str, Any]],
    num_answers: int = 1,
) -> dict[str, Any]:
    kernel_cls = resolve_kernel(kernel_name)
    version, target, address_size = get_ptx_system_config()
    baseline_ptx = dump_kernel_ptx(kernel_cls())
    ptx_signature = parse_ptx_signature(baseline_ptx)

    candidates = [_evaluated_candidate_from_dict(result) for result in evaluation_results]

    prompt = build_follow_up_prompt_for_operator(
        candidates,
        kernel_cls,
        num_answers=num_answers,
        version=version,
        target=target,
        address_size=address_size,
        ptx_signature=ptx_signature,
    )

    result = {"prompt": prompt}
    log_json("FOLLOW-UP PROMPT", result)
    return result


TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "compile_candidate": compile_candidate,
    "evaluate_candidate": evaluate_candidate,
    "get_repair_prompt": get_repair_prompt,
    "get_follow_up_prompt": get_follow_up_prompt,
}


OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compile_candidate",
            "description": "Compile a PTX candidate payload without running full correctness or benchmark evaluation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_name": {"type": "string"},
                    "candidate": {
                        "description": "Candidate dictionary or serialized candidate text containing ptx and launch dimensions.",
                        "oneOf": [{"type": "object"}, {"type": "string"}],
                    },
                },
                "required": ["kernel_name", "candidate"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_candidate",
            "description": "Compile, verify correctness, benchmark, and return timing/speedup metrics for one PTX candidate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_name": {"type": "string"},
                    "candidate": {
                        "description": "Candidate dictionary or serialized candidate text containing ptx and launch dimensions.",
                        "oneOf": [{"type": "object"}, {"type": "string"}],
                    },
                    "round_index": {"type": "integer", "minimum": 1, "default": 1},
                    "candidate_index": {"type": "integer", "minimum": 1, "default": 1},
                    "clear_cache": {"type": "boolean", "default": False},
                },
                "required": ["kernel_name", "candidate"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_repair_prompt",
            "description": "Build a repair prompt from an evaluation result after compile, verification, or benchmark failure.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_name": {"type": "string"},
                    "evaluation_result": {"type": "object"},
                    "retry_index": {"type": "integer", "minimum": 1, "default": 1},
                    "max_retries": {"type": "integer", "minimum": 0, "default": 3},
                },
                "required": ["kernel_name", "evaluation_result"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_follow_up_prompt",
            "description": "Build a follow-up prompt from previous evaluated candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_name": {"type": "string"},
                    "evaluation_results": {"type": "array", "items": {"type": "object"}},
                    "num_answers": {"type": "integer", "minimum": 1, "default": 1},
                },
                "required": ["kernel_name", "evaluation_results"],
                "additionalProperties": False,
            },
        },
    },
]


def run_agent_loop(
    kernel_name: str,
    *,
    model: str = "gpt-5",
    max_tool_rounds: int = 20,
    reasoning_effort: str | None = "medium",
    trace_path: str | Path | None = "trace.json",
) -> str:
    client = OpenAI()

    kernel_context = get_kernel_prompt(kernel_name)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": kernel_context["prompt"]},
    ]

    trace: list[dict[str, Any]] = []

    for round_idx in range(max_tool_rounds):
        log_section(f"MODEL ROUND {round_idx + 1}")

        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": OPENAI_TOOLS,
        }

        if reasoning_effort is not None:
            create_kwargs["reasoning_effort"] = reasoning_effort

        response = client.chat.completions.create(**create_kwargs)

        message = response.choices[0].message

        print(message)
        message_dict = message.model_dump(exclude_none=True)

        messages.append(message_dict)

        trace.append(
            {
                "round": round_idx + 1,
                "type": "assistant",
                "message": _json_safe(message_dict),
            }
        )

        if VERBOSE:
            if message.content:
                log_subsection("ASSISTANT CONTENT")
                print(message.content)

            if message.tool_calls:
                log_subsection("ASSISTANT TOOL CALLS")
                for tool_call in message.tool_calls:
                    print(f"\nTool: {tool_call.function.name}")
                    print(tool_call.function.arguments)

        tool_calls = message.tool_calls or []
        if not tool_calls:
            if trace_path:
                Path(trace_path).write_text(
                    json.dumps(trace, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            return message.content or ""

        for tool_call in tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")

            log_json(f"TOOL INPUT: {name}", args)

            try:
                result = TOOL_FUNCTIONS[name](**args)
            except Exception as exc:
                result = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "tool": name,
                    "arguments": args,
                }

            safe_result = _json_safe(result)

            log_json(f"TOOL OUTPUT: {name}", safe_result)

            trace.append(
                {
                    "round": round_idx + 1,
                    "type": "tool",
                    "tool": name,
                    "args": _json_safe(args),
                    "result": safe_result,
                }
            )

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(safe_result, ensure_ascii=False),
                }
            )

        if trace_path:
            Path(trace_path).write_text(
                json.dumps(trace, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    if trace_path:
        Path(trace_path).write_text(
            json.dumps(trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return "Stopped after max_tool_rounds without a final answer."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an OpenAI tool-calling PTX optimization loop.")
    parser.add_argument("kernel", help="Kernel class name, for example MatrixMultiplicationKernel.")
    parser.add_argument("--model", default="gpt-5")
    parser.add_argument("--max-tool-rounds", type=int, default=20)
    parser.add_argument("--reasoning-effort", default="medium")
    parser.add_argument("--trace-path", default="trace.json")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    global VERBOSE

    args = parse_args()
    VERBOSE = args.verbose

    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    print(
        run_agent_loop(
            args.kernel,
            model=args.model,
            max_tool_rounds=args.max_tool_rounds,
            reasoning_effort=reasoning_effort,
            trace_path=args.trace_path,
        )
    )


if __name__ == "__main__":
    main()
