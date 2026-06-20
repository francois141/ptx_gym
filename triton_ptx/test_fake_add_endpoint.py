from __future__ import annotations

import json

import pytest
import torch
from omegaconf import OmegaConf

import triton_ptx.evaluation.evaluate as evaluate_module
from triton_ptx.evaluation.sandbox.evaluation import Timing
from triton_ptx.evaluation.sandbox.verification import OutputVerifier
from triton_ptx.llm_endpoint import create_llm_endpoint
from triton_ptx.llm_endpoint.fake_add_prompt import ADD_KERNEL_PTX
from triton_ptx import test_time_scaling_loop as tts


class ThreeSizeOutputVerifier(OutputVerifier):
    def __init__(self):
        super().__init__(sizes=(3, 1024, 2049), iters_per_size=1)

    def verify(self, op) -> bool:
        result = super().verify(op)
        self.last_report = {
            **self.last_report,
            "verified_sizes": list(self.sizes),
        }
        return result


def _sm89_available() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (8, 9)


def test_fake_add_endpoint_rejects_non_add_kernel():
    with pytest.raises(ValueError, match="only supports AddKernel"):
        create_llm_endpoint(
            "fake_add",
            options={
                "kernel_name": "ReLUKernel",
            },
        )


@pytest.mark.skipif(
    not _sm89_available(),
    reason="fake_add returns sm_89 PTX and requires an sm_89 CUDA device.",
)
def test_fake_add_endpoint_runs_three_scaling_rounds_with_three_verified_sizes(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(tts, "dump_kernel_ptx", lambda kernel: ADD_KERNEL_PTX)
    monkeypatch.setattr(evaluate_module, "OutputVerifier", ThreeSizeOutputVerifier)
    monkeypatch.setattr(
        evaluate_module,
        "benchmark_operator",
        lambda operator: {
            "triton": Timing(2.0, 2.0, 2.0),
            "ptx": Timing(1.0, 1.0, 1.0),
        },
    )

    config = OmegaConf.create(
        {
            "loop": {
                "rounds": 3,
                "k": 1,
                "max_retries": 0,
            },
            "generator": {
                "provider": "fake_add",
            },
            "storage": {
                "database_dir": str(tmp_path),
            },
        }
    )

    run_dir = tts.run_test_time_scaling_loop("AddKernel", config=config)

    round_results = []
    for round_index in range(1, 4):
        result_path = run_dir / f"output_{round_index}_0.json"
        assert result_path.exists()
        round_results.append(json.loads(result_path.read_text(encoding="utf-8")))

    assert [result["round_index"] for result in round_results] == [1, 2, 3]
    assert {result["payload"]["num_threads_x"] for result in round_results} == {128}
    assert all(result["verifier_report"]["verified_sizes"] == [3, 1024, 2049] for result in round_results)
    assert all(result["compiles"] for result in round_results)
    assert all(result["correct"] for result in round_results)
    assert all(result["passed"] for result in round_results)
    assert all("benchmarked successfully" in result["message"] for result in round_results)
