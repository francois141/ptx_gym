import sys

import pytest

import triton_ptx.generator as generator_factory
from triton_ptx import test_time_scaling_loop as tts


def test_build_prompter_selects_gemini(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(generator_factory, "_get_generator_class", lambda provider: lambda **kwargs: sentinel)

    assert generator_factory.build_prompter("gemini") is sentinel


def test_build_prompter_passes_model_and_options(monkeypatch):
    captured = {}

    def fake_openai_prompt(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(generator_factory, "_get_generator_class", lambda provider: fake_openai_prompt)

    generator_factory.build_prompter(
        "openai",
        model="gpt-5-mini",
        options={"reasoning_effort": "low", "unused": None},
    )

    assert captured == {
        "model": "gpt-5-mini",
        "reasoning_effort": "low",
    }


def test_build_prompter_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        generator_factory.build_prompter("unknown")


def test_parse_args_accepts_config_path(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test_time_scaling_loop.py",
            "AddKernel",
            "--config",
            "triton_ptx/configs/test_time_scaling_gemini.yaml",
        ],
    )

    args = tts.parse_args()

    assert args.kernel == "AddKernel"
    assert str(args.config) == "triton_ptx/configs/test_time_scaling_gemini.yaml"


def test_load_config_merges_with_defaults(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "loop:\n"
        "  rounds: 4\n"
        "generator:\n"
        "  provider: gemini\n"
        "  model: gemini-2.5-pro\n",
        encoding="utf-8",
    )

    config = tts.load_config(config_path)

    assert config.loop.rounds == 4
    assert config.loop.k == 2
    assert config.loop.max_retries == 3
    assert config.generator.provider == "gemini"
    assert config.generator.model == "gemini-2.5-pro"
