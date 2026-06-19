import sys

import pytest

from triton_ptx import test_time_scaling_loop as tts


def test_build_prompter_selects_gemini(monkeypatch):
    sentinel = object()

    monkeypatch.setattr(tts, "GeminiPrompt", lambda: sentinel)

    assert tts.build_prompter("gemini") is sentinel


def test_build_prompter_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unsupported provider"):
        tts.build_prompter("unknown")


def test_parse_args_accepts_gemini_flag(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["test_time_scaling_loop.py", "AddKernel", "--gemini"],
    )

    args = tts.parse_args()

    assert args.kernel == "AddKernel"
    assert args.gemini is True
    assert args.anthropic is False
