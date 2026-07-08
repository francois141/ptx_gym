from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from triton_ptx.evaluation.sanitizer import (
    _add_ptx_line_information,
    _run_sanitizer_tool,
    _reported_ptx_locations,
)


def test_add_ptx_line_information_preserves_original_line_numbers() -> None:
    """Annotate instructions with their lines while ignoring PTX declarations."""
    ptx = """.version 8.0
.target sm_80
.address_size 64
.visible .entry kernel()
{
    .reg .b64 address;
    mov.u64 address, 0;
    st.global.u32 [address], 1;
    ret;
}
"""

    annotated = _add_ptx_line_information(ptx)

    assert '.file 1 "candidate.ptx"' in annotated
    assert ".loc 1 7 0\n    mov.u64 address, 0;" in annotated
    assert ".loc 1 8 0\n    st.global.u32 [address], 1;" in annotated
    assert ".loc 1 9 0\n    ret;" in annotated
    assert ".loc 1 6 0" not in annotated


def test_reported_ptx_locations_includes_offending_source() -> None:
    """Expose the original PTX line and instruction from sanitizer output."""
    ptx = ".version 8.0\nmov.u64 address, 0;\nst.global.u32 [address], 1;\n"
    diagnostics = [
        "Invalid __global__ write of size 4 bytes",
        "at candidate.ptx:3 in kernel",
        "at candidate.ptx:3 in kernel",
    ]

    assert _reported_ptx_locations(diagnostics, ptx) == [{"line": 3, "source": "st.global.u32 [address], 1;"}]


def test_sanitizer_report_error_names_ptx_line(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Put the offending PTX line in the evaluator's primary error message."""
    completed = subprocess.CompletedProcess(
        args=[],
        returncode=99,
        stdout="",
        stderr=(
            "========= Invalid __global__ write of size 4 bytes\n"
            "=========     at candidate.ptx:2 in kernel\n"
            "========= ERROR SUMMARY: 1 error\n"
        ),
    )
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)

    report = _run_sanitizer_tool(
        "compute-sanitizer",
        "memcheck",
        tmp_path / "request.json",
        10,
        "mov.u64 address, 0;\nst.global.u32 [address], 1;\n",
    )

    assert report["error"] == ("PTX memory error at line 2: st.global.u32 [address], 1;")
