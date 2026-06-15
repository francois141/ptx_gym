import pytest

from triton_ptx.helpers import environment
from triton_ptx.helpers.environment import get_ptx_system_config, is_gpu_available


def test_guess_ptx_version_from_ptxas_output(monkeypatch):
    mock_output = """
ptxas: NVIDIA (R) Ptx optimizing assembler
Cuda compilation tools, release 12.8, V12.8.93
""".strip()

    monkeypatch.setattr(
        environment.subprocess,
        "check_output",
        lambda *args, **kwargs: mock_output.encode("utf-8"),
    )

    assert environment._guess_ptx_version_from_ptxas() == "8.7"


@pytest.mark.skipif(
    not is_gpu_available(),
    reason="CUDA is not available on this system.",
)
def test_get_ptx_system_config_returns_non_empty_values():
    version, target, address_size = get_ptx_system_config()

    assert version
    assert target
    assert address_size
