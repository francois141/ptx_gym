import pytest

from triton_ptx.helpers.environment import get_ptx_system_config, is_gpu_available


@pytest.mark.skipif(not is_gpu_available(), reason="CUDA is not available on this system.")
def test_get_ptx_system_config_returns_non_empty_values():
    version, target, address_size = get_ptx_system_config()

    assert version
    assert target
    assert address_size
