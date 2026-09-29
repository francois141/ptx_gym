from __future__ import annotations

import pytest

from ptx_gym.evaluation.types import Payload


def test_payload_from_input_accepts_num_threads_fields() -> None:
    payload = Payload.from_input(
        {
            "ptx": ".version 8.0\n.visible .entry test() { ret; }",
            "num_threads_x": 128,
            "num_threads_y": 2,
            "num_threads_z": 1,
        }
    )

    assert payload.threads_x == 128
    assert payload.threads_y == 2
    assert payload.threads_z == 1


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("threads_x", 0),
        ("threads_y", -1),
        ("threads_z", -1),
    ],
)
def test_payload_rejects_non_positive_thread_dimensions(
    field_name: str,
    field_value: int | bool,
) -> None:
    payload = {
        "ptx": ".version 8.0\n.visible .entry test() { ret; }",
        "threads_x": 128,
        "threads_y": 1,
        "threads_z": 1,
    }
    payload[field_name] = field_value

    with pytest.raises(ValueError, match=field_name):
        Payload.from_input(payload)
