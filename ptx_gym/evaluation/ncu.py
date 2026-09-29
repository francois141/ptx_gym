from __future__ import annotations

import csv
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from io import StringIO
from pathlib import Path
from typing import Any
from uuid import uuid4

import orjson
from ptx_gym.evaluation.types import Payload, is_valid_tuning_config

NCU_ENV_VAR = "NCU_PATH"
TMP_FILES_DIR = Path(__file__).resolve().parents[2] / "tmp_files"
PROFILED_KERNEL_NAME = "kernel"

_MAX_MESSAGES = 20


def _annotate_ptx_lines(ptx: str, source_path: Path) -> str:
    source_lines = ptx.splitlines()
    file_ids = [
        int(match) for match in re.findall(r"^\s*\.file\s+(\d+)\b", ptx, re.MULTILINE)
    ]
    file_id = max(file_ids, default=0) + 1
    escaped_path = str(source_path).replace("\\", "\\\\").replace('"', '\\"')
    file_directive = f'.file {file_id} "{escaped_path}"'
    annotated_lines = []
    inserted_file = False
    entry_seen = False
    body_depth = 0

    for line_number, line in enumerate(source_lines, start=1):
        stripped = line.strip()
        if not inserted_file and re.search(r"(?:\.visible\s+)?\.entry\b", stripped):
            annotated_lines.append(file_directive)
            inserted_file = True
        if re.search(r"(?:\.visible\s+)?\.entry\b", stripped):
            entry_seen = True
        if entry_seen and body_depth > 0 and _is_ptx_instruction(stripped):
            indentation = line[: len(line) - len(line.lstrip())]
            annotated_lines.append(f"{indentation}.loc {file_id} {line_number} 0")
        annotated_lines.append(line)
        if entry_seen:
            body_depth += line.count("{") - line.count("}")

    if not inserted_file:
        raise ValueError("PTX source does not contain an entry function.")
    return "\n".join(annotated_lines) + ("\n" if ptx.endswith("\n") else "")


def _is_ptx_instruction(stripped_line: str) -> bool:
    return bool(
        stripped_line
        and stripped_line.endswith(";")
        and not stripped_line.startswith((".", "//"))
    )


_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "hardware.sm_version_major": ("device__attribute_compute_capability_major",),
    "hardware.sm_version_minor": ("device__attribute_compute_capability_minor",),
    "hardware.display_name": ("device__attribute_display_name", "Device"),
    "hardware.compute_capability": ("CC",),
    "hardware.sm_count": ("device__attribute_multiprocessor_count",),
    "hardware.warp_size": ("device__attribute_warp_size",),
    "hardware.l2_cache_bytes": ("device__attribute_l2_cache_size",),
    "hardware.max_warps_per_sm": ("device__attribute_max_warps_per_multiprocessor",),
    "kernel.name": ("Kernel Name",),
    "kernel.grid_size": ("Grid Size",),
    "kernel.block_size": ("Block Size",),
    "kernel.grid_dim_x": ("launch__grid_dim_x",),
    "kernel.grid_dim_y": ("launch__grid_dim_y",),
    "kernel.grid_dim_z": ("launch__grid_dim_z",),
    "kernel.block_dim_x": ("launch__block_dim_x",),
    "kernel.block_dim_y": ("launch__block_dim_y",),
    "kernel.block_dim_z": ("launch__block_dim_z",),
    "kernel.shared_memory_static_bytes": (
        "launch__shared_mem_per_block_static",
        "launch__shared_memory_per_block_static",
    ),
    "kernel.shared_memory_dynamic_bytes": (
        "launch__shared_mem_per_block_dynamic",
        "launch__shared_memory_per_block_dynamic",
    ),
    "kernel.registers_per_thread": ("launch__registers_per_thread",),
    "kernel.execution_time": ("gpu__time_duration.sum",),
    "occupancy.achieved_pct": (
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
    ),
    "occupancy.waves_per_sm": ("launch__waves_per_multiprocessor",),
    "occupancy.limit_blocks": ("launch__occupancy_limit_blocks",),
    "occupancy.limit_registers": ("launch__occupancy_limit_registers",),
    "occupancy.limit_shared_memory": ("launch__occupancy_limit_shared_mem",),
    "occupancy.limit_warps": ("launch__occupancy_limit_warps",),
    "scheduler.issue_slot_utilization_pct": (
        "smsp__issue_active.avg.pct_of_peak_sustained_active",
    ),
    "scheduler.eligible_warps_per_cycle": (
        "smsp__warps_eligible.avg.per_cycle_active",
    ),
    "scheduler.active_warps_per_cycle": (
        "smsp__warps_active.avg.per_cycle_active",
        "sm__warps_active.avg.per_cycle_active",
    ),
    "scheduler.stall_memory_dependency_pct": (
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    ),
    "scheduler.stall_short_scoreboard_pct": (
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
    ),
    "scheduler.stall_long_scoreboard_pct": (
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
    ),
    "scheduler.stall_barrier_pct": (
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
    ),
    "scheduler.stall_not_selected_pct": (
        "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
    ),
    "scheduler.stall_math_pipe_unavailable_pct": (
        "smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct",
    ),
    "scheduler.stall_mio_throttle_pct": (
        "smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct",
    ),
    "scheduler.stall_lg_throttle_pct": (
        "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
    ),
    "scheduler.stall_wait_pct": ("smsp__warp_issue_stalled_wait_per_warp_active.pct",),
    "scheduler.stall_branch_resolving_pct": (
        "smsp__warp_issue_stalled_branch_resolving_per_warp_active.pct",
    ),
    "scheduler.stall_dispatch_pct": (
        "smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct",
    ),
    "memory.dram_throughput_pct": (
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "memory.l2_throughput_pct": ("lts__throughput.avg.pct_of_peak_sustained_elapsed",),
    "memory.l1_throughput_pct": (
        "l1tex__throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "memory.l1_hit_rate_pct": ("l1tex__t_sector_hit_rate.pct",),
    "memory.l2_hit_rate_pct": ("lts__t_sector_hit_rate.pct",),
    "memory.dram_bytes_read": ("dram__bytes_read.sum",),
    "memory.dram_bytes_write": ("dram__bytes_write.sum",),
    "memory.global_load_requests": ("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",),
    "memory.global_store_requests": (
        "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    ),
    "memory.global_load_sectors": ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",),
    "memory.global_load_transactions": (
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    ),
    "memory.global_store_sectors": ("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",),
    "memory.global_store_transactions": (
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    ),
    "memory.l1_read_sectors": ("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",),
    "memory.l1_write_sectors": ("l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",),
    "memory.l2_read_sectors": ("lts__t_sectors_op_read.sum",),
    "memory.l2_write_sectors": ("lts__t_sectors_op_write.sum",),
    "memory.shared_load_requests": ("l1tex__t_requests_pipe_lsu_mem_shared_op_ld.sum",),
    "memory.shared_store_requests": (
        "l1tex__t_requests_pipe_lsu_mem_shared_op_st.sum",
    ),
    "memory.shared_load_wavefronts": (
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    ),
    "memory.shared_load_transactions": (
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum",
    ),
    "memory.shared_store_wavefronts": (
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    ),
    "memory.shared_store_transactions": (
        "l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum",
    ),
    "memory.shared_load_excessive_wavefronts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum",
    ),
    "memory.shared_store_excessive_wavefronts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum",
    ),
    "memory.shared_bank_conflicts": (
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    ),
    "memory.replay_overhead": ("smsp__sass_inst_executed_op_memory_issues.sum",),
    "instructions.ipc": (
        "smsp__inst_executed.avg.per_cycle_active",
        "sm__inst_executed.avg.per_cycle_active",
    ),
    "instructions.executed": ("smsp__inst_executed.sum", "sm__inst_executed.sum"),
    "instructions.integer": ("smsp__sass_thread_inst_executed_op_integer_pred_on.sum",),
    "instructions.fp16": ("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",),
    "instructions.fp32": (
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    ),
    "instructions.fp64": ("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",),
    "instructions.flop_count_sp": (
        "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
        "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    ),
    "instructions.tensor_core_utilization_pct": (
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    ),
    "roofline.memory_peak_pct": (
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
    ),
    "roofline.compute_peak_pct": ("sm__throughput.avg.pct_of_peak_sustained_elapsed",),
}


def _parse_metric_value(value: str) -> int | float | str:
    cleaned_value = value.strip().replace(",", "")
    if not cleaned_value:
        return ""

    try:
        numeric_value = float(cleaned_value)
    except ValueError:
        return value.strip()

    if numeric_value.is_integer():
        return int(numeric_value)
    return numeric_value


def _is_blank_value(value: str) -> bool:
    return not value.strip()


def _metric_column(row: dict[str, str]) -> str | None:
    for column in ("Metric Name", "Metric", "Name"):
        if column in row:
            return column
    return None


def _value_column(row: dict[str, str]) -> str | None:
    for column in ("Metric Value", "Value"):
        if column in row:
            return column
    return None


def _unit_column(row: dict[str, str]) -> str | None:
    for column in ("Metric Unit", "Unit"):
        if column in row:
            return column
    return None


def _read_ncu_csv_rows(output: str) -> list[dict[str, str]]:
    csv_start = output.find('"ID"')
    if csv_start < 0:
        csv_start = output.find("ID,")
    if csv_start < 0:
        return []

    reader = csv.DictReader(StringIO(output[csv_start:]))
    return [
        {str(key): str(value) for key, value in row.items() if key is not None}
        for row in reader
    ]


def _collect_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    metrics: dict[str, dict[str, str]] = {}
    for row in rows:
        metric_column = _metric_column(row)
        value_column = _value_column(row)
        if metric_column is None or value_column is None:
            return _collect_wide_metrics(rows)

        metric_name = row.get(metric_column, "").strip()
        metric_value = row.get(value_column, "").strip()
        if not metric_name or not metric_value:
            continue

        metrics[metric_name] = row
    return metrics


def _collect_wide_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    metrics: dict[str, dict[str, str]] = {}
    units_by_metric = _collect_wide_units(rows)
    measurement_rows = _select_wide_measurement_rows(rows)
    for row in measurement_rows:
        for metric_name, metric_value in row.items():
            if _is_blank_value(metric_name) or _is_blank_value(metric_value):
                continue
            metrics.setdefault(
                metric_name.strip(),
                {
                    "Metric Name": metric_name.strip(),
                    "Metric Value": metric_value.strip(),
                    "Metric Unit": units_by_metric.get(metric_name.strip(), ""),
                },
            )
    return metrics


def _collect_wide_units(rows: list[dict[str, str]]) -> dict[str, str]:
    units: dict[str, str] = {}
    for row in rows:
        if row.get("ID", "").strip():
            continue
        for metric_name, unit in row.items():
            if _is_blank_value(metric_name) or _is_blank_value(unit):
                continue
            units[metric_name.strip()] = unit.strip()
    return units


def _select_wide_measurement_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    measurement_rows = [row for row in rows if row.get("ID", "").strip().isdigit()]
    candidate_rows = [
        row
        for row in measurement_rows
        if row.get("Kernel Name", "").strip() == "kernel"
    ]
    if candidate_rows:
        return candidate_rows
    return measurement_rows


def _insert_nested_value(
    report: dict[str, Any],
    path: str,
    value: Any,
) -> None:
    current = report
    keys = path.split(".")
    for key in keys[:-1]:
        child = current.setdefault(key, {})
        if not isinstance(child, dict):
            return
        current = child
    current[keys[-1]] = value


def _compact_metric(
    row: dict[str, str],
) -> int | float | str | dict[str, int | float | str]:
    value_column = _value_column(row)
    if value_column is None:
        return ""

    value = _parse_metric_value(row.get(value_column, ""))
    unit_column = _unit_column(row)
    unit = row.get(unit_column, "").strip() if unit_column is not None else ""
    if not unit:
        return value
    return {"value": value, "unit": unit}


def _sum_metric_values(
    values: list[int | float | str | dict[str, int | float | str]],
) -> int | float | str | dict[str, int | float | str] | list[Any]:
    numeric_values: list[int | float] = []
    unit = ""
    for value in values:
        raw_value = value.get("value") if isinstance(value, dict) else value
        if not isinstance(raw_value, (int, float)):
            return values
        numeric_values.append(raw_value)
        if isinstance(value, dict):
            unit_value = value.get("unit", "")
            if isinstance(unit_value, str) and unit_value:
                unit = unit_value

    total = sum(numeric_values)
    if float(total).is_integer():
        total = int(total)
    if not unit:
        return total
    return {"value": total, "unit": unit}


def _collect_messages(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for row in rows:
        section = row.get("Section Name", row.get("Section", "")).lower()
        joined_values = " ".join(value for value in row.values() if value).lower()
        if (
            "warning" not in section
            and "optimization" not in section
            and "warning" not in joined_values
            and "opportunity" not in joined_values
        ):
            continue

        compact_row = {
            key: value
            for key, value in row.items()
            if value and key in {"Section Name", "Rule Name", "Rule Type", "Message"}
        }
        if compact_row:
            messages.append(compact_row)

        if len(messages) >= _MAX_MESSAGES:
            break

    return messages


def _derive_sm_version(report: dict[str, Any]) -> None:
    hardware = report.get("hardware")
    if not isinstance(hardware, dict):
        return

    major = hardware.pop("sm_version_major", None)
    minor = hardware.pop("sm_version_minor", None)
    if major is None or minor is None:
        return

    major_value = major.get("value") if isinstance(major, dict) else major
    minor_value = minor.get("value") if isinstance(minor, dict) else minor
    hardware["sm_version"] = f"sm_{major_value}{minor_value}"


def _raw_metric_value(value: Any) -> Any:
    if isinstance(value, dict):
        return value.get("value")
    return value


def _numeric_metric_value(value: Any) -> float | None:
    raw_value = _raw_metric_value(value)
    if isinstance(raw_value, (int, float)):
        return float(raw_value)
    return None


def _parse_cuda_dim_tuple(value: Any) -> tuple[int, int, int] | None:
    raw_value = _raw_metric_value(value)
    if not isinstance(raw_value, str):
        return None

    dimensions = [int(match) for match in re.findall(r"\d+", raw_value)]
    if len(dimensions) != 3:
        return None
    return dimensions[0], dimensions[1], dimensions[2]


def _derive_kernel_dimensions(report: dict[str, Any]) -> None:
    kernel = report.get("kernel")
    if not isinstance(kernel, dict):
        return

    dimension_sources = {
        "block": _parse_cuda_dim_tuple(kernel.pop("block_size", None)),
        "grid": _parse_cuda_dim_tuple(kernel.pop("grid_size", None)),
    }
    for prefix, dimensions in dimension_sources.items():
        if dimensions is None:
            continue
        x_value, y_value, z_value = dimensions
        kernel[f"{prefix}_dim_x"] = x_value
        kernel[f"{prefix}_dim_y"] = y_value
        kernel[f"{prefix}_dim_z"] = z_value


def _derive_occupancy_limit(report: dict[str, Any]) -> None:
    occupancy = report.get("occupancy")
    if not isinstance(occupancy, dict):
        return

    limits = {
        key.removeprefix("limit_"): value
        for key, value in occupancy.items()
        if key.startswith("limit_")
    }
    numeric_limits: dict[str, float] = {}
    for key, value in limits.items():
        raw_value = value.get("value") if isinstance(value, dict) else value
        if isinstance(raw_value, (int, float)):
            numeric_limits[key] = float(raw_value)

    if numeric_limits:
        occupancy["primary_limit"] = min(
            numeric_limits,
            key=lambda limit_name: numeric_limits[limit_name],
        )


def _derive_theoretical_occupancy(report: dict[str, Any]) -> None:
    occupancy = report.get("occupancy")
    kernel = report.get("kernel")
    hardware = report.get("hardware")
    if (
        not isinstance(occupancy, dict)
        or not isinstance(kernel, dict)
        or not isinstance(hardware, dict)
        or "theoretical_pct" in occupancy
    ):
        return

    block_dim_x = _numeric_metric_value(kernel.get("block_dim_x"))
    block_dim_y = _numeric_metric_value(kernel.get("block_dim_y")) or 1
    block_dim_z = _numeric_metric_value(kernel.get("block_dim_z")) or 1
    warp_size = _numeric_metric_value(hardware.get("warp_size"))
    max_warps_per_sm = _numeric_metric_value(hardware.get("max_warps_per_sm"))
    limit_values = [
        _numeric_metric_value(occupancy.get("limit_blocks")),
        _numeric_metric_value(occupancy.get("limit_registers")),
        _numeric_metric_value(occupancy.get("limit_shared_memory")),
        _numeric_metric_value(occupancy.get("limit_warps")),
    ]
    numeric_limits = [value for value in limit_values if value is not None]
    if (
        block_dim_x is None
        or warp_size is None
        or max_warps_per_sm is None
        or not numeric_limits
    ):
        return

    threads_per_block = block_dim_x * block_dim_y * block_dim_z
    warps_per_block = -(-threads_per_block // warp_size)
    active_warps = min(numeric_limits) * warps_per_block
    occupancy["theoretical_pct"] = round(
        min(active_warps / max_warps_per_sm, 1.0) * 100,
        2,
    )


def _derive_memory_efficiency(report: dict[str, Any]) -> None:
    memory = report.get("memory")
    if not isinstance(memory, dict):
        return

    for access_kind in ("load", "store"):
        requests = _numeric_metric_value(memory.get(f"global_{access_kind}_requests"))
        sectors = _numeric_metric_value(memory.get(f"global_{access_kind}_sectors"))
        if requests and sectors is not None:
            memory[f"global_{access_kind}_sectors_per_request"] = round(
                sectors / requests,
                4,
            )

        shared_requests = _numeric_metric_value(
            memory.get(f"shared_{access_kind}_requests")
        )
        wavefronts = _numeric_metric_value(
            memory.get(f"shared_{access_kind}_wavefronts")
        )
        if shared_requests and wavefronts is not None:
            memory[f"shared_{access_kind}_ideal_wavefronts"] = int(shared_requests)
            memory[f"shared_{access_kind}_wavefronts_per_request"] = round(
                wavefronts / shared_requests,
                4,
            )

    for cache_level in ("l1", "l2"):
        for direction in ("read", "write"):
            sectors = _numeric_metric_value(
                memory.get(f"{cache_level}_{direction}_sectors")
            )
            if sectors is not None:
                memory[f"{cache_level}_bytes_{direction}"] = int(sectors * 32)


def _all_ncu_metrics(metrics: dict[str, dict[str, str]]) -> dict[str, Any]:
    return {
        metric_name: _compact_metric(row)
        for metric_name, row in sorted(metrics.items())
    }


def _parse_ncu_output(output: str) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = _read_ncu_csv_rows(output)
    if not rows:
        return {}, {}

    metrics = _collect_metrics(rows)
    report: dict[str, Any] = {}
    for output_path, aliases in _METRIC_ALIASES.items():
        matched_rows = [metrics[alias] for alias in aliases if alias in metrics]
        if not matched_rows:
            continue

        values = [_compact_metric(row) for row in matched_rows]
        if output_path in {
            "instructions.fp32",
            "instructions.flop_count_sp",
            "scheduler.stall_memory_dependency_pct",
        }:
            value = _sum_metric_values(values)
        else:
            value = values[0]
        _insert_nested_value(report, output_path, value)

    messages = _collect_messages(rows)
    if messages:
        report["warnings"] = messages

    _derive_sm_version(report)
    _derive_kernel_dimensions(report)
    _derive_occupancy_limit(report)
    _derive_theoretical_occupancy(report)
    _derive_memory_efficiency(report)
    return report, _all_ncu_metrics(metrics)


@dataclass(frozen=True)
class NCUResult:
    """Result of profiling one candidate with NVIDIA Nsight Compute.

    Attributes:
        summary: Compact optimization-oriented Nsight Compute metrics.
        metrics: Every named metric returned by Nsight Compute.
        error: Nsight Compute diagnostics written to standard error.
        return_code: Process exit status, or ``None`` when profiling did not run.
        source_report: CSV source-page output with SASS instruction correlation.
    """

    summary: dict[str, Any]
    metrics: dict[str, Any]
    error: str
    return_code: int | None
    source_report: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable profiling report.

        Returns:
            Availability, process status, and compact profiler metrics.
        """
        return {
            "available": self.return_code is not None,
            **asdict(self),
        }


def profile_ptx_with_ncu(
    kernel_name: str,
    payload: Payload,
    *,
    tuning_config: Mapping[str, int | bool] | None = None,
    timeout_seconds: int = 300,
) -> NCUResult:
    """Collect detailed hardware metrics for one PTX kernel launch.

    Args:
        kernel_name: Registered kernel class used to launch the candidate.
        tuning_config: Active operator tuning values used to derive the launch grid.
        payload: Compiled PTX and launch dimensions.
        timeout_seconds: Maximum Nsight Compute runtime in seconds.

    Returns:
        Nsight Compute JSON summary, diagnostics, and process status. A missing
        executable or launch failure is represented in the result instead of
        failing an otherwise valid evaluation.

    Raises:
        ValueError: If ``timeout_seconds`` is not positive.
    """
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if tuning_config is not None and not is_valid_tuning_config(tuning_config):
        raise ValueError(
            "tuning_config must map parameter names to booleans or positive integers."
        )

    ncu_path = os.environ.get(NCU_ENV_VAR)
    if ncu_path is None:
        return NCUResult(
            summary={},
            metrics={},
            error=f"ncu not found; set {NCU_ENV_VAR} to the executable path.",
            return_code=None,
        )

    request_dir = TMP_FILES_DIR / uuid4().hex
    request_dir.mkdir(parents=True)
    request_path = request_dir / "request.json"
    profile_path = request_dir / "profile.ncu-rep"
    source_path = request_dir / "candidate.ptx"
    try:
        source_path.write_text(payload.ptx, encoding="utf-8")
        profiled_payload = replace(
            payload,
            ptx=_annotate_ptx_lines(payload.ptx, source_path),
        )
        request_path.write_bytes(
            orjson.dumps(
                {
                    "kernel_name": kernel_name,
                    "candidate": {
                        **profiled_payload.to_launch_dict(),
                        **(
                            {"tuning_config": dict(tuning_config)}
                            if tuning_config
                            else {}
                        ),
                    },
                }
            )
        )
        try:
            completed = subprocess.run(
                [
                    ncu_path,
                    "--target-processes",
                    "application-only",
                    "--kernel-name",
                    f"regex:^{PROFILED_KERNEL_NAME}$",
                    "--set",
                    "full",
                    "--import-source",
                    "yes",
                    "--source-folders",
                    str(request_dir),
                    "--page",
                    "raw",
                    "--csv",
                    "--export",
                    str(profile_path),
                    "--force-overwrite",
                    sys.executable,
                    "-m",
                    "ptx_gym.evaluation.ncu_runner",
                    str(request_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            timeout_output = exc.stdout
            summary, metrics = _parse_ncu_output(
                timeout_output.decode(errors="replace")
                if isinstance(timeout_output, bytes)
                else timeout_output or ""
            )
            return NCUResult(
                summary=summary,
                metrics=metrics,
                error=f"Nsight Compute timed out after {timeout_seconds} seconds.",
                return_code=None,
            )
        except OSError as exc:
            return NCUResult(
                summary={},
                metrics={},
                error=f"Failed to start Nsight Compute: {exc}",
                return_code=None,
            )

        summary, metrics = _parse_ncu_output(completed.stdout)
        source_report = ""
        if completed.returncode == 0 and profile_path.exists():
            try:
                source_completed = subprocess.run(
                    [
                        ncu_path,
                        "--import",
                        str(profile_path),
                        "--page",
                        "source",
                        "--print-source",
                        "cuda,sass",
                        "--csv",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                completed.stderr = (
                    f"{completed.stderr}\nNCU source export failed: {exc}"
                ).strip()
            else:
                source_report = source_completed.stdout
                if source_completed.returncode != 0:
                    source_error = source_completed.stderr.strip()
                    if source_error:
                        completed.stderr = f"{completed.stderr}\n{source_error}".strip()
        return NCUResult(
            summary=summary,
            metrics=metrics,
            error=completed.stderr,
            return_code=completed.returncode,
            source_report=source_report,
        )
    finally:
        shutil.rmtree(request_dir)
