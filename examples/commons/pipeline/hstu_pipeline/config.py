# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule config parsing for the HSTU pipeline adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from commons.pipeline.engine import SameProgressSyncSide
from commons.pipeline.engine.task import _SameProgressSync

HSTU_PIPELINE_CONFIG_ENV = "HSTU_PIPELINE_CONFIG"

HSTU_PIPELINE_TASKS = frozenset(
    (
        "h2d",
        "start_shuffle",
        "finish_shuffle",
        "start_input_dist",
        "wait_input_dist",
        "prefetch_embeddings",
        "zero_grad",
        "global_tokens_allreduce",
        "compute_output_dist",
        "ranking_embedding_forward",
        "dense_forward",
        "backward",
        "finalize_model_grads",
        "optimizer_step",
        "watchdog_step",
    )
)


@dataclass
class HSTUPipelineScheduleConfig:
    """Externalized HSTU pipeline scheduling knobs.

    The config owns only HSTU adapter choices. It intentionally does not
    change the framework-agnostic engine API.
    """

    thread_map: Any = None
    lookahead: Dict[str, int] = field(default_factory=dict)
    same_progress_sync: Dict[str, Tuple[_SameProgressSync, ...]] = field(
        default_factory=dict
    )
    source: Optional[str] = None

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any],
        *,
        source: Optional[str] = None,
    ) -> "HSTUPipelineScheduleConfig":
        unknown = set(raw) - {
            "version",
            "thread_map",
            "lookahead",
            "same_progress_sync",
        }
        if unknown:
            raise ValueError(
                f"Unknown HSTU pipeline config field(s): {sorted(unknown)}"
            )
        version = raw.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported HSTU pipeline config version: {version!r}")

        same_progress = _parse_same_progress_sync(raw.get("same_progress_sync"))
        thread_map = raw.get("thread_map")
        _validate_thread_map_config(thread_map)
        return cls(
            thread_map=thread_map,
            lookahead=_parse_lookahead(raw.get("lookahead")),
            same_progress_sync=same_progress,
            source=source,
        )

    def lookahead_for(self, task_name: str, default: Optional[int]) -> Optional[int]:
        return self.lookahead.get(task_name, default)

    def same_progress_for(
        self,
        task_name: str,
        default: Tuple[Any, ...],
    ) -> Tuple[Any, ...]:
        return self.same_progress_sync.get(task_name, default)


def load_hstu_pipeline_schedule_config(
    value: Optional[Any],
) -> Optional[HSTUPipelineScheduleConfig]:
    if value is None or value == "":
        return None
    if isinstance(value, HSTUPipelineScheduleConfig):
        return value
    if isinstance(value, Mapping):
        return HSTUPipelineScheduleConfig.from_mapping(value)

    raw = str(value).strip()
    if not raw:
        return None
    if raw.startswith("{"):
        data = json.loads(raw)
        source = "<inline-json>"
    else:
        path = Path(raw).expanduser()
        data = json.loads(path.read_text())
        source = str(path)
    if not isinstance(data, Mapping):
        raise TypeError(
            f"HSTU pipeline config must decode to a JSON object, got {type(data).__name__}"
        )
    return HSTUPipelineScheduleConfig.from_mapping(data, source=source)


def _validate_task_name(name: str, *, field_name: str) -> None:
    if name not in HSTU_PIPELINE_TASKS:
        raise ValueError(
            f"Unknown HSTU task {name!r} in {field_name}. "
            f"Known: {sorted(HSTU_PIPELINE_TASKS)}"
        )


def _validate_thread_map_config(value: Any) -> None:
    if value is None or isinstance(value, str):
        return
    if not isinstance(value, Mapping):
        raise TypeError(
            "thread_map must be a preset string, task-to-thread object, or null; "
            f"got {type(value).__name__}"
        )
    for task_name, thread_name in value.items():
        if not isinstance(task_name, str) or not isinstance(thread_name, str):
            raise TypeError("thread_map object entries must be string:string")
        _validate_task_name(task_name, field_name="thread_map")


def _parse_lookahead(value: Any) -> Dict[str, int]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"lookahead must be a task-to-int object, got {type(value).__name__}"
        )
    parsed: Dict[str, int] = {}
    for task_name, lookahead in value.items():
        if not isinstance(task_name, str):
            raise TypeError("lookahead task names must be strings")
        _validate_task_name(task_name, field_name="lookahead")
        if isinstance(lookahead, bool) or not isinstance(lookahead, int):
            raise TypeError(
                f"lookahead[{task_name!r}] must be an integer, got {lookahead!r}"
            )
        if lookahead < 0:
            raise ValueError(
                f"lookahead[{task_name!r}] must be non-negative, got {lookahead}"
            )
        parsed[task_name] = lookahead
    return parsed


def _parse_same_progress_sync(
    value: Any,
) -> Dict[str, Tuple[_SameProgressSync, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(
            f"same_progress_sync must be an object, got {type(value).__name__}"
        )

    task_specs: Dict[str, Tuple[_SameProgressSync, ...]] = {}

    for task_name, raw_sync in value.items():
        if not isinstance(task_name, str):
            raise TypeError("same_progress_sync task names must be strings")
        _validate_task_name(task_name, field_name="same_progress_sync")
        task_specs[task_name] = _parse_same_progress_sync_task(
            raw_sync, field_name=f"same_progress_sync.{task_name}"
        )
    return task_specs


def _parse_same_progress_sync_task(
    value: Any,
    *,
    field_name: str,
) -> Tuple[_SameProgressSync, ...]:
    if value is None or value is False:
        return ()
    if isinstance(value, str):
        return (_parse_same_progress_sync_edge(value, field_name=field_name),)
    if isinstance(value, Mapping):
        return (_parse_same_progress_sync_edge(value, field_name=field_name),)
    if isinstance(value, (list, tuple)):
        return tuple(
            _parse_same_progress_sync_edge(
                raw_edge, field_name=f"{field_name}[{index}]"
            )
            for index, raw_edge in enumerate(value)
        )
    raise TypeError(
        f"{field_name} must be a task name, edge object, or edge list, "
        f"got {type(value).__name__}"
    )


def _parse_same_progress_sync_edge(value: Any, *, field_name: str) -> _SameProgressSync:
    if isinstance(value, str):
        dep_name = value
        side = SameProgressSyncSide.BOTH
    elif isinstance(value, Mapping):
        unknown = set(value) - {"task", "name", "sides"}
        if unknown:
            raise ValueError(f"Unknown {field_name} field(s): {sorted(unknown)}")
        has_task = "task" in value
        has_name = "name" in value
        if has_task == has_name:
            raise ValueError(f"{field_name} must contain exactly one of task/name")
        dep_name = value["task"] if has_task else value["name"]
        side = _parse_side(value.get("sides", "both"), field_name=f"{field_name}.sides")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        dep_name, raw_side = value
        side = _parse_side(raw_side, field_name=f"{field_name}.sides")
    else:
        raise TypeError(
            f"{field_name} entries must be task names or (task, side) edges; "
            f"got {type(value).__name__}: {value!r}"
        )
    if not isinstance(dep_name, str):
        raise TypeError(
            f"{field_name}.task must be a string, got {type(dep_name).__name__}"
        )
    _validate_task_name(dep_name, field_name=field_name)
    return _SameProgressSync(dep_name, side)


def _parse_side(value: Any, *, field_name: str) -> SameProgressSyncSide:
    if isinstance(value, SameProgressSyncSide):
        side = value
        _validate_side_bits(side, field_name=field_name)
        return side
    if isinstance(value, int) and not isinstance(value, bool):
        if value < 0:
            raise ValueError(f"{field_name}={value} contains unknown bits")
        side = SameProgressSyncSide(value)
        _validate_side_bits(side, field_name=field_name)
        return side
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be one of both/cpu/gpu/none or an int flag, "
            f"got {type(value).__name__}"
        )

    normalized = value.strip().lower().replace("_", "-")
    if normalized in {"", "none", "off", "0", "-"}:
        return SameProgressSyncSide.NONE
    if normalized in {"both", "cpu+gpu", "gpu+cpu", "cpu,gpu", "gpu,cpu"}:
        return SameProgressSyncSide.BOTH
    if normalized == "cpu":
        return SameProgressSyncSide.CPU
    if normalized == "gpu":
        return SameProgressSyncSide.GPU

    side = SameProgressSyncSide.NONE
    for token in normalized.replace(",", "+").split("+"):
        if token == "cpu":
            side |= SameProgressSyncSide.CPU
        elif token == "gpu":
            side |= SameProgressSyncSide.GPU
        else:
            raise ValueError(f"Unknown {field_name} side token: {token!r}")
    return side


def _validate_side_bits(side: SameProgressSyncSide, *, field_name: str) -> None:
    valid_mask = int(SameProgressSyncSide.BOTH)
    if int(side) & ~valid_mask:
        raise ValueError(
            f"{field_name}={int(side)} contains unknown bits; "
            f"valid bits are CPU={int(SameProgressSyncSide.CPU)} and "
            f"GPU={int(SameProgressSyncSide.GPU)}"
        )
