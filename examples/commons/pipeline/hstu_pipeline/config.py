# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Schedule config parsing for the HSTU pipeline adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from commons.pipeline.engine import SameProgressSyncSide

HSTU_PIPELINE_CONFIG_ENV = "HSTU_PIPELINE_CONFIG"

HSTU_NONCRITICAL_TASKS: Tuple[str, ...] = (
    "h2d",
    "start_shuffle",
    "finish_shuffle",
    "start_input_dist",
    "wait_input_dist",
    "prefetch_embeddings",
)

HSTU_PIPELINE_TASKS = frozenset(
    HSTU_NONCRITICAL_TASKS
    + (
        "zero_grad",
        "global_tokens_allreduce",
        "compute_output_dist",
        "ranking_embedding_forward",
        "forward",
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
    same_progress_sync: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    noncritical_same_progress_sync: Optional[Tuple[str, ...]] = None
    same_progress_sync_sides: Dict[str, SameProgressSyncSide] = field(
        default_factory=dict
    )
    default_same_progress_sync_side: SameProgressSyncSide = SameProgressSyncSide.BOTH
    split_ranking_forward: Optional[bool] = None
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
            "same_progress_sync_sides",
            "split_ranking_forward",
        }
        if unknown:
            raise ValueError(
                f"Unknown HSTU pipeline config field(s): {sorted(unknown)}"
            )
        version = raw.get("version", 1)
        if version != 1:
            raise ValueError(f"Unsupported HSTU pipeline config version: {version!r}")

        same_progress = _parse_same_progress_sync(raw.get("same_progress_sync"))
        side_default, side_tasks = _parse_same_progress_sync_sides(
            raw.get("same_progress_sync_sides")
        )
        split = (
            None
            if "split_ranking_forward" not in raw
            else _parse_bool(
                raw["split_ranking_forward"], field_name="split_ranking_forward"
            )
        )
        thread_map = raw.get("thread_map")
        _validate_thread_map_config(thread_map)
        return cls(
            thread_map=thread_map,
            lookahead=_parse_lookahead(raw.get("lookahead")),
            same_progress_sync=same_progress[1],
            noncritical_same_progress_sync=same_progress[0],
            same_progress_sync_sides=side_tasks,
            default_same_progress_sync_side=side_default,
            split_ranking_forward=split,
            source=source,
        )

    def lookahead_for(self, task_name: str, default: Optional[int]) -> Optional[int]:
        return self.lookahead.get(task_name, default)

    def same_progress_for(
        self,
        task_name: str,
        default: Tuple[str, ...],
        *,
        noncritical_tasks: Tuple[str, ...] = HSTU_NONCRITICAL_TASKS,
    ) -> Tuple[str, ...]:
        if task_name in self.same_progress_sync:
            return self.same_progress_sync[task_name]
        if (
            task_name in noncritical_tasks
            and self.noncritical_same_progress_sync is not None
        ):
            return self.noncritical_same_progress_sync
        return default

    def same_progress_side_for(self, task_name: str) -> SameProgressSyncSide:
        return self.same_progress_sync_sides.get(
            task_name, self.default_same_progress_sync_side
        )


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


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        if value in (0, 1):
            return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"{field_name} must be a boolean, got {value!r}")


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


def _parse_name_tuple(value: Any, *, field_name: str) -> Tuple[str, ...]:
    if value is None or value is False:
        return ()
    if isinstance(value, int) and not isinstance(value, bool):
        if value == 0:
            return ()
        raise TypeError(f"{field_name} must be a task name list, got {value!r}")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"none", "off", "0", "-"}:
            return ()
        names = tuple(part.strip() for part in stripped.split("+") if part.strip())
    elif isinstance(value, (list, tuple)):
        names = tuple(value)
    else:
        raise TypeError(
            f"{field_name} must be a task name string/list, got {type(value).__name__}"
        )

    for name in names:
        if not isinstance(name, str):
            raise TypeError(f"{field_name} entries must be strings, got {name!r}")
        _validate_task_name(name, field_name=field_name)
    return names


def _parse_same_progress_sync(
    value: Any,
) -> Tuple[Optional[Tuple[str, ...]], Dict[str, Tuple[str, ...]]]:
    if value is None:
        return None, {}
    if not isinstance(value, Mapping):
        raise TypeError(
            "same_progress_sync must be an object with noncritical_default "
            f"and/or tasks, got {type(value).__name__}"
        )

    noncritical_default: Optional[Tuple[str, ...]] = None
    task_specs: Dict[str, Tuple[str, ...]] = {}
    if "noncritical_default" in value:
        noncritical_default = _parse_name_tuple(
            value["noncritical_default"],
            field_name="same_progress_sync.noncritical_default",
        )

    if "tasks" in value:
        tasks_value = value["tasks"]
        if not isinstance(tasks_value, Mapping):
            raise TypeError("same_progress_sync.tasks must be an object")
        task_items = tasks_value.items()
    else:
        task_items = (
            (key, raw)
            for key, raw in value.items()
            if key not in {"noncritical_default"}
        )

    for task_name, raw_sync in task_items:
        if not isinstance(task_name, str):
            raise TypeError("same_progress_sync task names must be strings")
        _validate_task_name(task_name, field_name="same_progress_sync")
        task_specs[task_name] = _parse_name_tuple(
            raw_sync, field_name=f"same_progress_sync.tasks.{task_name}"
        )
    return noncritical_default, task_specs


def _parse_same_progress_sync_sides(
    value: Any,
) -> Tuple[SameProgressSyncSide, Dict[str, SameProgressSyncSide]]:
    if value is None:
        return SameProgressSyncSide.BOTH, {}
    if not isinstance(value, Mapping):
        return _parse_side(value, field_name="same_progress_sync_sides"), {}

    default_side = SameProgressSyncSide.BOTH
    task_sides: Dict[str, SameProgressSyncSide] = {}
    if "default" in value:
        default_side = _parse_side(
            value["default"], field_name="same_progress_sync_sides.default"
        )
    if "tasks" in value:
        tasks_value = value["tasks"]
        if not isinstance(tasks_value, Mapping):
            raise TypeError("same_progress_sync_sides.tasks must be an object")
        side_items = tasks_value.items()
    else:
        side_items = (
            (key, raw) for key, raw in value.items() if key not in {"default"}
        )
    for task_name, raw_side in side_items:
        if not isinstance(task_name, str):
            raise TypeError("same_progress_sync_sides task names must be strings")
        _validate_task_name(task_name, field_name="same_progress_sync_sides")
        task_sides[task_name] = _parse_side(
            raw_side, field_name=f"same_progress_sync_sides.tasks.{task_name}"
        )
    return default_side, task_sides


def _parse_side(value: Any, *, field_name: str) -> SameProgressSyncSide:
    if isinstance(value, SameProgressSyncSide):
        side = value
        _validate_side_bits(side, field_name=field_name)
        return side
    if isinstance(value, int) and not isinstance(value, bool):
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
