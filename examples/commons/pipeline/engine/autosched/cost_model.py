# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline cost model for the auto-scheduler (SPEC §4.3).

Two pieces:

- `CostProfiler` — instruments a user-supplied default schedule with
  CUDA events, runs N warmup iterations, dumps per-task durations as
  JSON.
- `CostModel` — loads a JSON dump (or a hand-edited dict), offers
  per-task cost lookups for the list scheduler.

The JSON format is flat:

```json
{
    "task_name_a": {"cpu_us": 12.3, "gpu_us": 456.7},
    "task_name_b": {"cpu_us": 8.9,  "gpu_us": 1234.5}
}
```

Units are microseconds (μs). CPU time is wall-clock around
`task.run(ctx)`; GPU time is between pre- and post-`run` CUDA events
on the task's declared stream.
"""

import json
import pathlib
import time
from typing import Any, Dict, Iterable, Mapping

import torch

__all__ = ["TaskCost", "CostModel", "CostProfiler"]


class TaskCost:
    """Per-task timing in microseconds. Immutable, hashable."""

    __slots__ = ("cpu_us", "gpu_us")

    def __init__(self, cpu_us: float, gpu_us: float) -> None:
        if cpu_us < 0 or gpu_us < 0:
            raise ValueError(
                f"TaskCost durations must be non-negative; got "
                f"cpu_us={cpu_us}, gpu_us={gpu_us}"
            )
        self.cpu_us = float(cpu_us)
        self.gpu_us = float(gpu_us)

    @property
    def total_us(self) -> float:
        """Max of CPU and GPU time — the limiting factor for
        critical-path analysis."""
        return max(self.cpu_us, self.gpu_us)

    def to_dict(self) -> Dict[str, float]:
        return {"cpu_us": self.cpu_us, "gpu_us": self.gpu_us}

    def __repr__(self) -> str:
        return f"TaskCost(cpu_us={self.cpu_us:.1f}, gpu_us={self.gpu_us:.1f})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TaskCost):
            return NotImplemented
        return self.cpu_us == other.cpu_us and self.gpu_us == other.gpu_us

    def __hash__(self) -> int:
        return hash((self.cpu_us, self.gpu_us))


class CostModel:
    """Per-task cost lookup. Load from JSON (`CostModel.from_json`),
    from a dict (`CostModel({...})`), or build from a `CostProfiler`
    result."""

    def __init__(self, costs: Mapping[str, TaskCost]) -> None:
        self._costs: Dict[str, TaskCost] = dict(costs)

    def get(self, task_name: str) -> TaskCost:
        if task_name not in self._costs:
            raise KeyError(
                f"No cost entry for task {task_name!r}. Available: "
                f"{sorted(self._costs.keys())}."
            )
        return self._costs[task_name]

    def task_names(self) -> Iterable[str]:
        return self._costs.keys()

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {name: cost.to_dict() for name, cost in self._costs.items()}

    def save_json(self, path: pathlib.Path) -> None:
        path = pathlib.Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    @classmethod
    def from_dict(cls, raw: Mapping[str, Mapping[str, float]]) -> "CostModel":
        costs = {
            name: TaskCost(cpu_us=entry["cpu_us"], gpu_us=entry["gpu_us"])
            for name, entry in raw.items()
        }
        return cls(costs)

    @classmethod
    def from_json(cls, path: pathlib.Path) -> "CostModel":
        path = pathlib.Path(path)
        raw = json.loads(path.read_text())
        return cls.from_dict(raw)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CostModel):
            return NotImplemented
        return self._costs == other._costs


class CostProfiler:
    """Offline warmup profiler. Given a pipe and an iterator that
    supplies N batches, measure per-task CPU wall-clock and GPU
    event time, aggregate (mean) across N iters.

    Usage:

        profiler = CostProfiler(pipe)
        profiler.run(batch_iter, steps=5)
        model = profiler.as_cost_model()
        model.save_json(pathlib.Path("cost.json"))

    CPU timing: `time.perf_counter_ns()` around `task.run(ctx)`.
    GPU timing: `torch.cuda.Event(enable_timing=True)` records
    around `task.run`, with `.elapsed_time(...)` giving ms.

    The profiler monkey-patches each task's `.run` method to
    inject pre/post instrumentation — it does NOT mutate the engine
    driver (`_run_one_internal_iter`) itself. The patches are
    reverted in a `finally` block after `run()` so nothing sticks
    if instrumentation raises mid-iteration.
    """

    def __init__(self, pipe: Any) -> None:
        self._pipe = pipe
        # Running sums per task_name: (cpu_ns, gpu_ms, count).
        self._agg: Dict[str, Dict[str, float]] = {}

    def _record(self, task_name: str, cpu_ns: float, gpu_ms: float) -> None:
        entry = self._agg.setdefault(
            task_name, {"cpu_ns": 0.0, "gpu_ms": 0.0, "count": 0.0}
        )
        entry["cpu_ns"] += cpu_ns
        entry["gpu_ms"] += gpu_ms
        entry["count"] += 1

    def run(self, batch_iter: Any, steps: int = 5) -> None:
        """Drive `pipe.progress(batch_iter)` for `steps` iterations,
        recording per-task CPU and GPU time. Idempotent — multiple
        calls accumulate into the same aggregator (call reset() to
        restart)."""
        pipe = self._pipe
        cuda_available = torch.cuda.is_available()

        # Walk the schedule's stages/tasks once to build a name→task
        # lookup we can patch.
        schedule = pipe._schedule  # intentional internal access
        tasks = {t.name: t for stage in schedule.stages for t in stage.tasks}

        # Wrap each task's `run` method to capture timings. Restore
        # originals in a finally block so partial failure doesn't
        # corrupt the pipe.
        originals: Dict[str, Any] = {}
        try:
            for name, task in tasks.items():
                originals[name] = task.run

                def _make_wrapper(task_name=name, orig=task.run):
                    def _wrapped(ctx):
                        start_evt = (
                            torch.cuda.Event(enable_timing=True)
                            if cuda_available
                            else None
                        )
                        end_evt = (
                            torch.cuda.Event(enable_timing=True)
                            if cuda_available
                            else None
                        )
                        if start_evt is not None:
                            start_evt.record()
                        cpu_start = time.perf_counter_ns()
                        orig(ctx)
                        cpu_end = time.perf_counter_ns()
                        if end_evt is not None:
                            end_evt.record()
                        cpu_ns = float(cpu_end - cpu_start)
                        gpu_ms = 0.0
                        if start_evt is not None and end_evt is not None:
                            end_evt.synchronize()
                            gpu_ms = float(start_evt.elapsed_time(end_evt))
                        self._record(task_name, cpu_ns, gpu_ms)

                    return _wrapped

                task.run = _make_wrapper()

            for _ in range(steps):
                try:
                    pipe.progress(batch_iter)
                except StopIteration:
                    break
        finally:
            for name, orig in originals.items():
                tasks[name].run = orig

    def as_cost_model(self) -> CostModel:
        """Materialize the running aggregates as a `CostModel`.
        Takes the mean per task."""
        costs: Dict[str, TaskCost] = {}
        for name, entry in self._agg.items():
            count = entry["count"]
            if count == 0:
                continue
            cpu_us = (entry["cpu_ns"] / count) / 1000.0  # ns → us
            gpu_us = (entry["gpu_ms"] / count) * 1000.0  # ms → us
            costs[name] = TaskCost(cpu_us=cpu_us, gpu_us=gpu_us)
        return CostModel(costs)

    def reset(self) -> None:
        self._agg.clear()
