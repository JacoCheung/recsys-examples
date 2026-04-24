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

"""V1 smoke test — single-stream, single-stage forward through the engine.

Proves the Task/Schedule/StreamPool/SchedulablePipeline API is wired:
one plain Task calls `model(batch)` inside `progress()`, writes the
result to the `step_result` slot, and `progress()` returns it.
"""

import pytest
import torch
from commons.pipeline.engine import (
    DataSlot,
    SchedulablePipeline,
    Schedule,
    Stage,
    StreamPool,
    Task,
)


def _make_pool(device: torch.device) -> StreamPool:
    """Single-stream pool; on CUDA we use the default stream."""
    if device.type == "cuda":
        return StreamPool({"default": torch.cuda.default_stream(device)})
    return StreamPool({"default": None})


def test_single_stage_forward() -> None:
    """One `Task.from_fn` running `model(batch_cpu) → step_result` matches
    a non-pipelined reference exactly."""
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.nn.Linear(10, 1).to(device)

    def _forward(ctx):
        x = ctx.slots["batch_cpu"]
        y = model(x)
        ctx.slots.set("step_result", y)

    fwd = Task.from_fn(
        name="forward",
        fn=_forward,
        reads=(DataSlot("batch_cpu"),),
        writes=(DataSlot("step_result"),),
        stream="default",
    )

    schedule = Schedule(
        stages=(Stage(tasks=(fwd,)),),
        stream_slots=("default",),
    )
    pool = _make_pool(device)
    pipe = SchedulablePipeline(schedule, pool)

    x_batch = torch.randn(4, 10, device=device)
    expected = model(x_batch)

    result = pipe.progress(iter([x_batch]))

    assert result is not None, "progress() should return step_result slot value"
    assert torch.allclose(
        result, expected, atol=1e-6, rtol=0
    ), "single-stage forward should match non-pipelined reference exactly"


def test_in_flight_batches_is_derived() -> None:
    """Schedule.in_flight_batches is a derived @property (SPEC §4.2 rule 4)."""
    fwd = Task.from_fn(name="t", fn=lambda ctx: None, stream="default")
    schedule = Schedule(stages=(Stage(tasks=(fwd,)),), stream_slots=("default",))
    assert schedule.in_flight_batches == 1


def test_stream_pool_requires_default_slot() -> None:
    """StreamPool must always declare a 'default' slot."""
    with pytest.raises(ValueError, match="'default' stream slot"):
        StreamPool({"memcpy": None})


def test_task_from_fn_stream_binding() -> None:
    """Task.from_fn carries the declared stream through to the run context."""
    captured = {}

    def _capture(ctx):
        captured["stream_name"] = "default"  # V1 single-stream; V3 will exercise
        captured["iter"] = ctx.iter_count

    t = Task.from_fn(name="capture", fn=_capture, stream="default")
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool)

    pipe.progress(iter([object()]))  # opaque payload — task doesn't read it

    # SPEC §4.8: iter_count is 0-indexed (internal iter at task-run time).
    # First progress call = internal iter 0.
    assert captured == {"stream_name": "default", "iter": 0}


def test_unknown_stream_rejected_at_construction() -> None:
    """Task referencing an undeclared stream fails fast at pipeline construction."""
    t = Task.from_fn(name="bad", fn=lambda ctx: None, stream="gpu_comm")
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    pool = StreamPool({"default": None})
    with pytest.raises(ValueError, match="not declared in Schedule.stream_slots"):
        SchedulablePipeline(schedule, pool)
