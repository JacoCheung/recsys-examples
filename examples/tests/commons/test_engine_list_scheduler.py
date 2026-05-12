# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for ``schedule_tasks``."""

import pytest
from commons.pipeline.engine import DataSlot, SchedulablePipeline, StreamPool, Task
from commons.pipeline.engine.autosched import CostModel, schedule_tasks, validate


def _noop(ctx):
    return None


def _costs(**kwargs) -> CostModel:
    return CostModel.from_dict(
        {name: {"cpu_us": 0.0, "gpu_us": float(cost)} for name, cost in kwargs.items()}
    )


def _names(schedule):
    return [task.name for task in schedule.stages[0].tasks]


def test_scheduler_rejects_empty_and_duplicate_inputs() -> None:
    with pytest.raises(ValueError, match="no tasks"):
        schedule_tasks([], CostModel({}), stream_slots=("default",))

    a = Task.from_fn("dup", _noop)
    b = Task.from_fn("dup", _noop)
    with pytest.raises(ValueError, match="Duplicate task name"):
        schedule_tasks([a, b], _costs(dup=1), stream_slots=("default",))


def test_scheduler_orders_slot_dependency_chain() -> None:
    a = Task.from_fn("a", _noop, writes=("x",))
    b = Task.from_fn("b", _noop, reads=("x",), writes=("y",))
    c = Task.from_fn("c", _noop, reads=("y",))

    schedule = schedule_tasks([c, a, b], _costs(a=1, b=1, c=1), ("default",))

    assert _names(schedule) == ["a", "b", "c"]


def test_scheduler_prioritizes_longer_independent_critical_path() -> None:
    a = Task.from_fn("a", _noop, writes=("x_a",))
    b = Task.from_fn("b", _noop, reads=("x_a",))
    c = Task.from_fn("c", _noop, writes=("x_c",))
    d = Task.from_fn("d", _noop, reads=("x_c",))

    schedule = schedule_tasks(
        [d, c, b, a],
        _costs(a=100, b=100, c=1, d=1),
        ("default",),
    )
    names = _names(schedule)

    assert names.index("a") < names.index("c")
    assert names.index("a") < names.index("b")
    assert names.index("c") < names.index("d")


def test_scheduler_rejects_cycles_and_unknown_depends_on() -> None:
    a = Task.from_fn("a", _noop, depends_on=("b",))
    b = Task.from_fn("b", _noop, depends_on=("a",))
    with pytest.raises(ValueError, match="cycle"):
        schedule_tasks([a, b], _costs(a=1, b=1), ("default",))

    unknown = Task.from_fn("unknown", _noop, depends_on=("missing",))
    with pytest.raises(ValueError, match="not in the input task list"):
        schedule_tasks([unknown], _costs(unknown=1), ("default",))


def test_scheduler_output_passes_validator_for_prefetch_shape() -> None:
    h2d = Task.from_fn(
        "h2d",
        _noop,
        stream="memcpy",
        lookahead=1,
        reads=(DataSlot("batch_cpu", 1),),
        writes=(DataSlot("batch_gpu", 1),),
    )
    forward = Task.from_fn("forward", _noop, reads=("batch_gpu",), writes=("loss",))
    backward = Task.from_fn("backward", _noop, reads=("loss",), depends_on=("forward",))

    schedule = schedule_tasks(
        [backward, forward, h2d],
        _costs(h2d=1, forward=10, backward=10),
        ("default", "memcpy"),
    )

    validate(schedule)
    names = _names(schedule)
    assert names.index("h2d") < names.index("forward") < names.index("backward")


def test_scheduled_schedule_executes_end_to_end() -> None:
    def _forward(ctx):
        ctx.slots.set("step_result", ctx.slots["batch_cpu"] + 1)

    forward = Task.from_fn("forward", _forward, writes=("step_result",))
    extra = Task.from_fn("extra", _noop, depends_on=("forward",))
    schedule = schedule_tasks(
        [extra, forward], _costs(forward=10, extra=1), ("default",)
    )
    pipe = SchedulablePipeline(schedule, StreamPool({"default": None}))

    assert [pipe.progress(iter([i])) for i in range(3)] == [1, 2, 3]
