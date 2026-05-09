# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the SPEC_p4 v2 declarative ``Pipeline`` wrapper.

``Pipeline(tasks=[...], stream_pool=...)`` is the user-facing entry
point: a flat list of tasks plus a stream pool, with the engine
deriving stage layout, stream slots, and ring depth automatically.
The wrapper delegates execution to ``SchedulablePipeline``.

These tests run on CPU-only Tasks (no real CUDA work in their bodies)
so they don't require a GPU; they exercise the wrapper's construction
contract, derived schedule shape, and end-to-end progress() flow.
"""

import pytest
from commons.pipeline.engine import (
    DataSlot,
    Pipeline,
    SchedulablePipeline,
    Schedule,
    Stage,
    StreamPool,
    Task,
)


def _cpu_pool() -> StreamPool:
    """StreamPool that maps every slot to None — fine on CPU and on
    CUDA hosts (None resolves to the anchor-device default at use time)."""
    return StreamPool({"default": None, "memcpy": None})


def _src(ctx) -> None:
    ctx.slots.set("x", 7)


def _add(ctx) -> None:
    ctx.slots.set("step_result", ctx.slots["x"] + 3)


# ---------------------------------------------------------------------
# Construction contract
# ---------------------------------------------------------------------


def test_pipeline_empty_task_list_rejected() -> None:
    with pytest.raises(ValueError, match="at least one task"):
        Pipeline(tasks=(), stream_pool=_cpu_pool())


def test_pipeline_non_task_entry_rejected() -> None:
    with pytest.raises(TypeError, match="must be Task instances"):
        Pipeline(tasks=("not a task",), stream_pool=_cpu_pool())  # type: ignore[arg-type]


def test_pipeline_derives_schedule_in_flight_from_lookahead() -> None:
    """Schedule.in_flight_batches = max(lookahead) + 1."""
    src = Task.from_fn("src", _src, lookahead=2, writes=("x",))
    add = Task.from_fn("add", _add, lookahead=0, reads=("x",), writes=("step_result",))
    pipe = Pipeline(tasks=(src, add), stream_pool=_cpu_pool())

    assert pipe.schedule.in_flight_batches == 3  # max(2, 0) + 1


def test_pipeline_stream_slots_union_of_task_streams() -> None:
    """Pipeline.schedule.stream_slots is the sorted union of declared
    streams plus 'default' (so the engine always has an anchor)."""
    a = Task.from_fn("a", _src, stream="memcpy", writes=("x",))
    b = Task.from_fn("b", _add, stream="default", reads=("x",), writes=("step_result",))
    pipe = Pipeline(tasks=(a, b), stream_pool=_cpu_pool())

    assert pipe.schedule.stream_slots == ("default", "memcpy")


def test_pipeline_default_always_present_even_if_tasks_dont_use_it() -> None:
    a = Task.from_fn("a", _src, stream="memcpy", writes=("x",))
    b = Task.from_fn("b", _add, stream="memcpy", reads=("x",), writes=("step_result",))
    pipe = Pipeline(tasks=(a, b), stream_pool=_cpu_pool())

    assert "default" in pipe.schedule.stream_slots


def test_pipeline_single_stage_layout() -> None:
    """All tasks land in one Stage in declaration order."""
    src = Task.from_fn("src", _src, writes=("x",))
    add = Task.from_fn("add", _add, reads=("x",), writes=("step_result",))
    pipe = Pipeline(tasks=(src, add), stream_pool=_cpu_pool())

    assert len(pipe.schedule.stages) == 1
    assert tuple(t.name for t in pipe.schedule.stages[0].tasks) == ("src", "add")


def test_pipeline_iterable_input_accepts_generator() -> None:
    """tasks= can be any Iterable[Task], not just a tuple/list."""

    def task_gen():
        yield Task.from_fn("src", _src, writes=("x",))
        yield Task.from_fn("add", _add, reads=("x",), writes=("step_result",))

    pipe = Pipeline(tasks=task_gen(), stream_pool=_cpu_pool())
    assert len(pipe.schedule.all_tasks()) == 2


# ---------------------------------------------------------------------
# End-to-end progress() flow
# ---------------------------------------------------------------------


def test_pipeline_progress_returns_step_result() -> None:
    """Two-task DAG: src writes "x"=7, add writes step_result=x+3=10."""
    src = Task.from_fn("src", _src, writes=("x",))
    add = Task.from_fn("add", _add, reads=("x",), writes=("step_result",))
    pipe = Pipeline(tasks=(src, add), stream_pool=_cpu_pool())

    result = pipe.progress(iter([None]))
    assert result == 10


def test_pipeline_step_helper() -> None:
    src = Task.from_fn("src", _src, writes=("x",))
    add = Task.from_fn("add", _add, reads=("x",), writes=("step_result",))
    pipe = Pipeline(tasks=(src, add), stream_pool=_cpu_pool())

    assert pipe.step(None) == 10


def test_pipeline_context_manager_shutdown() -> None:
    """`with Pipeline(...) as p:` shuts down resources on exit."""
    src = Task.from_fn("src", _src, writes=("x",))
    add = Task.from_fn("add", _add, reads=("x",), writes=("step_result",))

    with Pipeline(tasks=(src, add), stream_pool=_cpu_pool()) as pipe:
        assert pipe.step(None) == 10
    # Shutdown is idempotent / no-op-friendly; second call shouldn't raise.
    pipe.shutdown()


# ---------------------------------------------------------------------
# Equivalence with imperative SchedulablePipeline
# ---------------------------------------------------------------------


def test_pipeline_equivalent_to_explicit_schedule() -> None:
    """Pipeline(tasks=...) is semantically identical to the explicit
    Schedule + SchedulablePipeline form for the same task list.

    Tasks are rebuilt for each engine instance so they don't share
    init() state across the two pipelines.
    """

    def _build_imperative_tasks():
        return (
            Task.from_fn("src", _src, writes=(DataSlot("x", 0),)),
            Task.from_fn(
                "add",
                _add,
                reads=(DataSlot("x", 0),),
                writes=(DataSlot("step_result", 0),),
            ),
        )

    def _build_declarative_tasks():
        return (
            Task.from_fn("src", _src, writes=("x",)),
            Task.from_fn("add", _add, reads=("x",), writes=("step_result",)),
        )

    schedule = Schedule(
        stages=(Stage(tasks=_build_imperative_tasks()),),
        stream_slots=("default",),
    )
    imperative = SchedulablePipeline(schedule, _cpu_pool())
    declarative = Pipeline(tasks=_build_declarative_tasks(), stream_pool=_cpu_pool())

    assert imperative.progress(iter([None])) == declarative.progress(iter([None]))
