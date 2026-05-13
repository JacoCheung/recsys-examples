# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for cross-stream wait inference."""

import pytest
from commons.pipeline.engine import (
    DataSlot,
    SameProgressSyncSide,
    Schedule,
    Stage,
    Task,
)
from commons.pipeline.engine.deps import infer_cross_stream_waits


def _noop(ctx):
    return None


def _task(name: str, *, stream: str = "default", lookahead: int = 0, **kwargs) -> Task:
    return Task.from_fn(
        name=name,
        fn=_noop,
        stream=stream,
        lookahead=lookahead,
        **kwargs,
    )


def _schedule(*tasks: Task) -> Schedule:
    streams = tuple(sorted({"default"} | {t.stream for t in tasks}))
    return Schedule(stages=(Stage(tasks=tasks),), stream_slots=streams)


def test_same_stream_edges_do_not_emit_waits() -> None:
    writer = _task("writer", writes=("x",))
    reader = _task("reader", reads=("x",))
    assert infer_cross_stream_waits(_schedule(writer, reader)) == {}


def test_cross_stream_slot_and_depends_on_edges_emit_waits() -> None:
    h2d = _task("h2d", stream="memcpy", writes=("batch_gpu",))
    allreduce = _task("allreduce", stream="comm")
    forward = _task(
        "forward",
        reads=("batch_gpu",),
        depends_on=("allreduce",),
    )

    assert infer_cross_stream_waits(_schedule(h2d, allreduce, forward)) == {
        "forward": ("comm", "memcpy")
    }


def test_same_progress_sync_gpu_side_controls_stream_waits() -> None:
    producer = _task("producer", stream="comm")
    cpu_only = _task(
        "cpu_only",
        same_progress_sync=(("producer", SameProgressSyncSide.CPU),),
    )
    gpu_only = _task(
        "gpu_only",
        same_progress_sync=(("producer", SameProgressSyncSide.GPU),),
    )

    assert infer_cross_stream_waits(_schedule(producer, cpu_only)) == {}
    assert infer_cross_stream_waits(_schedule(producer, gpu_only)) == {
        "gpu_only": ("comm",)
    }


def test_cross_iter_prefetch_edge_uses_slot_name() -> None:
    h2d = _task(
        "h2d",
        stream="memcpy",
        lookahead=1,
        writes=(DataSlot("batch_gpu", 1),),
    )
    forward = _task(
        "forward",
        lookahead=0,
        reads=(DataSlot("batch_gpu", 0),),
    )

    assert infer_cross_stream_waits(_schedule(h2d, forward)) == {"forward": ("memcpy",)}


def test_multiple_producer_streams_are_sorted() -> None:
    a = _task("a", stream="comm", writes=("x",))
    b = _task("b", stream="memcpy", writes=("y",))
    c = _task("c", reads=("x", "y"))

    assert infer_cross_stream_waits(_schedule(a, b, c)) == {"c": ("comm", "memcpy")}


def test_duplicate_exact_writer_rejected() -> None:
    a = _task("a", writes=("x",))
    b = _task("b", stream="memcpy", writes=("x",))

    with pytest.raises(ValueError, match="multiple writers"):
        infer_cross_stream_waits(_schedule(a, b))


def test_same_slot_name_written_on_multiple_streams_rejected() -> None:
    a = _task("a", stream="memcpy", lookahead=1, writes=(DataSlot("x", 1),))
    b = _task("b", stream="comm", lookahead=0, writes=(DataSlot("x", 0),))

    with pytest.raises(ValueError, match="multiple streams"):
        infer_cross_stream_waits(_schedule(a, b))


def test_unresolved_read_is_left_to_validator() -> None:
    reader = _task("reader", reads=("missing",))
    assert infer_cross_stream_waits(_schedule(reader)) == {}
