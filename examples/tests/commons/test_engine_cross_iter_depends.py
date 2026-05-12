# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for cross-iteration dependency event inference."""

import pytest
from commons.pipeline.engine import DataSlot, Schedule, Stage, Task
from commons.pipeline.engine.deps import infer_cross_stream_event_deps, topological_sort


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


def test_cross_iter_cross_stream_emits_event_triple_at_rotated_slot() -> None:
    producer = _task("prod", stream="memcpy", lookahead=2)
    consumer = _task(
        "cons",
        lookahead=1,
        cross_iter_depends_on=(("prod", -1),),
    )

    assert infer_cross_stream_event_deps(_schedule(producer, consumer)) == {
        "cons": (("prod", "memcpy", 0),)
    }


def test_cross_iter_same_stream_emits_no_event_triple() -> None:
    producer = _task("prod", lookahead=2)
    consumer = _task("cons", lookahead=1, cross_iter_depends_on=(("prod", -1),))

    assert infer_cross_stream_event_deps(_schedule(producer, consumer)) == {}


def test_cross_iter_rejects_rotated_out_and_future_reads() -> None:
    rotated_out = _schedule(
        _task("prod", stream="memcpy", lookahead=0),
        _task("cons", cross_iter_depends_on=(("prod", -1),)),
    )
    with pytest.raises(ValueError, match="rotated out of the ring"):
        infer_cross_stream_event_deps(rotated_out)

    future_read = _schedule(
        _task("prod", stream="memcpy", lookahead=0),
        _task("cons", lookahead=3, cross_iter_depends_on=(("prod", -1),)),
    )
    with pytest.raises(ValueError, match="future-read"):
        infer_cross_stream_event_deps(future_read)


def test_delta_zero_cross_iter_adds_topological_edge_and_event_triple() -> None:
    update = _task("update", stream="memcpy", lookahead=0)
    fwd = _task(
        "fwd",
        lookahead=1,
        cross_iter_depends_on=(("update", -1),),
    )
    schedule = _schedule(fwd, update)

    assert tuple(t.name for t in topological_sort(schedule)) == ("update", "fwd")
    assert infer_cross_stream_event_deps(schedule) == {
        "fwd": (("update", "memcpy", 0),)
    }


def test_redundant_cross_iter_data_edge_is_rejected() -> None:
    producer = _task(
        "h2d",
        stream="memcpy",
        lookahead=1,
        writes=(DataSlot("batch_gpu", 1),),
    )
    consumer = _task(
        "forward",
        lookahead=0,
        reads=(DataSlot("batch_gpu", 0),),
        cross_iter_depends_on=(("h2d", -1),),
    )

    with pytest.raises(ValueError, match="duplicates an implicit data edge"):
        infer_cross_stream_event_deps(_schedule(producer, consumer))


def test_same_progress_sync_uses_producer_slot_offset() -> None:
    producer = _task("prefetch", stream="prefetch", lookahead=1)
    consumer = _task("backward", same_progress_sync=("prefetch",))

    assert infer_cross_stream_event_deps(_schedule(producer, consumer)) == {
        "backward": (("prefetch", "prefetch", 1),)
    }
