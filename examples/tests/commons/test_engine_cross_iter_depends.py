# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``deps.infer_cross_stream_event_deps`` handling of
SPEC_p4 v2 cross-iter pure-control ``depends_on`` (the ``("X", -N)``
escape hatch).

The function returns ``Dict[consumer_name, Tuple[(producer_task,
producer_stream, slot_offset), ...]]``. For cross-iter dependencies
the ``slot_offset`` is ``producer.batch_offset + neg_offset`` (with
``neg_offset < 0``), reflecting where the producer's event sits in
the ring after ``|neg_offset|`` advances.
"""

import pytest
from commons.pipeline.engine import DataSlot, Schedule, Stage, Task
from commons.pipeline.engine.deps import infer_cross_stream_event_deps


def _schedule(*tasks: Task, stream_slots=("default",)) -> Schedule:
    extra_streams = tuple({t.stream for t in tasks} - set(stream_slots))
    return Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=tuple(sorted(set(stream_slots) | set(extra_streams))),
    )


def test_cross_iter_dep_cross_stream_emits_triple() -> None:
    """Producer at lookahead=2 on stream A; consumer at lookahead=0
    on stream B with cross_iter_depends_on=(("producer", -1)).
    Expected slot_offset = 2 + (-1) = 1."""
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=2, stream="memcpy")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        depends_on=(("prod", -1),),
    )
    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))

    assert "cons" in deps
    assert ("prod", "memcpy", 1) in deps["cons"]


def test_cross_iter_dep_same_stream_no_triple() -> None:
    """Same-stream cross-iter ordering is implicit via stream FIFO —
    no explicit triple needed."""
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=2, stream="default")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        depends_on=(("prod", -1),),
    )
    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))
    assert "cons" not in deps


def test_cross_iter_dep_unknown_producer_silently_dropped() -> None:
    """Reference to a producer not in the schedule is dropped (other
    edge inference paths follow the same convention)."""
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        depends_on=(("ghost", -1),),
    )
    deps = infer_cross_stream_event_deps(_schedule(consumer))
    assert "cons" not in deps


def test_cross_iter_dep_event_rotated_out_raises() -> None:
    """If the producer's ring offset after rotation would be negative
    (event already evicted), construction-time error fires."""
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=0, stream="memcpy")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        depends_on=(("prod", -1),),  # 0 + (-1) = -1, out of ring
    )

    with pytest.raises(ValueError, match="rotated out of the ring"):
        infer_cross_stream_event_deps(_schedule(producer, consumer))


def test_cross_iter_dep_combines_with_within_iter_data_edges() -> None:
    """Cross-iter pure-control + within-iter data edges co-exist on
    the same consumer."""
    h2d = Task.from_fn(
        "h2d",
        fn=lambda ctx: None,
        lookahead=2,
        stream="memcpy",
        writes=(DataSlot("batch_gpu", 2),),
    )
    optimizer = Task.from_fn(
        "optimizer", fn=lambda ctx: None, lookahead=2, stream="comm"
    )
    forward = Task.from_fn(
        "forward",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        reads=(DataSlot("batch_gpu", 0),),  # data edge: forward reads h2d's batch_gpu
        depends_on=(("optimizer", -1),),  # cross-iter pure-control: 2 + (-1) = 1
    )
    deps = infer_cross_stream_event_deps(_schedule(h2d, optimizer, forward))

    assert ("h2d", "memcpy", 0) in deps["forward"]  # within-iter data edge
    assert ("optimizer", "comm", 1) in deps["forward"]  # cross-iter ctrl edge


def test_cross_iter_dep_deeper_offset() -> None:
    """N=2 → slot_offset = producer.batch_offset - 2."""
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=3, stream="memcpy")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        depends_on=(("prod", -2),),
    )
    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))

    assert ("prod", "memcpy", 1) in deps["cons"]  # 3 + (-2) = 1


# ---------------------------------------------------------------------
# SPEC_p4 v2 §5: redundancy with reads/writes data edges
# ---------------------------------------------------------------------


def test_cross_iter_dep_redundant_with_data_edge_rejected() -> None:
    """If a data edge from reads/writes already implies the same
    cross-iter dependency, declaring it explicitly is redundant
    and gets rejected at construction time."""
    producer = Task.from_fn(
        "prod",
        fn=lambda ctx: None,
        lookahead=2,
        stream="memcpy",
        writes=(DataSlot("foo", 2),),
    )
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        reads=(DataSlot("foo", 0),),  # implies (prod, -2): writer 2 - reader 0
        depends_on=(("prod", -2),),  # restates that — redundant
    )

    with pytest.raises(ValueError, match="redundant cross-iter"):
        infer_cross_stream_event_deps(_schedule(producer, consumer))


def test_cross_iter_dep_different_n_not_redundant() -> None:
    """If reads/writes implies N=2 but cross-iter declares N=1, they
    are independent edges — no redundancy, no rejection."""
    producer = Task.from_fn(
        "prod",
        fn=lambda ctx: None,
        lookahead=2,
        stream="memcpy",
        writes=(DataSlot("foo", 2),),
    )
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        reads=(DataSlot("foo", 0),),  # implies (prod, -2)
        depends_on=(("prod", -1),),  # different N — independent edge
    )

    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))
    # Both edges present:
    assert ("prod", "memcpy", 0) in deps["cons"]  # data edge at reader offset
    assert ("prod", "memcpy", 1) in deps["cons"]  # explicit cross-iter (-1)


def test_cross_iter_dep_no_matching_data_edge_ok() -> None:
    """If consumer doesn't read any slot the producer writes, the
    cross-iter declaration is purely control-flow — keep it."""
    producer = Task.from_fn(
        "prod",
        fn=lambda ctx: None,
        lookahead=2,
        stream="memcpy",
        writes=(DataSlot("foo", 2),),
    )
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        # No reads from "foo" — the cross_iter declaration carries the
        # only ordering signal between these two tasks.
        depends_on=(("prod", -2),),
    )
    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))
    assert ("prod", "memcpy", 0) in deps["cons"]
