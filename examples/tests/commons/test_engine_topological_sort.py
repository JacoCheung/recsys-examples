# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct tests for ``deps.topological_sort``.

The engine sorts tasks topologically by within-progress DAG edges
before re-binning into a single Stage at ``SchedulablePipeline``
construction. This module exercises the sort directly:

  * Edge sources covered: ``reads/writes`` exact slot match,
    ``depends_on`` with same lookahead, ``same_progress_sync``
    (any lookahead).
  * Edge sources NOT covered (must NOT add a topo edge):
    ``cross_iter_depends_on``, ``depends_on`` with producer.la >
    consumer.la, ``reads/writes`` with mismatched offsets.
  * Cycle detection raises ``ValueError`` with "Cyclic" in the
    message (the validator wraps this into a ``[rule 7]`` error).
  * Tie-break among DAG-independent tasks falls back to declaration
    order so author intent is preserved when the DAG is silent.
"""

import pytest
from commons.pipeline.engine import (
    DataSlot,
    SameProgressSyncSide,
    Schedule,
    Stage,
    Task,
)
from commons.pipeline.engine.deps import topological_sort


def _noop(ctx):
    return None


def _schedule(*tasks: Task, stream_slots=("default",)) -> Schedule:
    extra_streams = tuple({t.stream for t in tasks} - set(stream_slots))
    return Schedule(
        stages=(Stage(tasks=tasks),),
        stream_slots=tuple(sorted(set(stream_slots) | set(extra_streams))),
    )


def _names(tasks):
    return [t.name for t in tasks]


# ---------------------------------------------------------------------
# Trivial / declaration-order tie-break
# ---------------------------------------------------------------------


def test_topo_two_task_depends_on_chain() -> None:
    """Producer first, consumer second with depends_on — order matches
    the DAG (which agrees with declaration order here)."""
    a = Task.from_fn(name="a", fn=_noop, stream="default")
    b = Task.from_fn(name="b", fn=_noop, depends_on=("a",), stream="default")
    sorted_tasks = topological_sort(_schedule(a, b))
    assert _names(sorted_tasks) == ["a", "b"]


def test_topo_reorders_when_consumer_declared_before_producer() -> None:
    """Declaration order is [consumer, producer] but consumer
    depends_on producer — topo sort surfaces producer first."""
    consumer = Task.from_fn(
        name="cons", fn=_noop, depends_on=("prod",), stream="default"
    )
    producer = Task.from_fn(name="prod", fn=_noop, stream="default")
    sorted_tasks = topological_sort(_schedule(consumer, producer))
    assert _names(sorted_tasks) == ["prod", "cons"]


def test_topo_tie_break_by_declaration_order() -> None:
    """Three independent tasks (no DAG edges) → topo returns them in
    declaration order (the documented tie-breaker)."""
    a = Task.from_fn(name="a", fn=_noop, stream="default")
    b = Task.from_fn(name="b", fn=_noop, stream="default")
    c = Task.from_fn(name="c", fn=_noop, stream="default")
    sorted_tasks = topological_sort(_schedule(a, b, c))
    assert _names(sorted_tasks) == ["a", "b", "c"]


# ---------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------


def test_topo_cycle_via_depends_on_raises() -> None:
    """A.depends_on=(B,) + B.depends_on=(A,) is a cycle; raise
    ValueError with "Cyclic" in the message."""
    a = Task.from_fn(name="a", fn=_noop, depends_on=("b",), stream="default")
    b = Task.from_fn(name="b", fn=_noop, depends_on=("a",), stream="default")
    with pytest.raises(ValueError, match="Cyclic"):
        topological_sort(_schedule(a, b))


def test_topo_cycle_via_same_progress_sync_raises() -> None:
    """Cycle via ``same_progress_sync`` (which also adds topo edges)
    raises with "Cyclic"."""
    a = Task.from_fn(name="a", fn=_noop, same_progress_sync=("b",), stream="default")
    b = Task.from_fn(name="b", fn=_noop, same_progress_sync=("a",), stream="default")
    with pytest.raises(ValueError, match="Cyclic"):
        topological_sort(_schedule(a, b))


def test_topo_uses_cpu_side_same_progress_sync_only() -> None:
    consumer = Task.from_fn(
        name="consumer",
        fn=_noop,
        same_progress_sync=("producer",),
        same_progress_sync_sides=SameProgressSyncSide.CPU,
    )
    producer = Task.from_fn(name="producer", fn=_noop)

    assert _names(topological_sort(_schedule(consumer, producer))) == [
        "producer",
        "consumer",
    ]


def test_topo_ignores_gpu_only_same_progress_sync() -> None:
    consumer = Task.from_fn(
        name="consumer",
        fn=_noop,
        same_progress_sync=("producer",),
        same_progress_sync_sides=SameProgressSyncSide.GPU,
    )
    producer = Task.from_fn(name="producer", fn=_noop)

    assert _names(topological_sort(_schedule(consumer, producer))) == [
        "consumer",
        "producer",
    ]


def test_topo_self_loop_via_depends_on() -> None:
    """A task that depends_on itself: deps.py's
    ``_build_same_progress_dag_edges`` skips ``producer.name ==
    task.name`` so the self-edge is ignored — topo runs successfully
    with the single task in its declaration position. Verifies no
    infinite loop and no crash."""
    a = Task.from_fn(name="a", fn=_noop, depends_on=("a",), stream="default")
    sorted_tasks = topological_sort(_schedule(a))
    assert _names(sorted_tasks) == ["a"]


# ---------------------------------------------------------------------
# Edges that must NOT participate in the within-progress DAG
# ---------------------------------------------------------------------


def test_topo_cross_lookahead_depends_on_no_edge() -> None:
    """``depends_on`` with producer.la > consumer.la → producer's
    batch-K work happened in an EARLIER progress, no within-progress
    edge. Verify by declaring [consumer_with_dep, producer] and
    asserting the topo result preserves DECLARATION order (cons, prod)
    — if the edge were added, prod would be reordered to come first."""
    consumer = Task.from_fn(
        name="cons",
        fn=_noop,
        lookahead=0,
        depends_on=("prod",),  # cross-lookahead — must be suppressed
        stream="default",
    )
    producer = Task.from_fn(
        name="prod",
        fn=_noop,
        lookahead=2,
        stream="default",
    )
    # Declare consumer first; with no within-progress edge, tie-break
    # = declaration order, so the result MUST be [cons, prod]. If the
    # cross-lookahead depends_on incorrectly added a topo edge, prod
    # would come first.
    sorted_tasks = topological_sort(_schedule(consumer, producer))
    assert _names(sorted_tasks) == ["cons", "prod"]


def test_topo_cross_iter_depends_on_no_edge() -> None:
    """``cross_iter_depends_on`` is a different-batch dep — does NOT
    participate in the within-progress DAG."""
    # cons (la=2) cross_iter_depends_on prod (la=2, -1). If an edge
    # were added it would still be acyclic in isolation, but adding a
    # slot edge that goes the OTHER direction would cycle. Use the
    # simplest acyclic-confirmation form: just verify topo succeeds
    # and the order is the declaration order (no edges → tie-break).
    a = Task.from_fn(
        name="a",
        fn=_noop,
        lookahead=2,
        cross_iter_depends_on=(("b", -1),),
        stream="default",
    )
    b = Task.from_fn(name="b", fn=_noop, lookahead=2, stream="default")
    sorted_tasks = topological_sort(_schedule(a, b))
    assert _names(sorted_tasks) == ["a", "b"]


def test_topo_reads_writes_exact_slot_match_adds_edge() -> None:
    """Writer of (X, k) → reader of (X, k) IS a topo edge (same
    offset = same batch in current progress)."""
    reader = Task.from_fn(
        name="r", fn=_noop, reads=(DataSlot("X", 0),), stream="default"
    )
    writer = Task.from_fn(
        name="w", fn=_noop, writes=(DataSlot("X", 0),), stream="default"
    )
    # Declare reader first to force the sort to do real work.
    sorted_tasks = topological_sort(_schedule(reader, writer))
    assert _names(sorted_tasks) == ["w", "r"]


def test_topo_reads_writes_cross_iter_offset_no_edge() -> None:
    """Writer of (X, 1) and reader of (X, 0) is a CROSS-ITER edge
    (different offsets). It does NOT participate in the within-
    progress DAG — declaration order is preserved as tie-break."""
    reader = Task.from_fn(
        name="r",
        fn=_noop,
        lookahead=0,
        reads=(DataSlot("X", 0),),
        stream="default",
    )
    writer = Task.from_fn(
        name="w",
        fn=_noop,
        lookahead=1,
        writes=(DataSlot("X", 1),),
        stream="default",
    )
    # Declare reader first. With no within-progress edge, topo
    # preserves declaration order.
    sorted_tasks = topological_sort(_schedule(reader, writer))
    assert _names(sorted_tasks) == ["r", "w"]
