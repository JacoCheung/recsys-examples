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
    """Producer on stream A; consumer at lookahead=1 on stream B with
    cross_iter_depends_on=(("producer", -1)). Slot offset =
    consumer.batch_offset + neg_offset = 1 + (-1) = 0."""
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=2, stream="memcpy")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=1,
        stream="default",
        cross_iter_depends_on=(("prod", -1),),
    )
    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))

    assert "cons" in deps
    assert ("prod", "memcpy", 0) in deps["cons"]


def test_cross_iter_dep_same_stream_no_triple() -> None:
    """Same-stream cross-iter ordering is implicit via stream FIFO —
    no explicit triple needed."""
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=2, stream="default")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        cross_iter_depends_on=(("prod", -1),),
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
        cross_iter_depends_on=(("ghost", -1),),
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
        cross_iter_depends_on=(("prod", -1),),  # 0 + (-1) = -1, out of ring
    )

    with pytest.raises(ValueError, match="rotated out of the ring"):
        infer_cross_stream_event_deps(_schedule(producer, consumer))


def test_cross_iter_dep_future_read_raises() -> None:
    """consumer.la > producer.la + |neg_offset| ⇒ Δ < 0 ⇒ producer
    has not yet processed batch K-N by consumer's progress. Must
    raise at construction time, not silently miss the dep at runtime.

    Scenario: consumer.la=3, producer.la=0, N=1. Δ = 0+1-3 = -2 (future).
    slot_offset = 3-1 = 2 ≥ 0, so the existing rotated-out check
    would NOT catch this — only the future-read check does.
    """
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=0, stream="memcpy")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=3,
        stream="default",
        cross_iter_depends_on=(("prod", -1),),
    )

    with pytest.raises(ValueError, match="future-read"):
        infer_cross_stream_event_deps(_schedule(producer, consumer))


def test_cross_iter_dep_delta_zero_same_stream_accepted_via_topo_edge() -> None:
    """Δ=0 cross_iter on the same stream is auto-promoted to the
    same_progress_sync mechanical contract: a topo-DAG edge from
    producer to consumer is added in _build_same_progress_dag_edges
    (no event triple needed because same-stream FIFO orders the
    submissions, given the topo edge ensures producer goes first).

    User's exact 1:1 example: fwd.la=1, update.la=0, both on stream
    "default", with fwd.cross_iter_depends_on=(("update", -1),) →
    Δ=0. Per SPEC_cross_iter_delta0_autoconvert.md.
    """
    from commons.pipeline.engine.deps import _build_same_progress_dag_edges

    producer = Task.from_fn(
        "update", fn=lambda ctx: None, lookahead=0, stream="default"
    )
    consumer = Task.from_fn(
        "fwd",
        fn=lambda ctx: None,
        lookahead=1,  # = producer.la(0) + N(1) → Δ=0
        stream="default",  # same stream
        cross_iter_depends_on=(("update", -1),),
    )

    # No raise — auto-promoted.
    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))
    # Same-stream: no event triple emitted (FIFO + topo edge handles it).
    assert "fwd" not in deps or deps["fwd"] == ()

    # The topo edge IS added (this is the key behavior change).
    edges = _build_same_progress_dag_edges((producer, consumer))
    assert "update" in edges["fwd"], (
        "Δ=0 cross_iter must add a topo edge update→fwd "
        "(auto-promoted to same_progress_sync semantics)"
    )


def test_cross_iter_dep_delta_zero_cross_stream_accepted_emits_at_producer_la() -> None:
    """Δ=0 cross_iter across streams is auto-promoted to the
    same_progress_sync mechanical contract: emits a wait_event triple
    at slot[producer.la] (same slot where producer recorded its event
    in this progress), AND adds a topo-DAG edge.

    Compare with the Δ=0 same_progress_sync test below: both must
    emit identical triple shape and topo edge.
    """
    from commons.pipeline.engine.deps import _build_same_progress_dag_edges

    producer = Task.from_fn("update", fn=lambda ctx: None, lookahead=0, stream="memcpy")
    consumer = Task.from_fn(
        "fwd",
        fn=lambda ctx: None,
        lookahead=1,  # = producer.la(0) + N(1) → Δ=0
        stream="default",
        cross_iter_depends_on=(("update", -1),),
    )

    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))
    # Auto-promoted: emit at slot[producer.la] = slot[0]
    assert deps["fwd"] == (("update", "memcpy", 0),), (
        f"Δ=0 cross_iter cross-stream must emit (producer, producer.stream, "
        f"producer.la) — same as same_progress_sync. Got: {deps.get('fwd')}"
    )

    # Topo edge added.
    edges = _build_same_progress_dag_edges((producer, consumer))
    assert "update" in edges["fwd"]


def test_cross_iter_dep_future_read_boundary_la_diff_one_too_far() -> None:
    """One step past the boundary: consumer.la = producer.la + N + 1
    ⇒ Δ = -1. Must raise.
    """
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=1, stream="memcpy")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=3,  # = producer.la(1) + N(1) + 1
        stream="default",
        cross_iter_depends_on=(("prod", -1),),
    )

    with pytest.raises(ValueError, match="future-read"):
        infer_cross_stream_event_deps(_schedule(producer, consumer))


def test_cross_iter_delta_zero_user_1to1_example_topo_orders_correctly() -> None:
    """User-provided 1:1 setup, post-auto-convert (per
    SPEC_cross_iter_delta0_autoconvert.md).

    Topology:
      fwd.la = 1, stream="default"
      bwd.la = 1, stream="default", depends_on=("fwd",)        # same la
      update.la = 0, stream="default", depends_on=("bwd",)     # cross-la (ring-rot)
      fwd.cross_iter_depends_on = (("update", -1),)            # Δ = 0+1-1 = 0
      All three tasks colocated on a single thread.

    With the Δ=0 auto-convert, cross_iter Δ=0 contributes a topo edge
    update → fwd (same as same_progress_sync would). topo edges:
      fwd → bwd                              (depends_on, same la=1)
      update → fwd                           (cross_iter Δ=0 auto-promoted)
      bwd → update : ABSENT                  (cross-la depends_on, ring)

    Topological_sort therefore returns: (update, fwd, bwd) — update
    fires first as the user intended.

    Pre-spec behavior was the inverse: cross_iter contributed no
    edge, declaration order put fwd first, fwd's wait_event lookup
    missed → silent wrong sync. The pre-spec test name was
    ``test_cross_iter_delta_zero_user_1to1_example_topo_misorders``
    asserting the misorder; this test now asserts the corrected order.
    """
    from commons.pipeline.engine.deps import (
        _build_same_progress_dag_edges,
        topological_sort,
    )

    fwd = Task.from_fn(
        "fwd",
        fn=lambda ctx: None,
        lookahead=1,
        stream="default",
        cross_iter_depends_on=(("update", -1),),  # Δ=0 — auto-promoted
    )
    bwd = Task.from_fn(
        "bwd",
        fn=lambda ctx: None,
        lookahead=1,
        stream="default",
        depends_on=("fwd",),
    )
    update = Task.from_fn(
        "update",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        depends_on=("bwd",),
    )

    schedule = _schedule(fwd, bwd, update)
    sorted_names = tuple(t.name for t in topological_sort(schedule))
    assert sorted_names == ("update", "fwd", "bwd"), (
        f"Got {sorted_names}. Expected (update, fwd, bwd) — Δ=0 cross_iter "
        f"auto-promote must add the update→fwd topo edge so update fires "
        f"first; declaration order is then only a tie-break."
    )

    # Confirm the topo edge was actually added to the DAG.
    edges = _build_same_progress_dag_edges((fwd, bwd, update))
    assert "update" in edges["fwd"], (
        "cross_iter_depends_on with Δ=0 must contribute a topo edge "
        "update → fwd (auto-promoted to same_progress_sync semantics)."
    )


def test_cross_iter_topo_edge_only_when_delta_zero() -> None:
    """Regression for the topo-DAG edge inclusion rule per
    SPEC_cross_iter_delta0_autoconvert.md:

      * Δ ≥ 1 (genuine cross-progress): cross_iter adds NO topo edge.
        Producer ran in a strictly earlier progress; ring rotation
        places its event in the consumer's slot[consumer.la−N]; no
        in-progress ordering needed. Tie-break falls back to declaration.
      * Δ = 0 (same-progress, different batches): cross_iter ADDS a
        topo edge producer → consumer (auto-promoted to the
        same_progress_sync mechanical contract).
    """
    from commons.pipeline.engine.deps import (
        _build_same_progress_dag_edges,
        topological_sort,
    )

    # ----- Δ=2 (≥1) case: NO topo edge -----
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=1, stream="memcpy")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        cross_iter_depends_on=(("prod", -1),),  # Δ = 1+1-0 = 2
    )
    edges = _build_same_progress_dag_edges((producer, consumer))
    assert "prod" not in edges["cons"], (
        "Δ ≥ 1 cross_iter must NOT contribute a topo edge "
        "(handled by ring rotation across progresses)."
    )
    # With consumer declared SECOND, no topo edge means declaration
    # order is the tie-break → (producer, consumer) order:
    sorted_names = tuple(
        t.name for t in topological_sort(_schedule(producer, consumer))
    )
    assert sorted_names == ("prod", "cons")
    # Reverse declaration: still no topo edge, so consumer-first leaks:
    sorted_names_rev = tuple(
        t.name for t in topological_sort(_schedule(consumer, producer))
    )
    assert sorted_names_rev == ("cons", "prod"), (
        "Δ ≥ 1: declaration order must drive topo tie-break, " f"got {sorted_names_rev}"
    )

    # ----- Δ=0 case: topo edge PRESENT -----
    update_task = Task.from_fn(
        "update", fn=lambda ctx: None, lookahead=0, stream="memcpy"
    )
    fwd = Task.from_fn(
        "fwd",
        fn=lambda ctx: None,
        lookahead=1,
        stream="default",
        cross_iter_depends_on=(("update", -1),),  # Δ = 0+1-1 = 0
    )
    edges_d0 = _build_same_progress_dag_edges((update_task, fwd))
    assert "update" in edges_d0["fwd"], (
        "Δ = 0 cross_iter MUST contribute a topo edge update → fwd "
        "(auto-promoted to same_progress_sync semantics per SPEC)."
    )
    # Now reversing declaration order does NOT misorder, because the
    # topo edge forces update first regardless:
    sorted_d0_rev = tuple(t.name for t in topological_sort(_schedule(fwd, update_task)))
    assert sorted_d0_rev == ("update", "fwd"), (
        "Δ = 0: topo edge must override declaration order; " f"got {sorted_d0_rev}"
    )


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
        lookahead=1,
        stream="default",
        reads=(DataSlot("batch_gpu", 1),),  # name match, ring-rotated read
        cross_iter_depends_on=(("optimizer", -1),),  # slot = 1 + (-1) = 0
    )
    deps = infer_cross_stream_event_deps(_schedule(h2d, optimizer, forward))

    assert ("h2d", "memcpy", 1) in deps["forward"]  # data edge at reader offset
    assert ("optimizer", "comm", 0) in deps["forward"]  # cross-iter ctrl edge


def test_cross_iter_dep_deeper_offset() -> None:
    """N=2 → slot_offset = consumer.batch_offset - 2."""
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=3, stream="memcpy")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=3,  # consumer.la must be >= |neg| (=2)
        stream="default",
        cross_iter_depends_on=(("prod", -2),),
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
        lookahead=4,
        stream="memcpy",
        writes=(DataSlot("foo", 4),),
    )
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=2,
        stream="default",
        reads=(DataSlot("foo", 2),),  # implies (prod, -2): writer 4 - reader 2
        cross_iter_depends_on=(("prod", -2),),  # restates that — redundant
    )

    with pytest.raises(ValueError, match="redundant cross-iter"):
        infer_cross_stream_event_deps(_schedule(producer, consumer))


def test_cross_iter_dep_different_n_not_redundant() -> None:
    """If reads/writes implies one ring-rotation distance and cross-
    iter declares a different N, they are independent edges — no
    redundancy, no rejection."""
    producer = Task.from_fn(
        "prod",
        fn=lambda ctx: None,
        lookahead=4,
        stream="memcpy",
        writes=(DataSlot("foo", 4),),
    )
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=2,
        stream="default",
        reads=(DataSlot("foo", 2),),  # implies (prod, -2): writer 4 - reader 2
        cross_iter_depends_on=(("prod", -1),),  # different N (=1) — not redundant
    )

    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))
    # Both edges present:
    assert ("prod", "memcpy", 2) in deps["cons"]  # data edge at reader offset
    assert ("prod", "memcpy", 1) in deps["cons"]  # explicit cross-iter (-1) → 2-1


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
        lookahead=2,  # >= |neg|=2
        stream="default",
        # No reads from "foo" — the cross_iter declaration carries the
        # only ordering signal between these two tasks.
        cross_iter_depends_on=(("prod", -2),),
    )
    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))
    assert ("prod", "memcpy", 0) in deps["cons"]


# ---------------------------------------------------------------------
# Gap A — same_progress_sync slot-offset behavior
# ---------------------------------------------------------------------
#
# ``same_progress_sync`` is a same-progress GPU coherency wait. Engine
# emits a wait_event triple at slot offset = PRODUCER's batch_offset
# (the producer's own slot, where it just recorded its event in this
# same progress() call) — NOT consumer.batch_offset (which would be a
# ring-rotated stale event). This is the key behavioral difference vs
# ``depends_on``, which keys at consumer.batch_offset.


def test_same_progress_sync_same_stream_no_triple() -> None:
    """Same-stream coherency is a no-op (CUDA stream FIFO already
    serializes), so no triple is emitted."""
    producer = Task.from_fn("prod", fn=lambda ctx: None, lookahead=2, stream="default")
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        same_progress_sync=("prod",),
    )
    deps = infer_cross_stream_event_deps(_schedule(producer, consumer))
    assert "cons" not in deps


def test_same_progress_sync_unknown_producer_silently_dropped() -> None:
    """Unknown producer name: deps.py is permissive and silently
    drops the edge (the schedule validator catches typos at rule 6;
    deps.py is downstream of validation)."""
    consumer = Task.from_fn(
        "cons",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        same_progress_sync=("ghost",),
    )
    # No producer named "ghost" in the schedule. deps.py must NOT
    # raise — that's the validator's job.
    deps = infer_cross_stream_event_deps(_schedule(consumer))
    assert "cons" not in deps


def test_same_progress_sync_vs_depends_on_offset_difference() -> None:
    """Critical contrast: same producer/consumer la pair (2 → 0):

      * ``same_progress_sync=("prod",)`` keys the slot at PRODUCER's
        lookahead (=2) — current-progress event, no ring rotation.
      * ``depends_on=("prod",)`` keys the slot at CONSUMER's lookahead
        (=0) — ring-rotated.

    Both arms together pin down the offset divergence and the negative
    (the OTHER form's slot must NOT appear).
    """
    # Arm 1: same_progress_sync — slot at producer.la=2.
    prod_a = Task.from_fn("prod_a", fn=lambda ctx: None, lookahead=2, stream="memcpy")
    cons_sps = Task.from_fn(
        "cons_sps",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        same_progress_sync=("prod_a",),
    )
    deps_sps = infer_cross_stream_event_deps(_schedule(prod_a, cons_sps))
    assert ("prod_a", "memcpy", 2) in deps_sps["cons_sps"]
    # Sanity: consumer.la (=0) must NOT be the slot offset for sps.
    assert ("prod_a", "memcpy", 0) not in deps_sps["cons_sps"]

    # Arm 2: depends_on — slot at consumer.la=0.
    prod_b = Task.from_fn("prod_b", fn=lambda ctx: None, lookahead=2, stream="memcpy")
    cons_dep = Task.from_fn(
        "cons_dep",
        fn=lambda ctx: None,
        lookahead=0,
        stream="default",
        depends_on=("prod_b",),
    )
    deps_dep = infer_cross_stream_event_deps(_schedule(prod_b, cons_dep))
    assert ("prod_b", "memcpy", 0) in deps_dep["cons_dep"]
    # And confirm the same triple is NOT what same_progress_sync emits.
    assert ("prod_b", "memcpy", 2) not in deps_dep["cons_dep"]
