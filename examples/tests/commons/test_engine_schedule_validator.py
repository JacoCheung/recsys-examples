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

"""Schedule validator tests."""

import pytest
from commons.pipeline.engine import (
    DataSlot,
    Schedule,
    ScheduleValidationError,
    Stage,
    StreamPool,
    Task,
)
from commons.pipeline.engine.autosched import validate


def _noop(ctx):
    return None


def _pool(*names: str) -> StreamPool:
    """Build a minimal CPU StreamPool covering the named slots."""
    return StreamPool({n: None for n in names})


# ----------------------------------------------------------------------
# Task invariants
# ----------------------------------------------------------------------


def test_task_duplicate_task_name_rejected() -> None:
    a = Task.from_fn(name="dup", fn=_noop, stream="default")
    b = Task.from_fn(name="dup", fn=_noop, stream="default")
    schedule = Schedule(stages=(Stage(tasks=(a, b)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[task\].*Duplicate"):
        validate(schedule, _pool("default"))


def test_task_slot_offset_mismatch_read_rejected() -> None:
    """Task at lookahead=0 declaring a read of DataSlot with
    lookahead=1 is a metadata lie — `ctx.slots` at runtime targets
    `ring.at(task.lookahead=0)`, not offset=1."""
    t = Task.from_fn(
        name="bad_read",
        fn=_noop,
        reads=(DataSlot("x", batch_offset=1),),  # mismatch
        stream="default",
        lookahead=0,
    )
    # Add a phantom writer so the read resolves; this keeps the test
    # focused on the task-level slot-offset check.
    w = Task.from_fn(
        name="w",
        fn=_noop,
        writes=(DataSlot("x", batch_offset=1),),
        stream="default",
        lookahead=1,
    )
    schedule = Schedule(stages=(Stage(tasks=(w, t)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[task\].*batch_offset must equal"
    ):
        validate(schedule, _pool("default"))


def test_task_reserved_slot_name_caught_regardless_of_offset() -> None:
    """The reserved-name guard must fire BEFORE the offset-consistency
    check — otherwise a user could dodge the guard by tweaking the
    slot's batch_offset. Here the slot's offset mismatches the task
    (task=0, slot=1) AND the name is reserved. Reserved-name error
    must surface, not the offset-mismatch error."""
    t = Task.from_fn(
        name="bad",
        fn=_noop,
        writes=(DataSlot("batch_cpu", batch_offset=1),),  # mismatched offset
        stream="default",
        lookahead=0,
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[task\].*reserved for engine-populated"
    ):
        validate(schedule, _pool("default"))


def test_task_slot_offset_mismatch_write_rejected() -> None:
    """Same invariant on the write side."""
    t = Task.from_fn(
        name="bad_write",
        fn=_noop,
        writes=(DataSlot("y", batch_offset=2),),  # mismatch
        stream="default",
        lookahead=0,
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[task\].*batch_offset must equal"
    ):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Resource invariants
# ----------------------------------------------------------------------


def test_resource_stream_not_in_schedule_slots() -> None:
    t = Task.from_fn(name="bad", fn=_noop, stream="comm")
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[resource\].*not declared"):
        validate(schedule, _pool("default", "comm"))


def test_resource_stream_not_in_pool() -> None:
    """Stream declared in `stream_slots` but missing from the pool."""
    t = Task.from_fn(name="bad", fn=_noop, stream="memcpy")
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default", "memcpy"))
    # Pool lacks 'memcpy'
    with pytest.raises(
        ScheduleValidationError, match=r"\[resource\].*not present in StreamPool"
    ):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Data-slot invariants
# ----------------------------------------------------------------------


def test_data_duplicate_exact_writer_rejected() -> None:
    a = Task.from_fn(name="a", fn=_noop, writes=(DataSlot("x"),), stream="default")
    b = Task.from_fn(name="b", fn=_noop, writes=(DataSlot("x"),), stream="default")
    schedule = Schedule(stages=(Stage(tasks=(a, b)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[data\].*multiple writers"):
        validate(schedule, _pool("default"))


def test_data_same_name_writers_on_different_streams() -> None:
    """The same logical slot name must not be written on two streams."""
    a = Task.from_fn(
        name="a",
        fn=_noop,
        writes=(DataSlot("x", batch_offset=1),),
        stream="memcpy",
        lookahead=1,
    )
    b = Task.from_fn(
        name="b",
        fn=_noop,
        writes=(DataSlot("x", batch_offset=0),),
        stream="comm",
        lookahead=0,
    )
    schedule = Schedule(
        stages=(Stage(tasks=(a, b)),),
        stream_slots=("default", "memcpy", "comm"),
    )
    with pytest.raises(ScheduleValidationError, match=r"\[data\].*multiple streams"):
        validate(schedule, _pool("default", "memcpy", "comm"))


def test_data_unresolved_slot_read_rejected() -> None:
    t = Task.from_fn(
        name="reader",
        fn=_noop,
        reads=(DataSlot("nonexistent"),),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[data\].*no task writes"):
        validate(schedule, _pool("default"))


def test_data_cross_stage_same_offset_reader_before_writer_rejected() -> None:
    """Same-offset readers need the writer earlier in stage order."""
    consumer = Task.from_fn(
        name="early_reader",
        fn=_noop,
        reads=(DataSlot("x"),),
        stream="default",
    )
    producer = Task.from_fn(
        name="late_writer",
        fn=_noop,
        writes=(DataSlot("x"),),
        stream="default",
    )
    schedule = Schedule(
        stages=(
            Stage(tasks=(consumer,)),  # reads first
            Stage(tasks=(producer,)),  # writes later — too late for same iter
        ),
        stream_slots=("default",),
    )
    with pytest.raises(
        ScheduleValidationError,
        match=r"\[data\].*Intra-iter reads require the writer",
    ):
        validate(schedule, _pool("default"))


def test_data_cross_iter_reader_before_writer_allowed() -> None:
    """Cross-iter prefetch is legal regardless of declaration order:
    writer at lookahead=1 writes "future" batch. Reader at
    lookahead=0 reads the ring's current slot, which (after
    advance) is what the writer wrote on the previous iter. The
    reader can be declared before the writer — no violation."""
    consumer = Task.from_fn(
        name="current_reader",
        fn=_noop,
        reads=(DataSlot("x", batch_offset=0),),
        stream="default",
        lookahead=0,
    )
    producer = Task.from_fn(
        name="next_batch_writer",
        fn=_noop,
        writes=(DataSlot("x", batch_offset=1),),
        stream="default",
        lookahead=1,
    )
    # Declare consumer first (at position 0), producer later.
    schedule = Schedule(
        stages=(Stage(tasks=(consumer, producer)),),
        stream_slots=("default",),
    )
    # Must not raise — cross-iter reads (writer_offset > reader_offset)
    # have no declaration-order constraint.
    validate(schedule, _pool("default"))


def test_data_future_read_rejected() -> None:
    """Writer at offset=0, reader at offset=1. Reader wants a FUTURE
    batch that no ring-advance path can surface."""
    writer = Task.from_fn(name="w", fn=_noop, writes=(DataSlot("x"),), stream="default")
    reader = Task.from_fn(
        name="r",
        fn=_noop,
        reads=(DataSlot("x", batch_offset=1),),
        stream="default",
        lookahead=1,
    )
    schedule = Schedule(
        stages=(Stage(tasks=(writer, reader)),),
        stream_slots=("default",),
    )
    with pytest.raises(ScheduleValidationError, match=r"\[data\].*lower offset"):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Dependency invariants
# ----------------------------------------------------------------------


def test_dependency_unresolved_depends_on_rejected() -> None:
    t = Task.from_fn(
        name="t",
        fn=_noop,
        depends_on=("no_such_task",),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[dependency\].*not a task name"
    ):
        validate(schedule, _pool("default"))


def test_dependency_forward_depends_on_now_accepted() -> None:
    """``depends_on`` no longer requires strict declaration-order
    precedence. The engine sorts tasks topologically by within-progress
    DAG edges (see ``deps.topological_sort``) so a producer declared
    after its consumer is reordered to run first.

    Cycles are still caught by the dependency DAG check.
    """
    a = Task.from_fn(
        name="a",
        fn=_noop,
        depends_on=("b",),  # b declared after a — engine will reorder
        stream="default",
    )
    b = Task.from_fn(name="b", fn=_noop, stream="default")
    schedule = Schedule(stages=(Stage(tasks=(a, b)),), stream_slots=("default",))
    # Should not raise — declaration order no longer matters.
    validate(schedule, _pool("default"))


def test_dependency_unknown_cross_iter_dep_rejected() -> None:
    """Unknown producer name in ``cross_iter_depends_on`` raises."""
    a = Task.from_fn(
        name="a",
        fn=_noop,
        cross_iter_depends_on=(("ghost", -1),),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(a,)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError,
        match=r"\[dependency\].*cross_iter_depends_on.*ghost",
    ):
        validate(schedule, _pool("default"))


def test_dependency_unknown_same_progress_sync_rejected() -> None:
    """Unknown producer name in ``same_progress_sync`` raises (without
    this check, a typo would silently drop the coherency wait)."""
    a = Task.from_fn(
        name="a",
        fn=_noop,
        same_progress_sync=("typo",),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(a,)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError,
        match=r"\[dependency\].*same_progress_sync.*typo",
    ):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Smoke: valid schedules
# ----------------------------------------------------------------------


def test_smoke_valid_prefetch_schedule_passes() -> None:
    """Realistic prefetch schedule: h2d@offset=1 on memcpy, fwd/bwd
    on default. Must pass all validator invariants."""
    h2d = Task.from_fn(
        name="h2d",
        fn=_noop,
        reads=(DataSlot("batch_cpu", batch_offset=1),),
        writes=(DataSlot("batch_gpu", batch_offset=1),),
        stream="memcpy",
        lookahead=1,
    )
    fwd = Task.from_fn(
        name="fwd",
        fn=_noop,
        reads=(DataSlot("batch_gpu"),),
        writes=(DataSlot("step_result"),),
        stream="default",
    )
    bwd = Task.from_fn(
        name="bwd",
        fn=_noop,
        reads=(DataSlot("step_result"),),
        depends_on=("fwd",),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(h2d, fwd, bwd)),),
        stream_slots=("default", "memcpy"),
    )
    # `batch_cpu` is engine-populated and does not need a user writer.
    # Should not raise.
    validate(schedule, _pool("default", "memcpy"))


# ----------------------------------------------------------------------
# Dependency invariant: acyclic same-progress graph
# ----------------------------------------------------------------------
#
# ``deps.topological_sort`` cycle failures are surfaced through the
# validator's typed exception.


def test_dependency_depends_on_only_cycle() -> None:
    """A cycle made entirely of ``depends_on`` edges with matching
    lookaheads (so they DO form within-progress topo edges)
    triggers the dependency invariant."""
    a = Task.from_fn(
        name="a",
        fn=_noop,
        depends_on=("b",),
        stream="default",
    )
    b = Task.from_fn(
        name="b",
        fn=_noop,
        depends_on=("a",),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(a, b)),),
        stream_slots=("default",),
    )
    with pytest.raises(ScheduleValidationError, match=r"\[dependency\]"):
        validate(schedule, _pool("default"))


def test_dependency_same_progress_sync_only_cycle() -> None:
    """A cycle made entirely of ``same_progress_sync`` edges
    triggers the dependency invariant (these always add within-progress edges,
    regardless of lookahead)."""
    a = Task.from_fn(
        name="a",
        fn=_noop,
        same_progress_sync=("b",),
        stream="default",
    )
    b = Task.from_fn(
        name="b",
        fn=_noop,
        same_progress_sync=("a",),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(a, b)),),
        stream_slots=("default",),
    )
    with pytest.raises(ScheduleValidationError, match=r"\[dependency\]"):
        validate(schedule, _pool("default"))


def test_dependency_cross_lookahead_depends_on_would_be_cycle_passes() -> None:
    """``depends_on`` with producer.la > consumer.la does NOT add a
    within-progress edge, so a configuration that would cycle if it
    did is still acyclic. The validator must accept it."""
    # consumer (la=0) depends_on producer (la=2). If the edge were
    # added (producer → consumer), and we also add a depends_on going
    # back the other way at MATCHING lookahead, we'd cycle. Using
    # cross-lookahead in one direction breaks the cycle even though
    # lexically two depends_on edges exist.
    consumer = Task.from_fn(
        name="cons",
        fn=_noop,
        lookahead=0,
        depends_on=("prod",),  # cross-la=2 — no topo edge
        stream="default",
    )
    producer = Task.from_fn(
        name="prod",
        fn=_noop,
        lookahead=2,
        # No back edge — without the suppressed topo edge, this is
        # just two unrelated tasks. Verifies the DAG check does not false-
        # positive on cross-lookahead depends_on.
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(consumer, producer)),),
        stream_slots=("default",),
    )
    # Must not raise — cross-lookahead doesn't form within-progress
    # cycles by construction.
    validate(schedule, _pool("default"))


def test_dependency_multistage_forward_depends_on_rejected() -> None:
    """Multi-stage schedules preserve stage barriers, so a same-progress
    dependency cannot rely on the single-stage topological reorder."""
    consumer = Task.from_fn(
        name="consumer",
        fn=_noop,
        depends_on=("producer",),
        stream="default",
    )
    producer = Task.from_fn(name="producer", fn=_noop, stream="default")
    schedule = Schedule(
        stages=(Stage(tasks=(consumer,)), Stage(tasks=(producer,))),
        stream_slots=("default",),
    )
    with pytest.raises(
        ScheduleValidationError,
        match=r"\[dependency\].*Multi-stage schedule.*producer.*consumer",
    ):
        validate(schedule, _pool("default"))


def test_dependency_multistage_ordered_depends_on_passes() -> None:
    producer = Task.from_fn(name="producer", fn=_noop, stream="default")
    consumer = Task.from_fn(
        name="consumer",
        fn=_noop,
        depends_on=("producer",),
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(producer,)), Stage(tasks=(consumer,))),
        stream_slots=("default",),
    )
    validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Dependency invariant: future reads
# ----------------------------------------------------------------------
#
# ``depends_on=("X",)`` with X.lookahead < self.lookahead is a
# "future-read": the producer's same-batch work hasn't run by the
# consumer's iteration, so the wait is unsatisfiable. The validator
# catches this. (Cross_iter_depends_on positive/zero offsets are caught at
# Task.__init__ time and covered by existing
# ``test_engine_task_lookahead.py`` tests.)


def test_dependency_future_read_via_depends_on_rejected() -> None:
    """``depends_on=("X",)`` where X.lookahead < self.lookahead is
    a future-read — the producer hasn't processed the consumer's
    batch yet by the consumer's iteration."""
    producer = Task.from_fn(
        name="prod",
        fn=_noop,
        lookahead=0,  # ran on this iter's batch K = 0; future-batches K=1,2 untouched
        stream="default",
    )
    consumer = Task.from_fn(
        name="cons",
        fn=_noop,
        lookahead=2,  # consumer is processing batch K=2 in this progress
        depends_on=(
            "prod",
        ),  # asks to wait for prod to have processed K=2 — impossible
        stream="default",
    )
    schedule = Schedule(
        stages=(Stage(tasks=(producer, consumer)),),
        stream_slots=("default",),
    )
    with pytest.raises(ScheduleValidationError, match=r"\[dependency\]"):
        validate(schedule, _pool("default"))
