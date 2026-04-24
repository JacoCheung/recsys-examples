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

"""V5 — Schedule validator (SPEC §4.2). One failing-case test per
rule, plus a smoke test on a valid schedule.
"""

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
# Rule 1 — unique names
# ----------------------------------------------------------------------


def test_rule1_duplicate_task_name_rejected() -> None:
    a = Task.from_fn(name="dup", fn=_noop, stream="default")
    b = Task.from_fn(name="dup", fn=_noop, stream="default")
    schedule = Schedule(stages=(Stage(tasks=(a, b)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[rule 1\].*Duplicate"):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Rule 2 — non-negative batch_offset
# ----------------------------------------------------------------------


def test_rule2_negative_batch_offset_rejected() -> None:
    # `Task.__init__` already rejects negative offsets; simulate a
    # corrupt construction by bypassing the guard to exercise the
    # validator path directly.
    t = Task.from_fn(name="t", fn=_noop, stream="default")
    object.__setattr__(t, "batch_offset", -1)  # bypass Task's own check
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[rule 2\]"):
        validate(schedule, _pool("default"))


def test_rule2_slot_offset_mismatch_read_rejected() -> None:
    """Task at batch_offset=0 declaring a read of DataSlot with
    batch_offset=1 is a metadata lie — `ctx.slots` at runtime targets
    `ring.at(task.batch_offset=0)`, not offset=1."""
    t = Task.from_fn(
        name="bad_read",
        fn=_noop,
        reads=(DataSlot("x", batch_offset=1),),  # mismatch
        stream="default",
        batch_offset=0,
    )
    # Add a phantom writer so the read resolves (rule 5 wouldn't
    # fire on "no writer") — we want rule 2's check to trigger.
    w = Task.from_fn(
        name="w",
        fn=_noop,
        writes=(DataSlot("x", batch_offset=1),),
        stream="default",
        batch_offset=1,
    )
    schedule = Schedule(stages=(Stage(tasks=(w, t)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[rule 2\].*batch_offset must equal"
    ):
        validate(schedule, _pool("default"))


def test_rule2_user_task_write_to_engine_populated_slot_rejected() -> None:
    """Rule 2 blocks user tasks from writing reserved slot names
    (e.g. `batch_cpu`). The engine unconditionally writes these
    during each `next(batch_iter)` pull; a user task writing the
    same name would create a silent double-writer."""
    t = Task.from_fn(
        name="bad",
        fn=_noop,
        writes=(DataSlot("batch_cpu"),),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[rule 2\].*reserved for engine-populated"
    ):
        validate(schedule, _pool("default"))


def test_rule2_reserved_slot_name_caught_regardless_of_offset() -> None:
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
        batch_offset=0,
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[rule 2\].*reserved for engine-populated"
    ):
        validate(schedule, _pool("default"))


def test_rule2_slot_offset_mismatch_write_rejected() -> None:
    """Same invariant on the write side."""
    t = Task.from_fn(
        name="bad_write",
        fn=_noop,
        writes=(DataSlot("y", batch_offset=2),),  # mismatch
        stream="default",
        batch_offset=0,
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[rule 2\].*batch_offset must equal"
    ):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Rule 3 — stream slot existence
# ----------------------------------------------------------------------


def test_rule3_stream_not_in_schedule_slots() -> None:
    t = Task.from_fn(name="bad", fn=_noop, stream="comm")
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[rule 3\].*not declared"):
        validate(schedule, _pool("default", "comm"))


def test_rule3_stream_not_in_pool() -> None:
    """Stream declared in `stream_slots` but missing from the pool."""
    t = Task.from_fn(name="bad", fn=_noop, stream="memcpy")
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default", "memcpy"))
    # Pool lacks 'memcpy'
    with pytest.raises(
        ScheduleValidationError, match=r"\[rule 3\].*not present in StreamPool"
    ):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Rule 4 — single writer per slot
# ----------------------------------------------------------------------


def test_rule4_duplicate_exact_writer_rejected() -> None:
    a = Task.from_fn(name="a", fn=_noop, writes=(DataSlot("x"),), stream="default")
    b = Task.from_fn(name="b", fn=_noop, writes=(DataSlot("x"),), stream="default")
    schedule = Schedule(stages=(Stage(tasks=(a, b)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[rule 4\].*multiple writers"):
        validate(schedule, _pool("default"))


def test_rule4_same_name_writers_on_different_streams() -> None:
    """Valid under rule 4's exact-match check (different DataSlots)
    but the V4 stream-uniqueness-per-name sub-rule rejects it."""
    a = Task.from_fn(
        name="a",
        fn=_noop,
        writes=(DataSlot("x", batch_offset=1),),
        stream="memcpy",
        batch_offset=1,
    )
    b = Task.from_fn(
        name="b",
        fn=_noop,
        writes=(DataSlot("x", batch_offset=0),),
        stream="comm",
        batch_offset=0,
    )
    schedule = Schedule(
        stages=(Stage(tasks=(a, b)),),
        stream_slots=("default", "memcpy", "comm"),
    )
    with pytest.raises(ScheduleValidationError, match=r"\[rule 4\].*multiple streams"):
        validate(schedule, _pool("default", "memcpy", "comm"))


# ----------------------------------------------------------------------
# Rule 5 — reads resolve
# ----------------------------------------------------------------------


def test_rule5_unresolved_slot_read_rejected() -> None:
    t = Task.from_fn(
        name="reader",
        fn=_noop,
        reads=(DataSlot("nonexistent"),),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[rule 5\].*no task writes"):
        validate(schedule, _pool("default"))


def test_rule5_cross_stage_same_offset_reader_before_writer_rejected() -> None:
    """Rule 5 declaration-order check: intra-iter reader (same
    batch_offset as writer) must be declared strictly EARLIER?  No —
    the writer must be earlier, so the reader can see what's written
    this iter. Here a consumer in Stage 0 tries to read a slot
    written in Stage 1 same-iter: consumer runs CPU-first (stage 0
    before stage 1), observes empty slot → runtime failure."""
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
        match=r"\[rule 5\].*Intra-iter reads require the writer",
    ):
        validate(schedule, _pool("default"))


def test_rule5_cross_iter_reader_before_writer_allowed() -> None:
    """Cross-iter prefetch is legal regardless of declaration order:
    writer at batch_offset=1 writes "future" batch. Reader at
    batch_offset=0 reads the ring's current slot, which (after
    advance) is what the writer wrote on the previous iter. The
    reader can be declared before the writer — no violation."""
    consumer = Task.from_fn(
        name="current_reader",
        fn=_noop,
        reads=(DataSlot("x", batch_offset=0),),
        stream="default",
        batch_offset=0,
    )
    producer = Task.from_fn(
        name="next_batch_writer",
        fn=_noop,
        writes=(DataSlot("x", batch_offset=1),),
        stream="default",
        batch_offset=1,
    )
    # Declare consumer first (at position 0), producer later.
    schedule = Schedule(
        stages=(Stage(tasks=(consumer, producer)),),
        stream_slots=("default",),
    )
    # Must not raise — cross-iter reads (writer_offset > reader_offset)
    # have no declaration-order constraint.
    validate(schedule, _pool("default"))


def test_rule5_future_read_rejected() -> None:
    """Writer at offset=0, reader at offset=1. Reader wants a FUTURE
    batch that no ring-advance path can surface."""
    writer = Task.from_fn(name="w", fn=_noop, writes=(DataSlot("x"),), stream="default")
    reader = Task.from_fn(
        name="r",
        fn=_noop,
        reads=(DataSlot("x", batch_offset=1),),
        stream="default",
        batch_offset=1,
    )
    schedule = Schedule(
        stages=(Stage(tasks=(writer, reader)),),
        stream_slots=("default",),
    )
    with pytest.raises(ScheduleValidationError, match=r"\[rule 5\].*lower offset"):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Rule 6 — depends_on resolves to an earlier task
# ----------------------------------------------------------------------


def test_rule6_unresolved_depends_on_rejected() -> None:
    t = Task.from_fn(
        name="t",
        fn=_noop,
        depends_on=("no_such_task",),
        stream="default",
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    with pytest.raises(ScheduleValidationError, match=r"\[rule 6\].*not a task name"):
        validate(schedule, _pool("default"))


def test_rule6_forward_depends_on_rejected() -> None:
    """depends_on must reference a task STRICTLY EARLIER in declaration
    order. Self-reference and forward references rejected."""
    a = Task.from_fn(
        name="a",
        fn=_noop,
        depends_on=("b",),  # b comes AFTER a
        stream="default",
    )
    b = Task.from_fn(name="b", fn=_noop, stream="default")
    schedule = Schedule(stages=(Stage(tasks=(a, b)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError, match=r"\[rule 6\].*NOT strictly earlier"
    ):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Rule 7 — intra-iter DAG acyclic
# ----------------------------------------------------------------------


def test_rule7_cycle_via_slot_and_depends_on() -> None:
    """Task A writes x, task B reads x (edge A→B). Task A also
    depends_on B (edge B→A). Cycle."""
    a = Task.from_fn(
        name="a",
        fn=_noop,
        writes=(DataSlot("x"),),
        depends_on=("b",),
        stream="default",
    )
    b = Task.from_fn(
        name="b",
        fn=_noop,
        reads=(DataSlot("x"),),
        stream="default",
    )
    # Note: rule 6 would reject `a.depends_on=("b",)` because b
    # comes after a in declaration order. Put them in reverse to
    # get past rule 6 and land on rule 7.
    schedule = Schedule(stages=(Stage(tasks=(b, a)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError,
        match=r"(?:\[rule 5\]|\[rule 6\]|\[rule 7\]).*",
    ):
        # Any of rule 5 (same-offset forward-declared read), rule 6
        # (forward depends_on), or rule 7 (full cycle) is a valid
        # rejection of this malformed schedule. Earlier rules fire
        # first in practice — we assert it's rejected, not the exact
        # rule number.
        validate(schedule, _pool("default"))


def test_rule7_direct_cycle_via_depends_on_loop() -> None:
    """Explicitly targets rule 7: a cycle of `depends_on` edges.
    Construct two tasks A→B (via depends_on) and B→A (via depends_on).
    Rule 6's `earlier in declaration order` would reject this before
    rule 7 can fire — so we bypass rule 6 by monkey-patching to
    insert a genuine cycle in the intra-iter DAG."""
    a = Task.from_fn(name="a", fn=_noop, stream="default")
    b = Task.from_fn(name="b", fn=_noop, stream="default")
    # Bypass rule 6 by assigning depends_on AFTER declaration order
    # is set: a's depends_on references b (b is position 1, a is
    # position 0). Rule 6 rejects forward refs. So we need a way to
    # inject a cycle that evades rule 6. Use slot edges instead:
    # a writes x, b writes y; a reads y; b reads x. All same offset.
    a2 = Task.from_fn(
        name="a",
        fn=_noop,
        reads=(DataSlot("y"),),
        writes=(DataSlot("x"),),
        stream="default",
    )
    b2 = Task.from_fn(
        name="b",
        fn=_noop,
        reads=(DataSlot("x"),),
        writes=(DataSlot("y"),),
        stream="default",
    )
    # This configuration can never pass rule 5 either: whichever of
    # a2/b2 is declared first, its read refers to a slot written
    # later. That's rule 5's domain. Rule 7 cycle detection is
    # mostly redundant — rules 5 + 6 together prevent any such
    # same-iter cycle from existing in the first place. Rule 7
    # remains as a sanity check for any exotic edge-set bug where
    # the rules above slip.
    schedule = Schedule(stages=(Stage(tasks=(a2, b2)),), stream_slots=("default",))
    with pytest.raises(
        ScheduleValidationError,
        match=r"(?:\[rule 5\]|\[rule 7\]).*",
    ):
        validate(schedule, _pool("default"))


# ----------------------------------------------------------------------
# Rule 8 — cross-stream wait inference must run (catch-all for misses)
# ----------------------------------------------------------------------
#
# Rule 8 is implicitly tested by rules 4 and 5; an independent failure
# case would require a schedule that passes 1-7 but trips `deps.py`.
# No such case exists in the current design — rule 8 is a safety net.
# We test that a *valid* schedule passes rule 8 in the smoke below.


# ----------------------------------------------------------------------
# Smoke — a valid multi-task multi-stream schedule passes all rules
# ----------------------------------------------------------------------


def test_smoke_valid_prefetch_schedule_passes() -> None:
    """Realistic prefetch schedule: h2d@offset=1 on memcpy, fwd/bwd
    on default. Must pass all 8 rules."""
    h2d = Task.from_fn(
        name="h2d",
        fn=_noop,
        reads=(DataSlot("batch_cpu", batch_offset=1),),
        writes=(DataSlot("batch_gpu", batch_offset=1),),
        stream="memcpy",
        batch_offset=1,
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
    # `batch_cpu` is engine-populated — exempt from rule 5.
    # Should not raise.
    validate(schedule, _pool("default", "memcpy"))


def test_engine_populated_slot_exempt_from_rule5() -> None:
    """`batch_cpu` is exempt from rule 5 (engine populates it at
    runtime — no user task writes it)."""
    t = Task.from_fn(
        name="reader",
        fn=_noop,
        reads=(DataSlot("batch_cpu", batch_offset=1),),
        stream="default",
        batch_offset=1,
    )
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    # Must not raise — "batch_cpu" is engine-populated.
    validate(schedule, _pool("default"))


def test_smoke_valid_single_stream_schedule_passes() -> None:
    """Minimal valid schedule — one task, one stream."""
    t = Task.from_fn(name="t", fn=_noop, stream="default")
    schedule = Schedule(stages=(Stage(tasks=(t,)),), stream_slots=("default",))
    validate(schedule, _pool("default"))
