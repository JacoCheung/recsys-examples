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

"""Schedule validator (SPEC §4.2). Eight rules.

Runs at `SchedulablePipeline.__init__` to reject malformed schedules
before the driver touches CUDA. Replaces the ad-hoc checks previously
scattered in `pipeline.py::__init__` and the duplicate-writer check
in `deps.py`.

The eight rules:

  1. Unique task names across the whole schedule.
  2. `task.batch_offset >= 0`.
  3. `task.stream` ∈ `Schedule.stream_slots` AND ∈ `StreamPool.names()`.
  4. At most one task `writes` any given `DataSlot(name, batch_offset)`.
     Additional V4 invariant: all writers of a given slot NAME share
     one stream (cross-iter wait inference requires unambiguous stream
     per name).
  5. Every `reads(slot)` resolves to a matching writer of the same
     slot NAME (any `batch_offset ≥ reader's offset`).
  6. Every `depends_on=("name",)` resolves to a task with that name
     occurring strictly earlier in declaration order.
  7. The merged intra-iter DAG (slot-read edges + `depends_on` edges
     + same-stream adjacency) is acyclic.
  8. Cross-stream wait_stream edges (inferred by
     `deps.infer_cross_stream_waits`) can be computed without error.
     This is really a consequence of rules 4 + its name-uniqueness
     sub-rule, but we run the analyzer once here to fail fast.
"""

from typing import Dict, List, Optional, Set

from ..deps import infer_cross_stream_waits
from ..schedule import Schedule
from ..streams import StreamPool
from ..task import DataSlot, Task

__all__ = ["ScheduleValidationError", "validate"]


# Slot names populated by the engine itself (not by any user task).
# Rule 5 exempts these from the "must have a writer" requirement.
# SPEC §4.7: the engine calls `next(batch_iter)` and deposits the
# pulled batch into `batch_cpu` at `batch_offset=max_offset`; tasks
# never write this slot themselves.
ENGINE_POPULATED_SLOT_NAMES = frozenset({"batch_cpu"})


class ScheduleValidationError(ValueError):
    """Raised when a Schedule violates one of the §4.2 rules."""


def validate(schedule: Schedule, stream_pool: Optional[StreamPool] = None) -> None:
    """Apply all eight rules to `schedule` (and optionally
    cross-check `stream_pool`). Raises
    `ScheduleValidationError` on the first violation found, with a
    message that names the rule and the offending tasks / slots.

    `stream_pool` is optional — when provided, rule 3 cross-checks
    that every `task.stream` also appears in
    `stream_pool.names()`. When not provided (e.g. pre-flight
    validation at Schedule build time), only the `stream_slots` side
    of rule 3 is enforced.
    """
    tasks = schedule.all_tasks()
    declared_slots = set(schedule.stream_slots)
    pool_names = set(stream_pool.names()) if stream_pool is not None else None

    # --- Rule 1: unique names --------------------------------------
    seen_names: Set[str] = set()
    for task in tasks:
        if task.name in seen_names:
            raise ScheduleValidationError(
                f"[rule 1] Duplicate task name {task.name!r}. Each "
                f"task must have a unique name across the schedule."
            )
        seen_names.add(task.name)

    # --- Rule 2: non-negative batch_offset + slot-offset consistency
    for task in tasks:
        if task.batch_offset < 0:
            raise ScheduleValidationError(
                f"[rule 2] Task {task.name!r}.batch_offset = "
                f"{task.batch_offset}; must be >= 0."
            )
        # Every declared slot in a task's reads/writes must target
        # the same ring position as the task's batch_offset. At
        # runtime `ctx.slots` = `ring.at(task.batch_offset)`, so a
        # mismatched `DataSlot.batch_offset` is metadata that lies
        # about which ring slot is actually touched — the analyzer
        # would build wrong cross-iter edges.
        for slot in task.reads:
            if slot.batch_offset != task.batch_offset:
                raise ScheduleValidationError(
                    f"[rule 2] Task {task.name!r} (batch_offset="
                    f"{task.batch_offset}) declares read of "
                    f"{slot!r}. The slot's batch_offset must equal "
                    f"the task's batch_offset — at runtime "
                    f"`ctx.slots` resolves to `ring.at(task.batch_offset)`."
                )
        for slot in task.writes:
            # Reserved-name guard runs FIRST — reject reserved slot
            # names regardless of the slot's batch_offset, so a user
            # can't dodge the guard by tweaking the offset.
            if slot.name in ENGINE_POPULATED_SLOT_NAMES:
                raise ScheduleValidationError(
                    f"[rule 2] Task {task.name!r} declares write of "
                    f"{slot!r}, but slot name {slot.name!r} is "
                    f"reserved for engine-populated slots (the "
                    f"pipeline driver writes it during `next(batch_iter)` "
                    f"pull). User tasks must not write reserved "
                    f"names; a double-writer would pass validation "
                    f"silently and corrupt data at runtime. Use a "
                    f"different slot name."
                )
            if slot.batch_offset != task.batch_offset:
                raise ScheduleValidationError(
                    f"[rule 2] Task {task.name!r} (batch_offset="
                    f"{task.batch_offset}) declares write of "
                    f"{slot!r}. The slot's batch_offset must equal "
                    f"the task's batch_offset — at runtime "
                    f"`ctx.slots` resolves to `ring.at(task.batch_offset)`."
                )

    # --- Rule 3: stream slot existence -----------------------------
    for task in tasks:
        if task.stream not in declared_slots:
            raise ScheduleValidationError(
                f"[rule 3] Task {task.name!r} binds to stream "
                f"{task.stream!r} not declared in "
                f"Schedule.stream_slots={schedule.stream_slots!r}."
            )
        if pool_names is not None and task.stream not in pool_names:
            raise ScheduleValidationError(
                f"[rule 3] Task {task.name!r} binds to stream "
                f"{task.stream!r} not present in StreamPool "
                f"{tuple(sorted(pool_names))!r}."
            )

    # --- Rule 4: single-writer-per-slot + name-stream uniqueness ---
    slot_exact_writer: Dict[DataSlot, Task] = {}
    writers_by_slot_name: Dict[str, List[Task]] = {}
    for task in tasks:
        for slot in task.writes:
            prior = slot_exact_writer.get(slot)
            if prior is not None:
                raise ScheduleValidationError(
                    f"[rule 4] DataSlot {slot!r} has multiple "
                    f"writers: {prior.name!r} and {task.name!r}. "
                    f"Single writer per slot required."
                )
            slot_exact_writer[slot] = task
            writers_by_slot_name.setdefault(slot.name, []).append(task)

    for name, writers in writers_by_slot_name.items():
        streams = {w.stream for w in writers}
        if len(streams) > 1:
            raise ScheduleValidationError(
                f"[rule 4] Slot name {name!r} is written by tasks on "
                f"multiple streams {sorted(streams)!r}. Name-level "
                f"stream uniqueness required for unambiguous "
                f"cross-stream wait inference. Writers: "
                f"{[w.name for w in writers]!r}. Fix: use distinct "
                f"slot names per stream, or merge the writers."
            )

    # --- Rule 5: reads resolve -------------------------------------
    name_to_position: Dict[str, int] = {t.name: i for i, t in enumerate(tasks)}
    for reader_idx, task in enumerate(tasks):
        for read_slot in task.reads:
            # Engine-populated slots (e.g. "batch_cpu") are deposited
            # by the pipeline driver at runtime, not by user tasks.
            # Rule 5 exempts them.
            if read_slot.name in ENGINE_POPULATED_SLOT_NAMES:
                continue
            writers = writers_by_slot_name.get(read_slot.name, [])
            if not writers:
                raise ScheduleValidationError(
                    f"[rule 5] Task {task.name!r} reads slot "
                    f"{read_slot!r} but no task writes slot name "
                    f"{read_slot.name!r} anywhere in the schedule."
                )
            # Reject writer_offset < reader_offset (future-read):
            # ring-advance cannot surface data from a lower offset.
            max_writer_offset = max(w.batch_offset for w in writers)
            if max_writer_offset < read_slot.batch_offset:
                raise ScheduleValidationError(
                    f"[rule 5] Task {task.name!r} reads slot "
                    f"{read_slot!r} but no writer of {read_slot.name!r} "
                    f"is at batch_offset >= {read_slot.batch_offset} "
                    f"(max writer offset = {max_writer_offset}). The "
                    f"ring-advance mechanism cannot surface data from "
                    f"a lower offset than the reader's offset."
                )
            # Declaration-order rule: for intra-iter reads (writer at
            # same offset as reader), the writer MUST be declared
            # strictly earlier than the reader. Otherwise the reader
            # runs (CPU-wise) before the writer this iteration and
            # observes either stale or missing data.
            # Cross-iter reads (writer_offset > reader_offset) have
            # no declaration-order constraint — ring-advance surfaces
            # the previous iter's data before the reader runs.
            same_offset_writers = [
                w for w in writers if w.batch_offset == read_slot.batch_offset
            ]
            # If no same-offset writer exists but a higher-offset
            # writer does, it's a legitimate cross-iter edge. If a
            # same-offset writer exists, it must be declared earlier
            # than the reader.
            if same_offset_writers:
                for writer in same_offset_writers:
                    writer_idx = name_to_position[writer.name]
                    if writer_idx >= reader_idx:
                        raise ScheduleValidationError(
                            f"[rule 5] Task {task.name!r} (position "
                            f"{reader_idx}) reads slot {read_slot!r} "
                            f"written by {writer.name!r} at position "
                            f"{writer_idx}. Intra-iter reads require "
                            f"the writer to appear strictly earlier "
                            f"in declaration order; {writer.name!r} "
                            f"runs {'at the same position' if writer_idx == reader_idx else 'AFTER'} "
                            f"the reader."
                        )

    # --- Rule 6: dependency fields resolve to known tasks -----------
    #
    # The Task class has three dependency fields, each with a distinct
    # semantic but the same "producer name must exist in the schedule"
    # validity rule:
    #
    #   ``depends_on``                 — same-batch logical dep
    #   ``cross_iter_depends_on``      — different-batch logical dep
    #   ``same_progress_sync``         — same-progress GPU coherency
    #
    # Without this check, a misspelled producer name silently drops the
    # edge (deps.py uses ``name_to_task.get(name)`` which returns None
    # for unknown names, then ``continue``s the loop) — the consumer
    # would silently lose the wait.
    #
    # NOTE on declaration order: the engine sorts tasks by within-
    # progress DAG topological order at SchedulablePipeline
    # construction (see ``deps.topological_sort``); declaration order
    # only serves as a tie-breaker. Therefore this rule no longer
    # enforces "producer declared earlier than consumer". The DAG must
    # still be acyclic, which rule 7 below verifies.
    name_to_task: Dict[str, "Task"] = {t.name: t for t in tasks}
    for task in tasks:
        for dep_name in task.depends_on:
            if dep_name not in name_to_task:
                raise ScheduleValidationError(
                    f"[rule 6] Task {task.name!r}.depends_on references "
                    f"{dep_name!r} which is not a task name in the "
                    f"schedule."
                )
            producer = name_to_task[dep_name]
            if producer.batch_offset < task.batch_offset:
                # Future-read: producer's lookahead is smaller, so it
                # has not yet processed the consumer's batch by the
                # consumer's iteration. Independently caught by
                # ``deps.infer_cross_stream_event_deps`` but surface
                # here too for early failure.
                raise ScheduleValidationError(
                    f"[rule 6] Task {task.name!r}.depends_on references "
                    f"{dep_name!r} but {dep_name!r}.lookahead="
                    f"{producer.batch_offset} < {task.name!r}.lookahead="
                    f"{task.batch_offset}. Cannot wait for a producer "
                    f"that has not yet run by the consumer's iteration."
                )
        for dep_name, _neg in getattr(task, "cross_iter_depends_on", ()):
            if dep_name not in name_to_task:
                raise ScheduleValidationError(
                    f"[rule 6] Task {task.name!r}.cross_iter_depends_on "
                    f"references {dep_name!r} which is not a task name "
                    f"in the schedule."
                )
        for dep_name in getattr(task, "same_progress_sync", ()):
            if dep_name not in name_to_task:
                raise ScheduleValidationError(
                    f"[rule 6] Task {task.name!r}.same_progress_sync "
                    f"references {dep_name!r} which is not a task name "
                    f"in the schedule."
                )

    # --- Rule 7: within-progress DAG acyclic -----------------------
    # The engine sorts tasks topologically by within-progress edges
    # (reads/writes exact-slot match, depends_on with matching
    # lookahead, same_progress_sync). A cycle in those edges makes
    # topological_sort impossible. Run it here for early failure.
    #
    # Cross-iter / different-batch edges (cross_iter_depends_on, or
    # reads/writes with differing offsets) do NOT participate in the
    # within-progress DAG — they're handled by BatchRing slot rotation
    # and cannot form cycles by construction.
    from ..deps import topological_sort

    try:
        topological_sort(schedule)
    except ValueError as e:
        raise ScheduleValidationError(
            f"[rule 7] Within-progress DAG is not acyclic: {e}"
        ) from e

    # --- Rule 8: cross-stream wait inference runs without error ----
    # Safety net (rules 4's sub-checks already cover every known
    # deps.py failure mode; this just wraps any future deps.py
    # assertions into our exception type so the contract holds).
    try:
        infer_cross_stream_waits(schedule)
    except ValueError as e:
        # Re-raise as our typed exception to preserve the
        # `validate()` contract. `ScheduleValidationError` is a
        # `ValueError` subclass, so catch blocks on the base type
        # still work.
        raise ScheduleValidationError(
            f"[rule 8] Cross-stream wait inference rejected the " f"schedule: {e}"
        ) from e
