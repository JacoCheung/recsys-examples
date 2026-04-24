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

    # --- Rule 6: depends_on resolves to an earlier task ------------
    # `name_to_position` already computed above for rule 5.
    for idx, task in enumerate(tasks):
        for dep_name in task.depends_on:
            if dep_name not in name_to_position:
                raise ScheduleValidationError(
                    f"[rule 6] Task {task.name!r}.depends_on references "
                    f"{dep_name!r} which is not a task name in the "
                    f"schedule."
                )
            if name_to_position[dep_name] >= idx:
                raise ScheduleValidationError(
                    f"[rule 6] Task {task.name!r}.depends_on references "
                    f"{dep_name!r} which is NOT strictly earlier in "
                    f"declaration order (dep is at position "
                    f"{name_to_position[dep_name]}, consumer at "
                    f"position {idx})."
                )

    # --- Rule 7: intra-iter DAG acyclic ----------------------------
    # Build adjacency: edge A → B iff B reads-slot-written-by-A with
    # matching offset OR B depends_on A OR A,B same stream and A
    # immediately precedes B. Only intra-iter reads (same offset) are
    # included — cross-iter edges don't form cycles by construction.
    adj: Dict[str, Set[str]] = {t.name: set() for t in tasks}
    for i, a in enumerate(tasks):
        # Same-stream adjacent-in-stage: within a stage, consecutive
        # same-stream tasks get an edge (CPU submission order on the
        # stream).
        for j in range(i + 1, len(tasks)):
            b = tasks[j]
            # Only within the same stage — cross-stage ordering is
            # implicit and not cyclic.
            same_stage = _same_stage(schedule, a, b)
            if not same_stage:
                break
            if a.stream == b.stream:
                adj[a.name].add(b.name)
                # consecutive same-stream in stage: only add edge
                # A → B for the immediate successor; further
                # same-stream tasks transitively reached.
                break
    for consumer in tasks:
        # Intra-iter slot edges
        for read_slot in consumer.reads:
            producer = slot_exact_writer.get(read_slot)
            if producer is not None and producer.name != consumer.name:
                adj[producer.name].add(consumer.name)
        # depends_on edges
        for dep_name in consumer.depends_on:
            if dep_name in adj:
                adj[dep_name].add(consumer.name)

    _raise_if_cyclic(adj)

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


def _same_stage(schedule: Schedule, a: Task, b: Task) -> bool:
    """True iff `a` and `b` belong to the same `Stage`. Used for
    rule 7's same-stream adjacency check."""
    for stage in schedule.stages:
        contains_a = a in stage.tasks
        contains_b = b in stage.tasks
        if contains_a and contains_b:
            return True
        if contains_a or contains_b:
            return False
    return False


def _raise_if_cyclic(adj: Dict[str, Set[str]]) -> None:
    """DFS cycle detection on a directed graph. Raises
    `ScheduleValidationError` on the first cycle found, citing the
    cycle's nodes."""
    WHITE, GRAY, BLACK = 0, 1, 2
    color: Dict[str, int] = {name: WHITE for name in adj}
    stack: List[str] = []

    def _visit(node: str) -> None:
        if color[node] == GRAY:
            cycle_start = stack.index(node)
            cycle = stack[cycle_start:] + [node]
            raise ScheduleValidationError(
                f"[rule 7] Cycle detected in intra-iter task DAG: "
                f"{' -> '.join(cycle)}"
            )
        if color[node] == BLACK:
            return
        color[node] = GRAY
        stack.append(node)
        for nxt in adj.get(node, ()):
            _visit(nxt)
        stack.pop()
        color[node] = BLACK

    for name in adj:
        if color[name] == WHITE:
            _visit(name)
