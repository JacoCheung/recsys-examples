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

"""Schedule validator.

Runs at ``SchedulablePipeline.__init__`` to reject malformed schedules
before the driver touches CUDA. The checks are grouped by invariant so
authors do not need to memorize a numbered checklist:

  * task: unique names, non-negative offsets, and truthful slot offsets.
  * resource: every referenced stream is declared and available.
  * data: slot writes are unique and reads can resolve.
  * dependency: named deps exist, are acyclic, and fit the schedule order.
    Multi-stage schedules keep stage barriers, so CPU-side same-progress
    deps must already point forward in declaration order.
"""

from typing import Dict, List, Optional, Set

from ..deps import _same_progress_dependency_predecessors, infer_cross_stream_waits
from ..schedule import Schedule
from ..streams import StreamPool
from ..task import DataSlot, Task

__all__ = ["ScheduleValidationError", "validate"]


# Slot names populated by the engine itself (not by any user task).
# Reads of these names do not need a user-authored writer.
ENGINE_POPULATED_SLOT_NAMES = frozenset({"batch_cpu"})


class ScheduleValidationError(ValueError):
    """Raised when a Schedule violates the engine contract."""


def validate(schedule: Schedule, stream_pool: Optional[StreamPool] = None) -> None:
    """Validate task, resource, data, and dependency invariants.

    ``stream_pool`` is optional. When provided, every ``task.stream``
    must also appear in ``stream_pool.names()``; otherwise only
    ``Schedule.stream_slots`` is checked.
    """
    tasks = schedule.all_tasks()
    declared_slots = set(schedule.stream_slots)
    pool_names = set(stream_pool.names()) if stream_pool is not None else None

    # --- Task invariants ------------------------------------------
    seen_names: Set[str] = set()
    for task in tasks:
        if task.name in seen_names:
            raise ScheduleValidationError(
                f"[task] Duplicate task name {task.name!r}. Each "
                f"task must have a unique name across the schedule."
            )
        seen_names.add(task.name)

    for task in tasks:
        if task.batch_offset < 0:
            raise ScheduleValidationError(
                f"[task] Task {task.name!r}.batch_offset = "
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
                    f"[task] Task {task.name!r} (batch_offset="
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
                    f"[task] Task {task.name!r} declares write of "
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
                    f"[task] Task {task.name!r} (batch_offset="
                    f"{task.batch_offset}) declares write of "
                    f"{slot!r}. The slot's batch_offset must equal "
                    f"the task's batch_offset — at runtime "
                    f"`ctx.slots` resolves to `ring.at(task.batch_offset)`."
                )

    # --- Resource invariants --------------------------------------
    for task in tasks:
        if task.stream not in declared_slots:
            raise ScheduleValidationError(
                f"[resource] Task {task.name!r} binds to stream "
                f"{task.stream!r} not declared in "
                f"Schedule.stream_slots={schedule.stream_slots!r}."
            )
        if pool_names is not None and task.stream not in pool_names:
            raise ScheduleValidationError(
                f"[resource] Task {task.name!r} binds to stream "
                f"{task.stream!r} not present in StreamPool "
                f"{tuple(sorted(pool_names))!r}."
            )

    # --- Data-slot invariants -------------------------------------
    slot_exact_writer: Dict[DataSlot, Task] = {}
    writers_by_slot_name: Dict[str, List[Task]] = {}
    for task in tasks:
        for slot in task.writes:
            prior = slot_exact_writer.get(slot)
            if prior is not None:
                raise ScheduleValidationError(
                    f"[data] DataSlot {slot!r} has multiple "
                    f"writers: {prior.name!r} and {task.name!r}. "
                    f"Single writer per slot required."
                )
            slot_exact_writer[slot] = task
            writers_by_slot_name.setdefault(slot.name, []).append(task)

    for name, writers in writers_by_slot_name.items():
        streams = {w.stream for w in writers}
        if len(streams) > 1:
            raise ScheduleValidationError(
                f"[data] Slot name {name!r} is written by tasks on "
                f"multiple streams {sorted(streams)!r}. Name-level "
                f"stream uniqueness required for unambiguous "
                f"cross-stream wait inference. Writers: "
                f"{[w.name for w in writers]!r}. Fix: use distinct "
                f"slot names per stream, or merge the writers."
            )

    name_to_position: Dict[str, int] = {t.name: i for i, t in enumerate(tasks)}
    for reader_idx, task in enumerate(tasks):
        for read_slot in task.reads:
            # Engine-populated slots (e.g. "batch_cpu") are deposited
            # by the pipeline driver at runtime, not by user tasks.
            if read_slot.name in ENGINE_POPULATED_SLOT_NAMES:
                continue
            writers = writers_by_slot_name.get(read_slot.name, [])
            if not writers:
                raise ScheduleValidationError(
                    f"[data] Task {task.name!r} reads slot "
                    f"{read_slot!r} but no task writes slot name "
                    f"{read_slot.name!r} anywhere in the schedule."
                )
            # Reject writer_offset < reader_offset (future-read):
            # ring-advance cannot surface data from a lower offset.
            max_writer_offset = max(w.batch_offset for w in writers)
            if max_writer_offset < read_slot.batch_offset:
                raise ScheduleValidationError(
                    f"[data] Task {task.name!r} reads slot "
                    f"{read_slot!r} but no writer of {read_slot.name!r} "
                    f"is at batch_offset >= {read_slot.batch_offset} "
                    f"(max writer offset = {max_writer_offset}). The "
                    f"ring-advance mechanism cannot surface data from "
                    f"a lower offset than the reader's offset."
                )
            # Declaration-order check: for intra-iter reads (writer at
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
                            f"[data] Task {task.name!r} (position "
                            f"{reader_idx}) reads slot {read_slot!r} "
                            f"written by {writer.name!r} at position "
                            f"{writer_idx}. Intra-iter reads require "
                            f"the writer to appear strictly earlier "
                            f"in declaration order; {writer.name!r} "
                            f"runs {'at the same position' if writer_idx == reader_idx else 'AFTER'} "
                            f"the reader."
                        )

    # --- Dependency and order invariants ---------------------------
    #
    # The Task class has three dependency fields with distinct
    # semantics, but all of them reference producer task names:
    #
    #   ``depends_on``                 — same-batch logical dep
    #   ``cross_iter_depends_on``      — different-batch logical dep
    #   ``same_progress_sync``         — current-progress sync
    #
    # Without this check, a misspelled producer name silently drops the
    # edge (deps.py uses ``name_to_task.get(name)`` which returns None
    # for unknown names, then ``continue``s the loop) — the consumer
    # would silently lose the wait.
    #
    # NOTE on declaration order: single-stage schedules are reordered
    # topologically at construction time, with declaration order only
    # used as a tie-breaker. Multi-stage schedules preserve stage
    # barriers and are checked below.
    name_to_task: Dict[str, "Task"] = {t.name: t for t in tasks}
    for task in tasks:
        for dep_name in task.depends_on:
            if dep_name not in name_to_task:
                raise ScheduleValidationError(
                    f"[dependency] Task {task.name!r}.depends_on references "
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
                    f"[dependency] Task {task.name!r}.depends_on references "
                    f"{dep_name!r} but {dep_name!r}.lookahead="
                    f"{producer.batch_offset} < {task.name!r}.lookahead="
                    f"{task.batch_offset}. Cannot wait for a producer "
                    f"that has not yet run by the consumer's iteration."
                )
        for dep_name, _neg in getattr(task, "cross_iter_depends_on", ()):
            if dep_name not in name_to_task:
                raise ScheduleValidationError(
                    f"[dependency] Task {task.name!r}.cross_iter_depends_on "
                    f"references {dep_name!r} which is not a task name "
                    f"in the schedule."
                )
        for dep_name in getattr(task, "same_progress_sync", ()):
            if dep_name not in name_to_task:
                raise ScheduleValidationError(
                    f"[dependency] Task {task.name!r}.same_progress_sync "
                    f"references {dep_name!r} which is not a task name "
                    f"in the schedule."
                )

    # The engine sorts single-stage schedules by within-progress
    # edges. A cycle in those edges makes topological_sort impossible,
    # so fail before constructing the pipeline.
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
            f"[dependency] Within-progress DAG is not acyclic: {e}"
        ) from e

    if len(schedule.stages) > 1:
        same_progress_edges = _same_progress_dependency_predecessors(schedule)
        for consumer_name, producer_names in same_progress_edges.items():
            consumer_pos = name_to_position[consumer_name]
            for producer_name in producer_names:
                producer_pos = name_to_position[producer_name]
                if producer_pos >= consumer_pos:
                    raise ScheduleValidationError(
                        f"[dependency] Multi-stage schedule declares a "
                        f"same-progress dependency from producer "
                        f"{producer_name!r} (position {producer_pos}) to "
                        f"consumer {consumer_name!r} (position "
                        f"{consumer_pos}), but multi-stage execution "
                        f"preserves declaration/stage order. Move the "
                        f"producer earlier, or collapse the tasks into a "
                        f"single stage so SchedulablePipeline can "
                        f"topologically reorder them."
                    )

    # Cross-stream wait inference should not discover a separate
    # contract violation. Wrap future dependency-analysis failures in
    # the validator's public exception type.
    try:
        infer_cross_stream_waits(schedule)
    except ValueError as e:
        # Re-raise as our typed exception to preserve the
        # `validate()` contract. `ScheduleValidationError` is a
        # `ValueError` subclass, so catch blocks on the base type
        # still work.
        raise ScheduleValidationError(
            f"[dependency] Cross-stream wait inference rejected the schedule: {e}"
        ) from e
