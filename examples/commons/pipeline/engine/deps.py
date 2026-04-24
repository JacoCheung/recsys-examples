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

"""Cross-stream `wait_stream` inference (SPEC §4.2 rule 8).

Given a `Schedule`, produce a mapping

    consumer_task_name -> tuple[producer_stream_name, ...]

that the pipeline driver consults before running each task. For every
cross-stream consumer→producer edge (either a slot read or a
`depends_on` reference), the consumer's stream must `wait_stream` on
the producer's stream so the producer's GPU work is visible before
the consumer submits.

Same-stream edges emit no wait entry — within a single CUDA stream,
task submission order on that stream already gives serial execution.

This module is framework-agnostic: stdlib + `.task` + `.schedule`
only. No torch import.
"""

from typing import Dict, List, Tuple

from .schedule import Schedule
from .task import DataSlot, Task

__all__ = ["infer_cross_stream_waits"]


def _flatten_in_order(schedule: Schedule) -> Tuple[Task, ...]:
    """Every task in declaration order (stage order × within-stage order)."""
    return schedule.all_tasks()


def infer_cross_stream_waits(
    schedule: Schedule,
) -> Dict[str, Tuple[str, ...]]:
    """Map each consumer task name to the producer stream names it
    must `wait_stream` on before running.

    Includes:
      - Slot reads (`task.reads`). Writers of the same slot NAME (at
        any `batch_offset`) are candidates. This correctly captures
        both intra-iter edges (writer@k, reader@k) and cross-iter
        prefetch edges (writer@k1, reader@k2 where k1 > k2 — the
        ring-advance mechanism migrates the slot store from the
        writer's offset position down to the reader's over
        iterations).
      - `task.depends_on` pure-ordering edges (by task name).

    Excludes:
      - Same-stream edges (ordering within one stream is already
        serial by CUDA's per-stream FIFO semantics).
      - Unresolved reads (no writer anywhere). V5 validator rejects
        these; the analyzer tolerates them so the engine runs on
        partially-validated schedules during V2-V4.

    Raises `ValueError` if two tasks write the exact same
    `(name, batch_offset)` slot. SPEC §4.2 rule 4.
    """
    tasks = _flatten_in_order(schedule)

    # Two views of writers:
    #   1) by exact DataSlot key — for the single-writer-per-slot rule
    #   2) by slot NAME — for cross-stream wait inference, which must
    #      see cross-iter (differing-offset) producer/consumer pairs
    #      since the ring-advance mechanism makes the writer's slot
    #      store reach the reader's offset after `k1 - k2` iterations.
    slot_exact_writer: Dict[DataSlot, Task] = {}
    writers_by_slot_name: Dict[str, List[Task]] = {}
    name_to_task: Dict[str, Task] = {}
    for task in tasks:
        name_to_task[task.name] = task
        for slot in task.writes:
            prior = slot_exact_writer.get(slot)
            if prior is not None:
                raise ValueError(
                    f"DataSlot {slot!r} has multiple writers: "
                    f"'{prior.name}' and '{task.name}'. SPEC §4.2 "
                    f"rule 4 requires single writer per slot."
                )
            slot_exact_writer[slot] = task
            writers_by_slot_name.setdefault(slot.name, []).append(task)

    # Additional invariant — name-level stream uniqueness. All
    # writers of a given slot name must share a stream. If
    # writer_a writes `X@1` on stream S1 and writer_b writes `X@0`
    # on stream S2, any reader of `X` would have to wait on BOTH
    # S1 and S2, and there's no principled way to know which
    # producer's data the reader actually consumes after ring
    # advance. Reject to keep cross-stream wait inference
    # unambiguous; re-enable with an explicit per-offset matcher
    # if a future slice needs this pattern.
    for name, writers in writers_by_slot_name.items():
        streams = {w.stream for w in writers}
        if len(streams) > 1:
            raise ValueError(
                f"Slot name {name!r} is written by tasks on multiple "
                f"streams {sorted(streams)!r}. Name-level stream "
                f"uniqueness is required for unambiguous cross-stream "
                f"wait inference. Use distinct slot names per stream "
                f"(e.g. 'X_on_memcpy' vs 'X_on_comm') or merge the "
                f"writers."
            )

    waits: Dict[str, Tuple[str, ...]] = {}
    for consumer in tasks:
        producer_streams: set = set()

        for read_slot in consumer.reads:
            # Any writer of this slot-name (regardless of offset) is
            # a candidate. Cross-iter prefetch: writer@k1 and
            # reader@k2 with k1 > k2 share a slot store via ring
            # advance — their CUDA streams still need wait_stream.
            for writer in writers_by_slot_name.get(read_slot.name, ()):
                if writer.stream != consumer.stream:
                    producer_streams.add(writer.stream)

        for dep_name in consumer.depends_on:
            producer = name_to_task.get(dep_name)
            if producer is None:
                continue
            if producer.stream != consumer.stream:
                producer_streams.add(producer.stream)

        if producer_streams:
            waits[consumer.name] = tuple(sorted(producer_streams))

    return waits
