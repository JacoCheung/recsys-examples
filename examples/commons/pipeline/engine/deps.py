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

"""Dependency inference utilities.

The engine builds two views of a schedule: a same-progress DAG used for
task submission order, and cross-stream wait sets used for CUDA sync.
Same-stream edges are omitted because CUDA stream FIFO already orders
submissions on one stream.

This module is framework-agnostic: stdlib plus local schedule/task
types only. No torch import.
"""

from typing import Dict, List, Set, Tuple

from .schedule import Schedule
from .task import (
    DataSlot,
    Task,
    same_progress_sync_uses_cpu,
    same_progress_sync_uses_gpu,
)

__all__ = [
    "infer_cross_stream_waits",
    "infer_cross_stream_event_deps",
    "producers_with_cross_stream_consumers",
    "topological_sort",
]


def _flatten_in_order(schedule: Schedule) -> Tuple[Task, ...]:
    """Every task in declaration order (stage order × within-stage order)."""
    return schedule.all_tasks()


def _build_same_progress_dag_edges(
    tasks: Tuple[Task, ...],
) -> Dict[str, Set[str]]:
    """Build consumer -> producer edges for one ``progress()`` call.

    Edges come from exact-slot reads, same-offset ``depends_on``,
    CPU-side ``same_progress_sync``, and delta-zero
    ``cross_iter_depends_on``.
    Different-offset data deps are handled by BatchRing rotation, so
    they do not constrain current-progress submission order.
    """
    incoming: Dict[str, Set[str]] = {t.name: set() for t in tasks}

    # Slot-based edges: writer of (X, k) → reader of (X, k).
    writers: Dict[DataSlot, str] = {}
    for task in tasks:
        for slot in task.writes:
            writers[slot] = task.name
    for task in tasks:
        for slot in task.reads:
            writer_name = writers.get(slot)
            if writer_name and writer_name != task.name:
                incoming[task.name].add(writer_name)

    name_to_task: Dict[str, Task] = {t.name: t for t in tasks}

    # ``depends_on`` (same-batch logical) — only emit edge when
    # producer and consumer have the same lookahead, otherwise the
    # producer's batch K work happened in an earlier progress.
    for task in tasks:
        for dep_name in task.depends_on:
            producer = name_to_task.get(dep_name)
            if producer is None or producer.name == task.name:
                continue
            if producer.batch_offset == task.batch_offset:
                incoming[task.name].add(producer.name)

    # CPU-side ``same_progress_sync`` — emits a current-progress edge
    # regardless of lookahead, since by definition both tasks are in the
    # current progress. This edge drives topo/ticket/host ordering.
    for task in tasks:
        if not same_progress_sync_uses_cpu(task):
            continue
        for dep_name in getattr(task, "same_progress_sync", ()):
            producer = name_to_task.get(dep_name)
            if producer is None or producer.name == task.name:
                continue
            incoming[task.name].add(producer.name)

    # Delta-zero cross-iter deps are same-progress ordering edges.
    for task in tasks:
        for dep_name, neg_offset in getattr(task, "cross_iter_depends_on", ()):
            producer = name_to_task.get(dep_name)
            if producer is None or producer.name == task.name:
                continue
            N = -neg_offset
            delta = producer.batch_offset + N - task.batch_offset
            if delta == 0:
                incoming[task.name].add(producer.name)

    return incoming


def _same_progress_dependency_predecessors(
    schedule: Schedule,
) -> Dict[str, Tuple[str, ...]]:
    """Return same-progress dependency edges as consumer -> producers.

    Edges here require the producer to be submitted before the consumer
    within a single ``progress()`` call. The pipeline uses the same edge
    set for topological ordering, and the validator uses it to ensure
    multi-stage schedules do not declare a dependency that crosses a
    stage barrier backwards.
    """

    return _same_progress_dependency_predecessors_for_tasks(_flatten_in_order(schedule))


def _same_progress_dependency_predecessors_for_tasks(
    tasks: Tuple[Task, ...],
) -> Dict[str, Tuple[str, ...]]:
    incoming = _build_same_progress_dag_edges(tasks)
    return {
        consumer_name: tuple(sorted(producer_names))
        for consumer_name, producer_names in incoming.items()
        if producer_names
    }


def topological_sort(schedule: Schedule) -> Tuple[Task, ...]:
    """Return tasks ordered by within-progress DAG topological sort.

    Replaces "declaration order" as the engine's execution-order
    source. Edges are computed by ``_build_same_progress_dag_edges``;
    tie-breaks among DAG-independent tasks fall back to declaration
    order so author intent is preserved when the DAG is silent.

    Raises ``ValueError`` if the DAG contains a cycle.
    """
    tasks = _flatten_in_order(schedule)
    predecessor_map = _same_progress_dependency_predecessors_for_tasks(tasks)
    incoming = {task.name: set(predecessor_map.get(task.name, ())) for task in tasks}

    # Outgoing adjacency for efficient propagation.
    outgoing: Dict[str, List[str]] = {t.name: [] for t in tasks}
    for consumer_name, producer_names in incoming.items():
        for producer_name in producer_names:
            outgoing[producer_name].append(consumer_name)

    # In-degree count.
    in_degree: Dict[str, int] = {t.name: len(incoming[t.name]) for t in tasks}

    # Tie-break by declaration order: tasks ready at the same step are
    # picked in their original schedule position.
    declaration_pos: Dict[str, int] = {t.name: i for i, t in enumerate(tasks)}

    ready: List[str] = sorted(
        [t.name for t in tasks if in_degree[t.name] == 0],
        key=lambda n: declaration_pos[n],
    )
    sorted_names: List[str] = []
    while ready:
        n = ready.pop(0)
        sorted_names.append(n)
        for next_name in outgoing[n]:
            in_degree[next_name] -= 1
            if in_degree[next_name] == 0:
                # Insert maintaining declaration-order tie-break.
                pos = declaration_pos[next_name]
                inserted = False
                for i, existing in enumerate(ready):
                    if declaration_pos[existing] > pos:
                        ready.insert(i, next_name)
                        inserted = True
                        break
                if not inserted:
                    ready.append(next_name)

    if len(sorted_names) != len(tasks):
        remaining = [t.name for t in tasks if t.name not in sorted_names]
        raise ValueError(
            f"Cyclic dependency detected among tasks: {sorted(remaining)!r}"
        )

    name_to_task = {t.name: t for t in tasks}
    return tuple(name_to_task[n] for n in sorted_names)


def infer_cross_stream_waits(
    schedule: Schedule,
) -> Dict[str, Tuple[str, ...]]:
    """Map each consumer to producer streams it must wait on.

    Slot reads match writers by slot name so cross-iter prefetch edges
    are included. Same-stream edges and unresolved reads are ignored
    here; validation handles malformed schedules before runtime.
    """
    tasks = _flatten_in_order(schedule)

    # Exact-slot indexing catches duplicate writers; name indexing
    # lets cross-iter readers wait on the stream that produced the
    # ring-rotated slot contents.
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
                    f"'{prior.name}' and '{task.name}'. Single writer "
                    f"per slot is required."
                )
            slot_exact_writer[slot] = task
            writers_by_slot_name.setdefault(slot.name, []).append(task)

    # Keep slot-name waits unambiguous. If a future use case needs
    # one slot name written by multiple streams, add an explicit
    # per-offset matcher instead of guessing here.
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
        producer_streams: Set[str] = set()

        for read_slot in consumer.reads:
            for writer in writers_by_slot_name.get(read_slot.name, ()):
                if writer.stream != consumer.stream:
                    producer_streams.add(writer.stream)

        for dep_name in consumer.depends_on:
            producer = name_to_task.get(dep_name)
            if producer is None:
                continue
            if producer.stream != consumer.stream:
                producer_streams.add(producer.stream)

        if same_progress_sync_uses_gpu(consumer):
            for dep_name in getattr(consumer, "same_progress_sync", ()):
                producer = name_to_task.get(dep_name)
                if producer is None:
                    continue
                if producer.stream != consumer.stream:
                    producer_streams.add(producer.stream)

        if producer_streams:
            waits[consumer.name] = tuple(sorted(producer_streams))

    return waits


def infer_cross_stream_event_deps(
    schedule: Schedule,
) -> Dict[str, Tuple[Tuple[str, str, int], ...]]:
    """Map each consumer to fine-grained cross-stream event waits.

    Each triple is ``(producer_task_name, producer_stream,
    slot_offset_at_consumer)``. The executor waits on the producer's
    recorded event in that ring slot, falling back to ``wait_stream``
    while the pipeline warms up. Offsets are from the consumer's view
    because BatchRing rotates producer events forward between progress
    calls.
    """
    tasks = _flatten_in_order(schedule)
    name_to_task: Dict[str, Task] = {t.name: t for t in tasks}

    # Event lookup for reads follows the same slot-name matching as
    # stream-level wait inference.
    writers_by_slot_name: Dict[str, List[Task]] = {}
    for task in tasks:
        for slot in task.writes:
            writers_by_slot_name.setdefault(slot.name, []).append(task)

    deps: Dict[str, Tuple[Tuple[str, str, int], ...]] = {}
    for consumer in tasks:
        triples: List[Tuple[str, str, int]] = []
        seen: Set[Tuple[str, str, int]] = set()

        def add_triple(producer: Task, slot_offset: int) -> None:
            key = (producer.name, producer.stream, slot_offset)
            if key not in seen:
                seen.add(key)
                triples.append(key)

        for read_slot in consumer.reads:
            for writer in writers_by_slot_name.get(read_slot.name, ()):
                if writer.stream == consumer.stream:
                    continue
                add_triple(writer, read_slot.batch_offset)

        for dep_name in consumer.depends_on:
            producer = name_to_task.get(dep_name)
            if producer is None:
                continue
            # Same-batch deps read the producer's event after ring
            # rotation has brought it to the consumer's offset.
            diff = producer.batch_offset - consumer.batch_offset
            if diff < 0:
                raise ValueError(
                    f"Task {consumer.name!r} depends_on=({dep_name!r}) "
                    f"but {dep_name!r}.lookahead={producer.batch_offset} "
                    f"< {consumer.name!r}.lookahead={consumer.batch_offset}. "
                    f"Cannot wait for a producer that has not yet run "
                    f"by the consumer's iteration (future-read)."
                )
            if producer.stream == consumer.stream:
                # Same-stream ordering is implicit via CUDA stream
                # FIFO. No explicit edge needed.
                continue
            add_triple(producer, consumer.batch_offset)

        # Cross-iter pure-control dependency: wait for a producer's
        # event from N progress calls earlier.
        for dep_name, neg_offset in getattr(consumer, "cross_iter_depends_on", ()):
            producer = name_to_task.get(dep_name)
            if producer is None:
                continue

            # Reject explicit control deps that duplicate a data edge.
            for read_slot in consumer.reads:
                for writer in writers_by_slot_name.get(read_slot.name, ()):
                    if writer.name != producer.name:
                        continue
                    implicit_n = writer.batch_offset - read_slot.batch_offset
                    if implicit_n == -neg_offset:
                        raise ValueError(
                            f"Task {consumer.name!r} declares a cross-iter "
                            f"dependency that duplicates an implicit data edge: "
                            f"cross_iter_depends_on=({dep_name!r}, "
                            f"{neg_offset}) but the same edge is already "
                            f"inferred from reads/writes: slot "
                            f"{read_slot.name!r} written by "
                            f"{producer.name!r} at batch_offset="
                            f"{writer.batch_offset}, read at batch_offset="
                            f"{read_slot.batch_offset} (implicit cross-iter "
                            f"diff={implicit_n}). Drop the explicit "
                            f"cross-iter declaration; redundant "
                            f"cross-iter restatements of data edges "
                            f"are rejected."
                        )

            # delta < 0 is a future-read, delta == 0 is same-progress,
            # and delta > 0 is a genuine cross-progress wait.
            N = -neg_offset  # neg_offset < 0, so N > 0
            delta = producer.batch_offset + N - consumer.batch_offset
            if delta == 0:
                if producer.stream == consumer.stream:
                    continue
                add_triple(producer, producer.batch_offset)
                continue
            if delta < 0:
                raise ValueError(
                    f"Task {consumer.name!r} declares cross_iter_depends_on="
                    f"({dep_name!r}, {neg_offset}) but "
                    f"consumer.lookahead={consumer.batch_offset} > "
                    f"producer.lookahead={producer.batch_offset} + |neg_offset|={N} "
                    f"= {producer.batch_offset + N}. The producer has not yet "
                    f"processed batch K-N by the consumer's progress "
                    f"(future-read across iterations). Either reduce "
                    f"consumer.lookahead, increase |neg_offset|, or pick "
                    f"a higher-lookahead producer."
                )
            if producer.stream == consumer.stream:
                continue
            # Cross-stream: emit a wait_event triple at slot_offset.
            slot_offset = consumer.batch_offset + neg_offset  # neg_offset < 0
            if slot_offset < 0:
                # The producer's event would already have been rotated
                # out of the ring by the time this consumer runs.
                # Surface at construction time rather than letting the
                # consumer silently miss the dependency.
                raise ValueError(
                    f"Task {consumer.name!r} declares cross_iter_depends_on="
                    f"({dep_name!r}, {neg_offset}), but consumer's ring offset "
                    f"after N advances would be {slot_offset} (= "
                    f"consumer.batch_offset {consumer.batch_offset} + "
                    f"{neg_offset}). The event has rotated out of the ring. "
                    f"Either reduce |N|, increase the consumer's lookahead, "
                    f"or rely on same-stream FIFO ordering between iterations."
                )
            add_triple(producer, slot_offset)

        # ``same_progress_sync=("X", ...)``: optional same-progress GPU
        # coherency wait.
        if same_progress_sync_uses_gpu(consumer):
            for dep_name in getattr(consumer, "same_progress_sync", ()):
                producer = name_to_task.get(dep_name)
                if producer is None:
                    continue
                if producer.stream == consumer.stream:
                    # Same-stream FIFO already serializes — no explicit
                    # wait_event needed.
                    continue
                add_triple(producer, producer.batch_offset)

        if triples:
            # Stable order for reproducibility / deterministic logs.
            deps[consumer.name] = tuple(sorted(triples))

    return deps


def producers_with_cross_stream_consumers(schedule: Schedule) -> set:
    """Set of producer task names that some cross-stream consumer
    waits on via ``infer_cross_stream_event_deps``.

    The executor only needs to call ``cudaEventRecord`` on producers
    whose event will be looked up later. Tasks with no cross-stream
    consumer (in particular, every same-stream-only task) can skip the
    record entirely — same-stream FIFO already orders work, and no
    other stream waits on the event.

    This drives the optional ``producers_to_record`` filter on
    ``_record_completion_event``.
    """
    deps = infer_cross_stream_event_deps(schedule)
    out: set = set()
    for triples in deps.values():
        for producer_name, _producer_stream, _slot_offset in triples:
            out.add(producer_name)
    return out
