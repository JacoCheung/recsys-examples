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

from typing import Dict, List, Set, Tuple

from .schedule import Schedule
from .task import DataSlot, Task

__all__ = [
    "infer_cross_stream_waits",
    "infer_cross_stream_event_deps",
    "topological_sort",
]


def _flatten_in_order(schedule: Schedule) -> Tuple[Task, ...]:
    """Every task in declaration order (stage order × within-stage order)."""
    return schedule.all_tasks()


def _build_same_progress_dag_edges(
    tasks: Tuple[Task, ...],
) -> Dict[str, Set[str]]:
    """Build the within-progress dependency DAG.

    Returns a mapping ``consumer_name -> set of producer names`` for
    edges that constrain execution order **within a single progress()
    call**. Edges come from:

      * ``reads/writes`` exact-slot match — same offset means same
        batch processed in the current progress, producer must run
        before consumer.
      * ``depends_on`` bare-name with ``producer.lookahead ==
        consumer.lookahead`` — same-batch logical dep where producer
        and consumer process the same batch in the current progress.
      * ``same_progress_sync`` — by definition a same-progress GPU
        coherency wait, so producer must run before consumer.
      * ``cross_iter_depends_on=((X, -N),)`` with Δ=0
        (``X.lookahead + N == consumer.lookahead``) — degenerate
        same-progress case; the producer and consumer both run in
        the current progress on different batches, but the wait
        semantics are identical to ``same_progress_sync``. Auto-promoted
        here so the user can keep the batch-flow API.

    Excluded (no current-progress edge needed):

      * ``reads/writes`` name match across different offsets — handled
        by BatchRing slot rotation; producer's data was written in an
        earlier progress.
      * ``depends_on`` bare-name with ``producer.lookahead >
        consumer.lookahead`` — producer processed the same batch in
        an earlier progress.
      * ``cross_iter_depends_on`` with Δ ≥ 1 — genuinely cross-progress
        dep; mediated by ring rotation, producer ran in an earlier
        progress.

    Same-stream FIFO is intentionally NOT used as an edge source: if
    two same-stream tasks have no logical dep between them, the engine
    leaves their relative execution order unconstrained (CUDA stream
    FIFO will serialize their GPU ops in submission order, but
    correctness shouldn't depend on declaration order).
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

    # ``same_progress_sync`` (current-iter GPU coherency) — always
    # emits a current-progress edge regardless of lookahead, since by
    # definition both tasks are in the current progress.
    for task in tasks:
        for dep_name in getattr(task, "same_progress_sync", ()):
            producer = name_to_task.get(dep_name)
            if producer is None or producer.name == task.name:
                continue
            incoming[task.name].add(producer.name)

    # ``cross_iter_depends_on=((X, -N),)`` with Δ=0 — auto-promoted to
    # the same_progress_sync semantic. Δ = X.la + N - consumer.la == 0
    # means producer ran in the *current* progress (no ring rotation
    # between record and read), so producer must fire before consumer
    # in this progress just like same_progress_sync does. Δ ≥ 1
    # genuinely cross-progress entries are handled by ring rotation
    # in infer_cross_stream_event_deps and contribute no topo edge.
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


def topological_sort(schedule: Schedule) -> Tuple[Task, ...]:
    """Return tasks ordered by within-progress DAG topological sort.

    Replaces "declaration order" as the engine's execution-order
    source. Edges are computed by ``_build_same_progress_dag_edges``;
    tie-breaks among DAG-independent tasks fall back to declaration
    order so author intent is preserved when the DAG is silent.

    Raises ``ValueError`` if the DAG contains a cycle.
    """
    tasks = _flatten_in_order(schedule)
    incoming = _build_same_progress_dag_edges(tasks)

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
    """Per-consumer list of fine-grained cross-stream waits, keyed for
    event-based sync.

    Returns a mapping ``consumer_name → tuple of (producer_task_name,
    producer_stream, slot_offset_at_consumer)``. The executor uses each
    triple to:

      1. look up the slot at ``slot_offset_at_consumer`` in the ring;
      2. fetch the producer's completion event from that slot's event
         registry (keyed by ``producer_task_name``);
      3. emit ``consumer_stream.wait_event(producer_event)`` — fine-
         grained sync, blocks only on that specific producer's task,
         not on the entire producer stream;
      4. if the slot's event registry has no entry for that producer
         yet (first iteration or pipeline still warming up), fall back
         to ``consumer_stream.wait_stream(producer_stream)``.

    The triples are keyed at ``slot_offset_at_consumer`` (not the
    writer's offset) because the engine ring advances each iteration:
    a producer that writes ``X@k1`` records its completion event onto
    the slot store at offset ``k1``; after ``k1 - k2`` advances, that
    slot store sits at offset ``k2`` and the consumer reading ``X@k2``
    finds the event there.

    For ``depends_on`` edges (no slot involved), the offset defaults to
    the consumer's own ``batch_offset`` — same-iter ordering is
    co-located there. This still allows event-based sync between two
    same-iter cross-stream tasks (e.g. ``backward.depends_on=
    ("prefetch_embeddings",)`` in the HSTU prefetch variant).

    Same-stream edges still emit no entry — within a single CUDA stream
    submission order on that stream gives serial execution.
    """
    tasks = _flatten_in_order(schedule)
    name_to_task: Dict[str, Task] = {t.name: t for t in tasks}

    # Index writers by exact (slot.name, slot.batch_offset) and by name
    # — same as infer_cross_stream_waits.
    writers_by_slot_name: Dict[str, List[Task]] = {}
    for task in tasks:
        for slot in task.writes:
            writers_by_slot_name.setdefault(slot.name, []).append(task)

    deps: Dict[str, Tuple[Tuple[str, str, int], ...]] = {}
    for consumer in tasks:
        triples: List[Tuple[str, str, int]] = []
        seen: set = set()

        for read_slot in consumer.reads:
            for writer in writers_by_slot_name.get(read_slot.name, ()):
                if writer.stream == consumer.stream:
                    continue
                key = (writer.name, writer.stream, read_slot.batch_offset)
                if key not in seen:
                    seen.add(key)
                    triples.append(key)

        for dep_name in consumer.depends_on:
            producer = name_to_task.get(dep_name)
            if producer is None:
                continue
            # Bare-name ``depends_on=("X",)`` expresses a **same-batch
            # logical dependency** from the user's authoring POV: "this
            # task processing batch K must wait for X to have processed
            # batch K". Engine maps this to a ring-rotated wait_event:
            # X recorded its event in slot offset=X.batch_offset in the
            # progress() call where X processed this batch K; after
            # (X.la - consumer.la) ring advances, that slot has rotated
            # to offset=consumer.batch_offset. That's where the
            # consumer looks up the event.
            #
            # Future-read (producer has smaller lookahead than the
            # consumer) is rejected — the user's authoring graph
            # cannot have a same-batch edge from a stage that has not
            # yet run.
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
            # The producer recorded its completion event onto the ring
            # slot at offset=producer.batch_offset; after ``diff`` ring
            # advances that slot has rotated to
            # offset=consumer.batch_offset. That is where the executor
            # looks up the event. (For same-lookahead, diff=0 and the
            # event is already at consumer.batch_offset.)
            key = (producer.name, producer.stream, consumer.batch_offset)
            if key not in seen:
                seen.add(key)
                triples.append(key)

        # SPEC_p4 v2 §6 step 8: cross-iter pure-control depends_on.
        # ``cross_iter_depends_on=((producer, -N), ...)`` declares
        # "wait for producer's output from N iterations earlier". The
        # producer recorded its completion event onto the ring slot at
        # offset ``producer.batch_offset``; after ``N`` ring advances
        # the slot now sits at offset ``producer.batch_offset - N``.
        # That is where the executor must look up the event.
        for dep_name, neg_offset in getattr(consumer, "cross_iter_depends_on", ()):
            producer = name_to_task.get(dep_name)
            if producer is None:
                continue

            # SPEC_p4 v2 §5: cross-iter depends_on restating a data edge
            # already inferred from reads/writes is rejected as
            # redundant. The implicit cross-iter offset for a data edge
            # is ``writer.batch_offset - reader.read_slot.batch_offset``
            # (= number of ring advances between produce and consume).
            # If that equals |neg_offset| for the same producer, the
            # explicit declaration adds nothing.
            for read_slot in consumer.reads:
                for writer in writers_by_slot_name.get(read_slot.name, ()):
                    if writer.name != producer.name:
                        continue
                    implicit_n = writer.batch_offset - read_slot.batch_offset
                    if implicit_n == -neg_offset:
                        raise ValueError(
                            f"Task {consumer.name!r} declares "
                            f"cross_iter_depends_on=({dep_name!r}, "
                            f"{neg_offset}) but the same edge is already "
                            f"inferred from reads/writes: slot "
                            f"{read_slot.name!r} written by "
                            f"{producer.name!r} at batch_offset="
                            f"{writer.batch_offset}, read at batch_offset="
                            f"{read_slot.batch_offset} (implicit cross-iter "
                            f"diff={implicit_n}). Drop the explicit "
                            f"cross-iter declaration — SPEC_p4 v2 §5 "
                            f"rejects redundant cross-iter restatements "
                            f"of data edges."
                        )

            # Slot lookup uses CONSUMER's batch_offset, not producer's,
            # in the genuinely cross-progress case (Δ ≥ 1). User intent:
            # "consumer processing batch K waits for producer to have
            # processed batch K-N". Working through ring rotation:
            # producer wrote slot offset=producer.la in its progress;
            # consumer reads in its own progress; the progress diff
            # Delta = producer.la + N - consumer.la, so the slot has
            # rotated to offset=consumer.la-N by then. Hence
            # consumer.batch_offset + neg_offset.
            #
            # Three Δ regimes:
            #
            #   (Δ < 0) future-read — producer hasn't processed batch
            #       K-N by the consumer's progress. Reject.
            #   (Δ = 0) degenerate same-progress — auto-promoted to
            #       the same_progress_sync semantic: emit at
            #       slot[producer.la] (where producer just recorded
            #       this progress's event), and rely on the topo-DAG
            #       edge added in _build_same_progress_dag_edges to
            #       guarantee producer-before-consumer ordering.
            #       Per SPEC_cross_iter_delta0_autoconvert.md.
            #   (Δ ≥ 1) genuine cross-progress — slot_offset =
            #       consumer.la - N must be ≥ 0 (event still in ring).
            N = -neg_offset  # neg_offset < 0, so N > 0
            delta = producer.batch_offset + N - consumer.batch_offset
            if delta == 0:
                # Auto-promoted same-progress wait. Same emission as
                # the same_progress_sync handler below: slot index =
                # producer.batch_offset (where producer recorded its
                # event in *this* progress). Same-stream short-circuits
                # without emitting a triple — CUDA stream FIFO orders
                # the submissions, and the topo edge in
                # _build_same_progress_dag_edges ensures producer is
                # submitted first.
                if producer.stream == consumer.stream:
                    continue
                key = (producer.name, producer.stream, producer.batch_offset)
                if key not in seen:
                    seen.add(key)
                    triples.append(key)
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
            # Δ ≥ 1 from here — genuine cross-progress relationship.
            if producer.stream == consumer.stream:
                # Same-stream cross-iter: stream FIFO orders all
                # submissions on the single stream. With Δ ≥ 1, the
                # producer ran in an earlier progress, so by the time
                # consumer's progress submits its work the producer's
                # work has already been committed to the stream. No
                # explicit wait_event triple needed (and slot_offset
                # might fall outside the ring without harm — the
                # event isn't being looked up).
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
            key = (producer.name, producer.stream, slot_offset)
            if key not in seen:
                seen.add(key)
                triples.append(key)

        # ``same_progress_sync=("X", ...)``: same-progress GPU coherency wait.
        # Not a logical data-flow edge — used when consumer and
        # producer process different batches in the same progress
        # call but still need stream-level synchronization (e.g.
        # backward reading dynamicemb cache that prefetch is writing
        # for the next batch).
        #
        # Engine emits wait_event at slot offset=producer.batch_offset
        # (producer's OWN slot, where it just recorded its event in
        # this same progress call) — NOT at consumer.batch_offset
        # (which would be a ring-rotated stale event).
        for dep_name in getattr(consumer, "same_progress_sync", ()):
            producer = name_to_task.get(dep_name)
            if producer is None:
                continue
            if producer.stream == consumer.stream:
                # Same-stream FIFO already serializes — no explicit
                # wait_event needed.
                continue
            key = (producer.name, producer.stream, producer.batch_offset)
            if key not in seen:
                seen.add(key)
                triples.append(key)

        if triples:
            # Stable order for reproducibility / deterministic logs.
            deps[consumer.name] = tuple(sorted(triples))

    return deps
