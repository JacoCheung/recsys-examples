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

"""Task execution strategies for SchedulablePipeline.

SequentialExecutor: single-threaded, declaration-order execution (default).
ThreadedExecutor: task-level multi-threaded execution with pluggable
    thread mapping.

Key design: **threads and CUDA streams are decoupled.** A task's
``stream`` attribute decides which CUDA stream context it runs in; the
``thread_map`` decides which CPU worker thread submits it. The same
stream's tasks may run on different threads (if the DAG allows), and
different streams' tasks may share a thread.

Thread mapping strategies (``thread_map`` parameter):

  ``"by_stream"`` (default)
      Group tasks by ``task.stream``. Same-stream tasks share a thread.
      Good default that avoids most cross-thread sync overhead.

  ``"per_task"``
      Every task gets its own thread. Maximum parallelism; useful when
      tasks have heavy CPU-side work.

  ``dict[str, str]``
      Explicit mapping ``{task_name: thread_id}``. Tasks not in the dict
      fall back to ``"default"`` thread.

  ``Callable[[Task], str]``
      Arbitrary function mapping each task to a thread id string.
"""

import contextlib
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Dict, Iterator, List, Optional, Tuple, Union

import torch

from .context import TaskContext
from .schedule import Stage
from .streams import StreamPool
from .task import DataSlot, Task

# NVTX is optional — used only for profiler annotation. The engine
# stays framework-agnostic w.r.t. nvtx by treating its absence as a
# no-op (CPU-only test hosts may not have it installed).
try:
    import nvtx as _nvtx
except ImportError:  # pragma: no cover - nvtx absence
    _nvtx = None

__all__ = ["SequentialExecutor", "ThreadedExecutor"]

# Type alias for the thread_map parameter
ThreadMap = Union[str, Dict[str, str], Callable[[Task], str], None]


def _apply_cross_stream_waits(
    task: Task,
    cross_stream_waits: Dict[str, Tuple[str, ...]],
    stream_pool: StreamPool,
    *,
    event_deps: Optional[Dict[str, Tuple[Tuple[str, str, int], ...]]] = None,
    ctx: Optional[TaskContext] = None,
) -> None:
    """Apply GPU-side cross-stream waits before a task runs.

    Prefers fine-grained ``wait_event(producer_event)`` (event-based
    sync) when ``event_deps`` and ``ctx`` are provided AND the producer
    has already recorded its completion event onto the ring slot for
    this iteration. Falls back to ``wait_stream(producer_stream)`` for
    edges that have no event yet — typically iteration 1 before the
    ring is fully primed, or producers that did not run this iteration.

    The fallback uses ``cross_stream_waits`` (stream-name list, computed
    by ``deps.infer_cross_stream_waits``). The fine-grained mode uses
    ``event_deps`` (triples of producer task / producer stream / slot
    offset, computed by ``deps.infer_cross_stream_event_deps``).
    """
    anchor = stream_pool.anchor_device
    if anchor is None:
        return

    consumer = torch.cuda.current_stream()

    if event_deps is not None and ctx is not None:
        # Fine-grained path. For each producer triple, prefer
        # wait_event over wait_stream when the event is on the slot.
        triples = event_deps.get(task.name, ())
        for producer_name, producer_stream, slot_offset in triples:
            slot = ctx.slots_at(slot_offset)
            event = slot.get_event(producer_name)
            if event is not None:
                consumer.wait_event(event)
            else:
                # Fallback: ring slot has no event for this producer
                # yet (first-iter / not-run-this-iter). Use the
                # coarser stream-level wait so we don't drop a
                # required ordering edge.
                prod = stream_pool.get(producer_stream)
                if prod is None:
                    prod = torch.cuda.default_stream(anchor)
                consumer.wait_stream(prod)
        return

    # Legacy path: stream-list only, no event lookup.
    waits = cross_stream_waits.get(task.name, ())
    if waits:
        for producer_stream in waits:
            prod = stream_pool.get(producer_stream)
            if prod is None:
                prod = torch.cuda.default_stream(anchor)
            consumer.wait_stream(prod)


@contextlib.contextmanager
def _nvtx_range(task: Task) -> Iterator[None]:
    """Wrap a task's execution in an NVTX range labelled by
    ``task.nvtx_tag`` (falling back to ``task.name`` when the tag is
    unset). No-op when nvtx is not importable or when CUDA is not
    available — the range would have no profiler to record into.
    """
    if _nvtx is None or not torch.cuda.is_available():
        yield
        return
    tag = task.nvtx_tag or task.name
    with _nvtx.annotate(tag):
        yield


def _record_completion_event(task: Task, ctx: TaskContext) -> None:
    """Record a CUDA event on the task's current stream and store it on
    the ring slot at ``task.batch_offset``, keyed by ``task.name``.

    Called after the task body returns successfully, while still inside
    the ``stream_pool.use(task.stream)`` context — so
    ``torch.cuda.current_stream()`` is the task's stream.

    The event object is **reused across iterations**: SlotStore preserves
    its event registry across ``BatchRing.advance()`` (the slot rotates
    in place), so the same ``torch.cuda.Event`` is re-recorded each iter
    that this task runs at this offset. ``Event.record()`` overwriting
    the previous record is the intended semantics — see SPEC §4.2 rule 8
    (event-based cross-stream sync).

    No-op on CPU-only runs (no CUDA available).
    """
    if not torch.cuda.is_available():
        return
    slot = ctx.slots_at(task.batch_offset)
    event = slot.get_event(task.name)
    if event is None:
        event = torch.cuda.Event()
        slot.set_event(task.name, event)
    event.record(torch.cuda.current_stream())


def _resolve_thread_id(task: Task, thread_map: ThreadMap) -> str:
    """Map a task to its thread id string."""
    if thread_map is None or thread_map == "by_stream":
        return task.stream or "default"
    if thread_map == "per_task":
        return task.name
    if callable(thread_map):
        return thread_map(task)
    if isinstance(thread_map, dict):
        return thread_map.get(task.name, "default")
    raise ValueError(
        f"thread_map must be 'by_stream', 'per_task', a dict, or a "
        f"callable, got {type(thread_map).__name__}"
    )


def _compute_cpu_deps(
    active: List[Task],
    thread_id_of: Dict[str, str],
    completion: Dict[str, threading.Event],
) -> Dict[str, List[threading.Event]]:
    """Compute CPU-side dependency events between tasks on different threads.

    Uses the task DAG (reads/writes + depends_on) to find cross-thread
    edges that need CPU-side ordering.  Same-thread edges are handled by
    sequential execution within the thread.

    Also: when ``thread_map`` splits same-stream tasks across worker
    threads (e.g. ``thread_map="per_task"``), CUDA stream FIFO order
    is determined by host-thread enqueue race rather than declaration
    order. We add a CPU dep from each same-stream predecessor (declared
    earlier) on a different thread, so submission order matches the
    schedule.

    GPU-side cross-stream waits are separate and handled by
    ``_apply_cross_stream_waits``.
    """
    # Build writer map for CPU-side ordering. Unlike GPU stream-wait
    # inference, host thread events should only model dependencies
    # within the same logical slot. A read of X@0 consumes data that
    # was produced in an earlier internal iteration, not the active
    # stage's producer of X@2.
    writers: Dict[DataSlot, str] = {}
    for task in active:
        for slot in task.writes:
            writers[slot] = task.name

    active_names = {t.name for t in active}
    cpu_deps: Dict[str, List[threading.Event]] = defaultdict(list)
    # Track the most recent task on each stream in declaration order.
    last_on_stream: Dict[str, str] = {}

    for task in active:
        my_thread = thread_id_of[task.name]
        my_stream = task.stream or "default"
        dep_names: set = set()

        # Slot-based deps: any active writer of the exact slot I read.
        for slot in task.reads:
            writer_name = writers.get(slot)
            if writer_name and writer_name != task.name:
                dep_names.add(writer_name)

        # ``depends_on`` (same-batch logical) edges — engine emits
        # ring-rotated GPU wait_event; CPU-side, the consumer thread
        # still needs the producer's host enqueue to be complete in
        # this same progress() call before continuing. (For same-
        # lookahead producer/consumer, this is straightforward;
        # for cross-lookahead, the producer's host enqueue happens
        # earlier in this progress for the consumer's batch K.)
        for dep_name in task.depends_on or ():
            if dep_name in active_names:
                dep_names.add(dep_name)

        # ``same_progress_sync`` (same-progress GPU coherency) edges —
        # consumer waits for the producer's current-iter completion
        # event, regardless of which batch each is processing. Same
        # CPU-side mechanism as ``depends_on``: cross-thread emits a
        # threading.Event wait so the producer's host enqueue +
        # event.record() happen before the consumer's wait_event is
        # enqueued.
        for dep_name in getattr(task, "same_progress_sync", ()):
            if dep_name in active_names:
                dep_names.add(dep_name)

        # ``cross_iter_depends_on`` with Δ=0 — auto-promoted to the
        # same_progress_sync mechanical contract (per
        # SPEC_cross_iter_delta0_autoconvert.md). When
        # producer.la + N == consumer.la, the producer ran in the
        # current progress on a different batch, identical wait
        # semantics to same_progress_sync. Emit the same CPU edge.
        # Δ ≥ 1 entries contribute no CPU edge here — those are
        # handled by ring-rotated wait_event in
        # ``_apply_cross_stream_waits``.
        for dep_name, neg_offset in getattr(task, "cross_iter_depends_on", ()):
            if dep_name not in active_names:
                continue
            producer_task = next((t for t in active if t.name == dep_name), None)
            if producer_task is None:
                continue
            N = -neg_offset
            delta = producer_task.batch_offset + N - task.batch_offset
            if delta == 0:
                dep_names.add(dep_name)

        # Same-stream predecessor (declaration order) — preserves
        # CUDA stream FIFO when thread_map splits same-stream tasks.
        # No-op when thread_map keeps them on the same thread (the
        # cross-thread filter below skips it).
        prev_same_stream = last_on_stream.get(my_stream)
        if prev_same_stream is not None:
            dep_names.add(prev_same_stream)
        last_on_stream[my_stream] = task.name

        # Only need CPU event for deps on DIFFERENT threads
        for dep_name in dep_names:
            if thread_id_of.get(dep_name) != my_thread:
                cpu_deps[task.name].append(completion[dep_name])

    return cpu_deps


# ------------------------------------------------------------------
# Sequential (Problem #1 default)
# ------------------------------------------------------------------


class SequentialExecutor:
    """Single-threaded task execution — identical to Problem #1 behavior."""

    def execute_stage(
        self,
        stage: Stage,
        ctx: TaskContext,
        iter_count: int,
        should_run: Callable[[Task], bool],
        cross_stream_waits: Dict[str, Tuple[str, ...]],
        stream_pool: StreamPool,
        *,
        event_deps: Optional[Dict[str, Tuple[Tuple[str, str, int], ...]]] = None,
    ) -> None:
        for task in stage.tasks:
            if not should_run(task):
                continue
            ctx._active_offset = task.batch_offset
            ctx.iter_count = iter_count
            with stream_pool.use(task.stream):
                _apply_cross_stream_waits(
                    task,
                    cross_stream_waits,
                    stream_pool,
                    event_deps=event_deps,
                    ctx=ctx,
                )
                with _nvtx_range(task):
                    task.run(ctx)
                _record_completion_event(task, ctx)

    def shutdown(self) -> None:
        """No-op for sequential executor."""


# ------------------------------------------------------------------
# NCCL ordered lock
# ------------------------------------------------------------------


class _NcclOrderedLock:
    """Ensures NCCL-tagged tasks execute in deterministic (ticket) order.

    Failure-aware: if any NCCL task fails, later tickets abort
    immediately instead of running their collective.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._next_ticket: int = 0
        self._failed: bool = False

    def acquire(self, ticket: int) -> None:
        with self._cond:
            while self._next_ticket != ticket and not self._failed:
                self._cond.wait()
            if self._failed:
                raise RuntimeError(
                    "NCCL ordered lock aborted: a prior NCCL task failed"
                )

    def release(self, failed: bool = False) -> None:
        with self._cond:
            if failed:
                self._failed = True
            self._next_ticket += 1
            self._cond.notify_all()

    def abort(self) -> None:
        """Wake up every waiter and force them to raise.

        Used when a non-NCCL task on another thread fails before its
        expected NCCL ticket has been released — without this, a worker
        already inside ``acquire(later_ticket)`` would block forever
        because no one will ever call ``release()`` for the missing
        earlier ticket. The cancellation flag in the executor only
        prevents NEW acquires; this wakes the existing wait().
        """
        with self._cond:
            self._failed = True
            self._cond.notify_all()

    def reset(self) -> None:
        with self._cond:
            self._next_ticket = 0
            self._failed = False


# ------------------------------------------------------------------
# Threaded (Problem #3)
# ------------------------------------------------------------------


class ThreadedExecutor:
    """Multi-threaded task executor with decoupled thread/stream mapping.

    Tasks are grouped by ``thread_map`` into thread chains. Each chain
    runs sequentially on one worker thread. Chains with no mutual
    dependencies run concurrently.

    CUDA stream context (``stream_pool.use(task.stream)``) is entered
    per-task, independent of thread assignment. This means the same
    CUDA stream can be used from different threads, and tasks on the
    same thread can target different CUDA streams.

    Parameters
    ----------
    thread_map : str | dict | callable, optional
        How to assign tasks to threads. See module docstring for details.
        Default: ``"by_stream"`` (group by task.stream).
    max_workers : int, optional
        Maximum worker threads. Auto-sized to the number of active
        thread groups if not specified.
    """

    def __init__(
        self,
        thread_map: ThreadMap = None,
        max_workers: Optional[int] = None,
    ) -> None:
        self._thread_map = thread_map
        self._max_workers = max_workers
        self._pool: Optional[ThreadPoolExecutor] = None
        self._pool_size: int = 0
        self._nccl_lock = _NcclOrderedLock()

    def _ensure_pool(self, n_threads: int) -> ThreadPoolExecutor:
        needed = self._max_workers or max(n_threads, 1)
        needed = max(needed, n_threads)
        if self._pool is not None and self._pool_size >= needed:
            return self._pool
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self._pool = ThreadPoolExecutor(
            max_workers=needed,
            thread_name_prefix="engine",
        )
        self._pool_size = needed
        return self._pool

    def execute_stage(
        self,
        stage: Stage,
        ctx: TaskContext,
        iter_count: int,
        should_run: Callable[[Task], bool],
        cross_stream_waits: Dict[str, Tuple[str, ...]],
        stream_pool: StreamPool,
        *,
        event_deps: Optional[Dict[str, Tuple[Tuple[str, str, int], ...]]] = None,
    ) -> None:
        active: List[Task] = [t for t in stage.tasks if should_run(t)]
        if not active:
            return

        # Single-task fast path
        if len(active) == 1:
            task = active[0]
            ctx._active_offset = task.batch_offset
            ctx.iter_count = iter_count
            with stream_pool.use(task.stream):
                _apply_cross_stream_waits(
                    task,
                    cross_stream_waits,
                    stream_pool,
                    event_deps=event_deps,
                    ctx=ctx,
                )
                with _nvtx_range(task):
                    task.run(ctx)
                _record_completion_event(task, ctx)
            return

        # Resolve thread assignment for each task
        thread_id_of: Dict[str, str] = {
            t.name: _resolve_thread_id(t, self._thread_map) for t in active
        }

        # Partition tasks by thread (preserving declaration order within)
        thread_to_tasks: Dict[str, List[Task]] = defaultdict(list)
        for task in active:
            thread_to_tasks[thread_id_of[task.name]].append(task)

        # All-same-thread fast path
        if len(thread_to_tasks) == 1:
            for task in active:
                ctx._active_offset = task.batch_offset
                ctx.iter_count = iter_count
                with stream_pool.use(task.stream):
                    _apply_cross_stream_waits(
                        task,
                        cross_stream_waits,
                        stream_pool,
                        event_deps=event_deps,
                        ctx=ctx,
                    )
                    with _nvtx_range(task):
                        task.run(ctx)
                    _record_completion_event(task, ctx)
            return

        pool = self._ensure_pool(len(thread_to_tasks))

        # Per-task completion events
        completion: Dict[str, threading.Event] = {
            t.name: threading.Event() for t in active
        }

        # CPU-side cross-thread dependency events (from DAG)
        cpu_deps = _compute_cpu_deps(active, thread_id_of, completion)

        # NCCL tickets (declaration order)
        nccl_tickets: Dict[str, int] = {}
        ticket = 0
        for task in active:
            if getattr(task, "nccl", False):
                nccl_tickets[task.name] = ticket
                ticket += 1
        self._nccl_lock.reset()

        # Cancellation + error collection
        cancelled = threading.Event()
        errors: List[BaseException] = []
        errors_lock = threading.Lock()

        def _run_thread_chain(tasks: List[Task]) -> None:
            try:
                for task in tasks:
                    if cancelled.is_set():
                        break

                    # Wait for cross-thread dependencies
                    for evt in cpu_deps.get(task.name, []):
                        evt.wait()
                        if cancelled.is_set():
                            break
                    if cancelled.is_set():
                        break

                    ctx._active_offset = task.batch_offset
                    ctx.iter_count = iter_count

                    with stream_pool.use(task.stream):
                        _apply_cross_stream_waits(
                            task,
                            cross_stream_waits,
                            stream_pool,
                            event_deps=event_deps,
                            ctx=ctx,
                        )
                        with _nvtx_range(task):
                            if task.name in nccl_tickets:
                                nccl_failed = False
                                self._nccl_lock.acquire(nccl_tickets[task.name])
                                try:
                                    task.run(ctx)
                                except BaseException:
                                    nccl_failed = True
                                    raise
                                finally:
                                    self._nccl_lock.release(failed=nccl_failed)
                            else:
                                task.run(ctx)
                        _record_completion_event(task, ctx)

                    completion[task.name].set()
            except BaseException as e:
                cancelled.set()
                # Wake any worker still blocked inside
                # _NcclOrderedLock.acquire() — without this, a thread
                # waiting for ticket=k can deadlock forever if the
                # thread that was supposed to release ticket=(k-1)
                # died before getting there.
                self._nccl_lock.abort()
                with errors_lock:
                    errors.append(e)
            finally:
                for t in tasks:
                    completion[t.name].set()

        # Submit each thread chain
        futures = []
        for thread_id, tasks in thread_to_tasks.items():
            futures.append(pool.submit(_run_thread_chain, tasks))

        # Stage barrier
        for f in futures:
            f.result()

        ctx.iter_count = iter_count

        if errors:
            raise errors[0]

    def shutdown(self) -> None:
        """Shut down the thread pool."""
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
            self._pool_size = 0
