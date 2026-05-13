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

"""Task execution strategies for ``SchedulablePipeline``.

Threads and CUDA streams are separate concerns: ``task.stream`` selects
the CUDA stream, while ``thread_map`` selects the CPU worker that submits
the task. ``ThreadedExecutor`` supports ``"by_stream"``, ``"per_task"``,
explicit dicts, and callables.
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
from .task import DataSlot, Task, same_progress_sync_uses_cpu

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

    Uses recorded producer events when available; falls back to
    stream-level waits during ring warmup or when a producer did not
    record an event for this slot.
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
                # Warmup/no-event fallback: keep the ordering edge.
                prod = stream_pool.get(producer_stream)
                if prod is None:
                    prod = torch.cuda.default_stream(anchor)
                consumer.wait_stream(prod)
        return

    # Stream-list fallback: no event lookup.
    waits = cross_stream_waits.get(task.name, ())
    if waits:
        for producer_stream in waits:
            prod = stream_pool.get(producer_stream)
            if prod is None:
                prod = torch.cuda.default_stream(anchor)
            consumer.wait_stream(prod)


@contextlib.contextmanager
def _nvtx_range(task: Task) -> Iterator[None]:
    """Wrap a task in an outer NVTX range when nvtx/CUDA are available."""
    if _nvtx is None or not torch.cuda.is_available():
        yield
        return
    tag = task.nvtx_tag or task.name
    with _nvtx.annotate(f"[engine] {tag}", color="orange"):
        yield


def _record_completion_event(
    task: Task,
    ctx: TaskContext,
    producers_to_record: Optional[set] = None,
) -> None:
    """Record a CUDA event on the task's current stream and store it on
    the task's ring slot.

    SlotStore keeps event objects across ring rotation, so re-recording
    the same ``torch.cuda.Event`` each iteration is intentional.
    """
    if not torch.cuda.is_available():
        return
    if producers_to_record is not None and task.name not in producers_to_record:
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

    Same-thread edges are handled by sequential execution. Cross-stream
    CUDA waits remain GPU-side; these events only order host submission
    when the DAG or same-stream declaration order crosses worker threads.
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

    task_by_name = {t.name: t for t in active}
    active_names = set(task_by_name.keys())
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
        # needs the producer's host enqueue to be complete in this
        # same progress() call before continuing.
        #
        # Cross-lookahead (producer.la > consumer.la) is filtered out
        # here, mirroring ``_build_same_progress_dag_edges``: the
        # producer's batch-K host work happened in an EARLIER
        # progress (event already on the ring slot via rotation), so
        # the current-progress producer enqueue is unrelated. Adding
        # this edge would also form a cycle with a reverse
        # ``same_progress_sync`` (which the topological-sort filter
        # makes legal), deadlocking the threaded executor.
        for dep_name in task.depends_on or ():
            if dep_name not in active_names:
                continue
            producer = task_by_name[dep_name]
            if producer.batch_offset > task.batch_offset:
                continue  # cross-la — handled by ring rotation, no CPU edge
            dep_names.add(dep_name)

        # CPU-side ``same_progress_sync`` edges. When enabled, the
        # consumer waits for the producer's current-progress host
        # enqueue + completion-event record before continuing. GPU-side
        # waits are inferred separately in deps.py.
        if same_progress_sync_uses_cpu(task):
            for dep_name in getattr(task, "same_progress_sync", ()):
                if dep_name in active_names:
                    dep_names.add(dep_name)

        # ``cross_iter_depends_on`` with Δ=0 has the same mechanical
        # contract as ``same_progress_sync``: the producer ran in the
        # current progress on a different batch, so emit the same CPU edge.
        # Δ ≥ 1 entries contribute no CPU edge here — those are
        # handled by ring-rotated wait_event in
        # ``_apply_cross_stream_waits``.
        for dep_name, neg_offset in getattr(task, "cross_iter_depends_on", ()):
            if dep_name not in active_names:
                continue
            producer_task = task_by_name[dep_name]
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
# Sequential
# ------------------------------------------------------------------


class SequentialExecutor:
    """Single-threaded task execution."""

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
        producers_to_record: Optional[set] = None,
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
                _record_completion_event(task, ctx, producers_to_record)

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
# Threaded
# ------------------------------------------------------------------


class ThreadedExecutor:
    """Multi-threaded task executor with decoupled thread/stream mapping.

    Tasks are grouped by ``thread_map`` into thread chains. Each chain
    runs sequentially on one worker thread. Chains with no mutual
    dependencies run concurrently. ``inline_thread`` can keep one chain
    on the caller thread for profiler/NVTX attribution.
    """

    def __init__(
        self,
        thread_map: ThreadMap = None,
        max_workers: Optional[int] = None,
        inline_thread: Optional[str] = "compute",
    ) -> None:
        self._thread_map = thread_map
        self._max_workers = max_workers
        self._inline_thread = inline_thread
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
        producers_to_record: Optional[set] = None,
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
                _record_completion_event(task, ctx, producers_to_record)
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
                    _record_completion_event(task, ctx, producers_to_record)
            return

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
                        _record_completion_event(task, ctx, producers_to_record)

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

        # Inline-thread split: pop the chain whose name matches
        # ``inline_thread`` (default "compute") and run it on the
        # calling thread, so its kernels and NVTX ranges attach to the
        # caller for nsys's NVTX→GPU-stream projection. The remaining
        # chains run in the pool concurrently with the inline run.
        inline_chain: Optional[List[Task]] = None
        if self._inline_thread is not None and self._inline_thread in thread_to_tasks:
            inline_chain = thread_to_tasks.pop(self._inline_thread)

        # Submit non-inlined thread chains to the pool first so they
        # start running concurrently with the inline chain.
        futures = []
        if thread_to_tasks:
            pool = self._ensure_pool(len(thread_to_tasks))
            for thread_id, tasks in thread_to_tasks.items():
                futures.append(pool.submit(_run_thread_chain, tasks))

        # Run the inlined chain on the calling thread. ``cpu_deps`` /
        # ``completion`` / ``_nccl_lock`` are shared closures, so cross-
        # thread ordering still holds.
        if inline_chain is not None:
            _run_thread_chain(inline_chain)

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
