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

"""SchedulablePipeline: the driver that consumes a Schedule + StreamPool.

    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool)
    for batch in loader:
        result = pipe.progress(iter([batch]))

Returns the value written to the `"step_result"` slot, or `None` if no
task wrote it.

Also exposes the vanilla-adoption classmethod `SchedulablePipeline.basic(
model, optimizer)` which wraps a standard training step into a pipeline.
"""

import contextlib
from typing import Any, Callable, Generic, Iterable, Iterator, Optional, TypeVar

import torch

from .context import BatchRing, TaskContext
from .deps import (
    infer_cross_stream_event_deps,
    infer_cross_stream_waits,
    producers_with_cross_stream_consumers,
)
from .executor import SequentialExecutor, ThreadedExecutor
from .schedule import Schedule, Stage
from .streams import StreamPool

# NVTX is optional; mirrors executor.py's pattern. Used to bracket each
# internal pipeline iteration so timeline analyzers can see boundaries
# between progress() iterations (in addition to per-task ranges already
# emitted by the executor).
try:
    import nvtx as _nvtx
except ImportError:  # pragma: no cover - nvtx absence
    _nvtx = None


def _progress_nvtx_range(iter_count: int, max_offset: int = 0):
    """Wrap one internal pipeline iteration in an NVTX range tagged
    ``progress[iter=N]``. No-op when nvtx is unavailable or CUDA is
    not initialized — the range would have no profiler to record into.

    The emitted ``N`` is *user-visible* — it equals ``iter_count -
    max_offset`` so the first steady-state iter is ``iter=0`` instead
    of ``iter=max_offset`` (the internal counter at that point). This
    aligns with the outer ``step N`` NVTX emitted by the training loop
    (e.g. ``training.py`` does ``range_push(f"step {train_iter}")``
    with ``train_iter`` starting at 0). Prefill iters get negative
    indices (``-max_offset .. -1``) so they remain distinguishable.
    """
    if _nvtx is None or not torch.cuda.is_available():
        return contextlib.nullcontext()
    visible = iter_count - max_offset
    return _nvtx.annotate(f"progress[iter={visible}]")


__all__ = ["Pipeline", "SchedulablePipeline"]


In = TypeVar("In")
Out = TypeVar("Out")


class SchedulablePipeline(Generic[In, Out]):
    """Drives a `Schedule` through iterations of a dataloader.

    For every `progress(batch_iter)` call:
      1. Pull one batch from `batch_iter` (raises `StopIteration` if
         exhausted).
      2. Populate `slots["batch_cpu"]` with the pulled batch; tasks
         read from the slot and never call `next()` themselves.
      3. Execute every stage's tasks in declaration order, each under
         its bound stream context.
      4. Return the value in the `"step_result"` slot (or `None`).
      5. Advance the ring (evicts the slot store).
    """

    RETURN_SLOT: str = "step_result"

    def __init__(
        self,
        schedule: Schedule,
        stream_pool: StreamPool,
        *,
        executor: Optional[object] = None,
        nvtx: bool = True,
    ) -> None:
        # Re-order tasks by within-progress DAG topological sort so
        # execution order is driven by reads/writes/depends_on/
        # same_progress_sync edges, NOT by author declaration order.
        # Declaration order is reduced to a tie-breaker for tasks the
        # DAG leaves unconstrained. This eliminates silent-bug risk
        # where author-declared order conflicts with the DAG (e.g. a
        # ``same_progress_sync`` consumer accidentally declared before
        # its producer).
        from .deps import topological_sort

        if len(schedule.stages) <= 1:
            sorted_tasks = topological_sort(schedule)
            schedule = Schedule(
                stages=(Stage(tasks=sorted_tasks),),
                stream_slots=schedule.stream_slots,
            )
        # Multi-stage schedules keep their stage barriers. The validator
        # rejects any same-progress dependency that would require moving
        # a producer across those barriers.

        self._schedule = schedule
        self._stream_pool = stream_pool
        self._nvtx = nvtx
        # Pluggable executor; defaults to sequential.
        if executor is None:
            self._executor = SequentialExecutor()
        elif executor == "threaded":
            self._executor = ThreadedExecutor()
        elif isinstance(executor, (SequentialExecutor, ThreadedExecutor)):
            self._executor = executor
        else:
            raise TypeError(
                f"executor must be None, 'threaded', SequentialExecutor, "
                f"or ThreadedExecutor, got {type(executor).__name__}"
            )

        # Validate the schedule contract before touching runtime state.
        from .autosched.validator import validate as _validate

        _validate(schedule, stream_pool)

        self._ring: BatchRing = BatchRing(schedule.in_flight_batches)
        self._ctx: TaskContext = TaskContext(self._ring, stream_pool)
        self._max_offset: int = schedule.in_flight_batches - 1

        # Cross-stream waits are inferred once and applied before each
        # task. Event deps are preferred; stream waits are the fallback
        # during warmup or when a producer event is unavailable.
        self._cross_stream_waits = infer_cross_stream_waits(schedule)
        self._cross_stream_event_deps = infer_cross_stream_event_deps(schedule)
        # Same-stream-only producers do not need completion events.
        self._producers_to_record = producers_with_cross_stream_consumers(schedule)

        # Prefill/drain state: iter_count is the internal iteration
        # counter; pulled is the running count of batches pulled from
        # the iterator; exhausted flips True when next() raised.
        self._internal_iter: int = 0
        self._pulled: int = 0
        self._exhausted: bool = False
        self._prefill_done: bool = False
        # Iterator identity marks the boundary between fully drained
        # dataloaders; switching mid-flight is rejected in ``progress``.
        self._driving_iter: Optional[object] = None

        # Number of pre-seeded batches the engine should not pull again.
        self._seeded: int = 0

        # One-time init hook per task (HugeCTR parity).
        for task in schedule.all_tasks():
            task.init(self._ctx)

    # ------------------------------------------------------------------
    # Internal bootstrap pre-population for adapter layers
    # ------------------------------------------------------------------

    def _seed_first_batch(self, slot_contents: dict) -> None:
        """Pre-populate ring.at(max_offset) with the given slot values
        before the first ``progress()`` call.

        Seeded tasks should be idempotent: if the slot already contains
        their outputs, they should skip the duplicated work. ``batch_cpu``
        is required because the prefill/drain mask counts seeded batches
        the same way it counts dataloader pulls.
        """
        if self._pulled > 0 or self._internal_iter > 0:
            raise RuntimeError(
                "_seed_first_batch() must be called before progress(); "
                f"pipeline already ran (pulled={self._pulled}, "
                f"internal_iter={self._internal_iter})."
            )
        if "batch_cpu" not in slot_contents:
            raise ValueError(
                "_seed_first_batch requires 'batch_cpu' in slot_contents "
                "(matches the engine's auto-pull slot name)."
            )
        target_slot = self._ring.at(self._max_offset)
        for name, value in slot_contents.items():
            target_slot.set(name, value)
        self._pulled += 1
        self._seeded += 1

    # ------------------------------------------------------------------
    # Prefill/drain mask
    # ------------------------------------------------------------------

    def _should_run(self, task, iter_count: int, pulled: int) -> bool:
        """Return whether a task's lookahead slot is live this iteration."""
        k = task.batch_offset
        lo = self._max_offset - k
        hi = pulled + (self._max_offset - k)
        return lo <= iter_count < hi

    def _run_one_internal_iter(self, batch_iter) -> Optional[object]:
        """Run one internal iteration and advance the ring."""
        with _progress_nvtx_range(self._internal_iter, self._max_offset):
            # Pull into the furthest-ahead slot unless it was seeded.
            if self._seeded > 0:
                self._seeded -= 1
            elif not self._exhausted:
                try:
                    batch = next(batch_iter)
                    self._ring.at(self._max_offset).set("batch_cpu", batch)
                    self._pulled += 1
                except StopIteration:
                    self._exhausted = True

            # After exhaustion, stop once even the offset=0 task's
            # widest mask can no longer fire.
            if (
                self._exhausted
                and self._internal_iter >= self._pulled + self._max_offset
            ):
                raise StopIteration

            iter_count = self._internal_iter
            mask = lambda task: self._should_run(task, iter_count, self._pulled)

            for stage in self._schedule.stages:
                self._executor.execute_stage(
                    stage,
                    self._ctx,
                    iter_count,
                    mask,
                    self._cross_stream_waits,
                    self._stream_pool,
                    event_deps=self._cross_stream_event_deps,
                    producers_to_record=self._producers_to_record,
                )

            # Restore active offset for any external inspection.
            self._ctx._active_offset = 0

            # Capture result before advancing; offset=0 is about to recycle.
            result = self._ring.current().get(self.RETURN_SLOT, None)
            self._ring.advance()
            self._internal_iter += 1
            return result

    def progress(self, batch_iter: Iterator) -> Optional[object]:
        """Advance one user-visible iteration.

        The first call performs any needed ring prefill. A new iterator
        resets pipeline state only after the previous iterator has fully
        drained. One ``SchedulablePipeline`` should be driven by one host
        thread at a time.
        """
        if self._driving_iter is not batch_iter:
            # Mid-flight = batches still propagating through deeper
            # offsets in the ring. Only possible when max_offset > 0;
            # max_offset == 0 schedules complete every batch within
            # one progress() call so the ring is always empty between
            # calls and switching iterators is safe.
            mid_flight = (
                self._driving_iter is not None
                and self._max_offset > 0
                and not self._exhausted
                and self._internal_iter > 0
            )
            if mid_flight:
                raise RuntimeError(
                    "SchedulablePipeline.progress() received a new "
                    "iterator before the previous one drained "
                    f"(_internal_iter={self._internal_iter}, "
                    f"_pulled={self._pulled}, _exhausted=False). The "
                    "previous slice still has in-flight batches in "
                    "the ring; restarting now would silently discard "
                    "them. Drive the previous iterator until it "
                    "raises StopIteration before switching."
                )
            self._driving_iter = batch_iter
            self._internal_iter = 0
            self._pulled = self._seeded
            self._exhausted = False
            self._prefill_done = False
        # Absorb the ring prefill before returning the first steady result.
        if not self._prefill_done:
            for _ in range(self._max_offset):
                self._run_one_internal_iter(batch_iter)
            self._prefill_done = True

        # Steady or drain iteration. `_run_one_internal_iter` raises
        # StopIteration itself when no task's mask can fire anymore.
        return self._run_one_internal_iter(batch_iter)

    def step(self, batch) -> Optional[object]:
        """Convenience: run one iteration on a single batch.

        Equivalent to `self.progress(iter([batch]))`. Intended for the
        T1 adoption path where the user has a `for batch in loader:`
        loop and wants minimal intrusion — one line change per step.
        """
        return self.progress(iter([batch]))

    def shutdown(self) -> None:
        """Release executor resources (e.g. thread pool)."""
        self._executor.shutdown()

    def __enter__(self) -> "SchedulablePipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Preset: vanilla training-step pipeline
    # ------------------------------------------------------------------

    @classmethod
    def basic(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        # overlap knobs
        prefetch: bool = False,
        memcpy_stream: bool = False,
        # execution strategy
        threaded: bool = False,
        thread_map: Optional[object] = None,
        executor: Optional[object] = None,
        # escape hooks (AMP / clip / scheduler / custom loss)
        forward_fn: Optional[Callable[[torch.nn.Module, Any], Any]] = None,
        loss_fn: Optional[Callable[[Any], torch.Tensor]] = None,
        backward_fn: Optional[Callable[[torch.Tensor], None]] = None,
        optimizer_step_fn: Optional[Callable[[], None]] = None,
        # robustness: user may pass device explicitly for parameter-less
        # or device-ambiguous modules
        device: Optional[torch.device] = None,
    ) -> "SchedulablePipeline":
        """Assemble a vanilla training-step pipeline.

        Default convention: `model(batch)` returns a scalar loss OR a
        tuple whose first element is the loss. The full return value
        passes through to `pipe.progress()` via the `"step_result"`
        slot.

        Escape kwargs cover the four places a realistic PyTorch loop
        touches that the engine can't reach into:

          forward_fn        - wrap the forward pass (e.g. autocast)
          loss_fn           - custom loss extraction from model output
          backward_fn       - custom backward (e.g. scaler.scale(l).backward())
          optimizer_step_fn - clip / scaler.step / scaler.update / scheduler.step

        Adoption bands:
          - T1 vanilla: `SchedulablePipeline.basic(model, optimizer)` + `pipe.step(batch)` → ≤8-line diff
          - T2 AMP/clip/scheduler via escape kwargs              → ≤15-line diff
        """
        # Avoid circular import: _presets imports Task/DataSlot
        # already; local import keeps `_presets` strictly internal.
        from ._presets import (  # noqa: PLC0415 — intentional lazy
            _make_backward_task,
            _make_forward_task,
            _make_h2d_task,
            _make_optimizer_task,
            _make_zero_grad_task,
        )

        # Device resolution: explicit kwarg > first parameter > first
        # buffer > CPU fallback. `next(model.parameters())` on a
        # parameterless module raises StopIteration — harden against it.
        if device is None:
            import itertools

            for tensor in itertools.chain(model.parameters(), model.buffers()):
                device = tensor.device
                break
            else:
                device = torch.device("cpu")

        # When prefetch is enabled, H2D moves into the next-batch
        # slot (batch_offset=1) so it overlaps the current batch's
        # forward/backward/optimizer on another stream.
        h2d_stream = "memcpy" if memcpy_stream else "default"
        h2d_offset = 1 if prefetch else 0

        tasks = (
            _make_h2d_task(device, stream=h2d_stream, batch_offset=h2d_offset),
            _make_zero_grad_task(optimizer, stream="default"),
            _make_forward_task(
                model,
                forward_fn=forward_fn,
                loss_fn=loss_fn,
                stream="default",
            ),
            _make_backward_task(backward_fn=backward_fn, stream="default"),
            _make_optimizer_task(
                optimizer,
                optimizer_step_fn=optimizer_step_fn,
                stream="default",
            ),
        )
        stream_slots = ("default", "memcpy") if memcpy_stream else ("default",)
        schedule = Schedule(
            stages=(Stage(tasks=tasks),),
            stream_slots=stream_slots,
        )
        pool_dict: dict = {
            "default": (
                torch.cuda.default_stream(device) if device.type == "cuda" else None
            )
        }
        if memcpy_stream:
            pool_dict["memcpy"] = (
                torch.cuda.Stream(device) if device.type == "cuda" else None
            )
        pool = StreamPool(pool_dict)
        # Resolve executor: explicit executor > threaded flag > default
        if executor is None and threaded:
            executor = ThreadedExecutor(thread_map=thread_map)
        return cls(schedule, pool, executor=executor)


class Pipeline(Generic[In]):
    """User-facing declarative pipeline.

    Wraps a flat list of tasks into a single-stage ``Schedule`` and
    derives stream slots from ``task.stream``. Use
    ``SchedulablePipeline`` directly when an adapter needs explicit
    multi-stage layout.
    """

    def __init__(
        self,
        tasks: "Iterable[Task]",
        stream_pool: StreamPool,
        *,
        executor: Optional[object] = None,
        nvtx: bool = True,
    ) -> None:
        from .task import Task as _Task  # local import keeps top-level lean

        task_tuple = tuple(tasks)
        if not task_tuple:
            raise ValueError("Pipeline requires at least one task.")
        for t in task_tuple:
            if not isinstance(t, _Task):
                raise TypeError(
                    f"Pipeline.tasks entries must be Task instances, got "
                    f"{type(t).__name__}: {t!r}"
                )

        slots = {t.stream for t in task_tuple}
        slots.add("default")
        stream_slots = tuple(sorted(slots))

        schedule = Schedule(
            stages=(Stage(tasks=task_tuple),),
            stream_slots=stream_slots,
        )

        self._impl: SchedulablePipeline = SchedulablePipeline(
            schedule, stream_pool, executor=executor, nvtx=nvtx
        )

    @property
    def schedule(self) -> Schedule:
        """Engine-side ``Schedule`` synthesized from the task list."""
        return self._impl._schedule

    def progress(self, batch_iter) -> Optional[object]:
        """Run one user-facing iteration. See
        :meth:`SchedulablePipeline.progress`."""
        return self._impl.progress(batch_iter)

    def step(self, batch) -> Optional[object]:
        """Run one iteration on a single batch."""
        return self._impl.step(batch)

    def shutdown(self) -> None:
        """Release engine resources (e.g. thread pool)."""
        return self._impl.shutdown()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()
