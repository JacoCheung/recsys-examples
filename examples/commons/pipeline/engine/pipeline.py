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

"""SchedulablePipeline — the driver that consumes a Schedule + StreamPool.

v1 scope (single-stream, single-batch, no prefill/drain):

    pool = StreamPool({"default": None})
    pipe = SchedulablePipeline(schedule, pool)
    for batch in loader:
        result = pipe.progress(iter([batch]))

Returns the value written to the `"step_result"` slot (SPEC §4.4).
`None` if no task wrote it.

Also exposes the vanilla-adoption classmethod `SchedulablePipeline.basic(
model, optimizer)` which wraps a standard training step into a pipeline
(SPEC §4.7 T1/T2 — ≤8 / ≤15 line diff adoption).

V3 adds cross-stream `wait_stream` insertion; V4 adds N-batch ring +
prefill/drain; V5 adds validator; V9 adds auto-scheduler.
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

    v1 behavior — for every `progress(batch_iter)` call:
      1. Pull one batch from `batch_iter` (raises `StopIteration` if
         exhausted; V1 propagates it directly).
      2. Populate `slots["batch_cpu"]` with the pulled batch
         (SPEC §4.7 protocol — tasks read from the slot, never call
         `next()` themselves).
      3. Execute every stage's tasks in declaration order, each under
         its bound stream context.
      4. Return the value in the `"step_result"` slot (or `None`).
      5. Advance the ring (evicts the slot store).

    Key omissions vs later slices:
      - No cross-stream wait_stream insertion (lands in V3).
      - No prefill/drain (lands in V4; V1 requires N=1).
      - No `depends_on` edge enforcement (V5 validator).
      - No `loss.backward()` gating via autograd spike (V2).
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
        # else: multi-stage schedules keep their original stage-level
        # ordering (no current callsites; revisit when one appears).

        self._schedule = schedule
        self._stream_pool = stream_pool
        self._nvtx = nvtx
        # Problem #3: pluggable executor. Default is sequential
        # (backward-compatible with Problem #1).
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

        # V4: multi-batch in-flight supported via BatchRing + §4.8
        # prefill/drain mask. V1 hardcoded n=1; V4 removes that cap.

        # V5: all 8 §4.2 validity rules enforced here (replaces the
        # ad-hoc pre-V5 checks that previously lived in this ctor +
        # `deps.py`).
        from .autosched.validator import validate as _validate

        _validate(schedule, stream_pool)

        self._ring: BatchRing = BatchRing(schedule.in_flight_batches)
        self._ctx: TaskContext = TaskContext(self._ring, stream_pool)
        self._max_offset: int = schedule.in_flight_batches - 1

        # Cross-stream wait inference (SPEC §4.2 rule 8, §4.8 deps.py).
        # Computed once at construction; applied before each task run().
        # Two views: stream-list (legacy, used as first-iter fallback)
        # and event-deps (fine-grained, prefers wait_event when the
        # producer's completion event is on the ring slot).
        self._cross_stream_waits = infer_cross_stream_waits(schedule)
        self._cross_stream_event_deps = infer_cross_stream_event_deps(schedule)
        # Only producers some cross-stream consumer actually waits on
        # need ``cudaEventRecord`` after running; the rest skip the
        # record (same-stream FIFO already orders their work).
        self._producers_to_record = producers_with_cross_stream_consumers(schedule)

        # SPEC §4.8 state: iter_count is the internal iteration
        # counter; pulled is the running count of batches pulled from
        # the iterator; exhausted flips True when next() raised.
        self._internal_iter: int = 0
        self._pulled: int = 0
        self._exhausted: bool = False
        self._prefill_done: bool = False
        # Identity of the iterator currently being driven. When the
        # caller hands in a new iterator (e.g. switching from the train
        # loader to the eval loader), ``progress()`` resets the §4.8
        # state so a fresh prefill kicks in. Mirrors legacy
        # ``TrainPipeline._next_batch`` iterator-identity check
        # (train_pipeline.py:418).
        self._driving_iter: Optional[object] = None

        # Bootstrap counter: how many batches have been seeded via
        # ``_seed_first_batch`` (pre-populated into ring slots BEFORE
        # any progress() call). Engine will NOT pull from the
        # dataloader for the next ``_seeded`` iters — the slot already
        # has its batch_cpu set and tasks should skip any work that's
        # already been done externally.
        self._seeded: int = 0

        # One-time init hook per task (HugeCTR parity).
        for task in schedule.all_tasks():
            task.init(self._ctx)

    # ------------------------------------------------------------------
    # Internal bootstrap pre-population (Problem #2 bootstrap fix)
    # ------------------------------------------------------------------
    #
    # Underscore-prefixed: this is an internal hook used by adapter
    # layers (e.g. HSTUPipeline) that need to run framework-setup
    # work on a real batch before delegating to the engine. End users
    # of SchedulablePipeline should NOT call this directly — same
    # privacy convention as torchrec's _pipeline_model /
    # _init_pipelined_modules.

    def _seed_first_batch(self, slot_contents: dict) -> None:
        """Pre-populate ring.at(max_offset) with the given slot values
        before the first ``progress()`` call, and mark one batch as
        pulled.

        The engine will:
          1. Skip the ``next(batch_iter)`` call for the first iter
             (per seeded batch — supports multiple seeds if needed).
          2. Treat the pre-populated slot as a legitimate in-flight
             batch (it'll advance down to compute naturally).

        Tasks that rely on slot contents being set by earlier pipeline
        stages (e.g. ``h2d`` sets ``batch_gpu`` from ``batch_cpu``)
        should be made **idempotent** by checking for prior slot values
        in their body.

        Parameters
        ----------
        slot_contents : dict
            Values to seed into ``ring.at(max_offset)``'s slot store.
            Must include ``"batch_cpu"`` to satisfy the §4.8 mask;
            typically also includes ``batch_gpu``, per-batch context
            objects, and anything downstream tasks read.
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
    # V4 §4.8 implementation
    # ------------------------------------------------------------------

    def _should_run(self, task, iter_count: int, pulled: int) -> bool:
        """SPEC §4.8 mask formula.

        Task with `batch_offset = k` runs at iteration `iter_count`
        iff `(max_offset - k) ≤ iter_count < M_known + (max_offset - k)`
        where `M_known = pulled` while pulling is live, `= final M`
        after StopIteration. Both cases: `pulled` tracks batches
        loaded into the ring so far.
        """
        k = task.batch_offset
        lo = self._max_offset - k
        hi = pulled + (self._max_offset - k)
        return lo <= iter_count < hi

    def _run_one_internal_iter(self, batch_iter) -> Optional[object]:
        """One internal pipeline iteration: pull (maybe) + apply §4.8
        mask + run qualifying tasks + capture result + advance ring.

        Raises `StopIteration` iff, after the pull attempt, no task's
        mask can be satisfied now OR in any future iteration — this
        is the "end" phase of §4.8. Callers that drive this from
        prefill propagate StopIteration up to the user (which is
        correct: if M < prefill-count the pipeline ends during
        prefill, and the first `progress()` call raises).
        """
        with _progress_nvtx_range(self._internal_iter, self._max_offset):
            # Pull next batch into the furthest-ahead slot if iterator
            # still has batches. Populates `batch_cpu` at
            # batch_offset=max_offset per SPEC §4.7 protocol.
            #
            # Exception: if `_seed_first_batch` was called, the first
            # `_seeded` iters skip the pull — the slot is already populated
            # and `_pulled` has already been bumped.
            if self._seeded > 0:
                self._seeded -= 1
            elif not self._exhausted:
                try:
                    batch = next(batch_iter)
                    self._ring.at(self._max_offset).set("batch_cpu", batch)
                    self._pulled += 1
                except StopIteration:
                    self._exhausted = True

            # End check AFTER pull attempt: at this `_internal_iter`,
            # with this `_pulled` count, can any task's mask still fire?
            # The k=0 task has the widest window (hi = pulled + max_offset).
            # So once `_internal_iter >= pulled + max_offset` with the
            # iterator exhausted, no further useful work is possible.
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

            # Capture result (what the offset=0 compute task wrote) BEFORE
            # advancing the ring — the current slot is about to be evicted.
            result = self._ring.current().get(self.RETURN_SLOT, None)
            self._ring.advance()
            self._internal_iter += 1
            return result

    def progress(self, batch_iter: Iterator) -> Optional[object]:
        """User-facing driver — SPEC §4.8 contract.

        M batches in → M results out. Call M+1 raises `StopIteration`.
        Matches legacy `TrainPipeline.progress(...)`:

            it = iter(dataloader)
            while True:
                try:
                    r = pipe.progress(it)
                except StopIteration:
                    break
                use(r)

        First call absorbs `max_offset` prefill iterations so the
        user never sees a `None` from an incomplete ring.

        When the caller hands in a different iterator than last time
        (e.g. train → eval), the §4.8 state is reset so a fresh
        prefill kicks in on the new iterator. The previous slice must
        already have drained (i.e. the ``StopIteration`` that ends a
        slice has been observed) — switching iterators mid-flight
        would silently discard in-flight batches and is rejected here
        with a ``RuntimeError``. Mirrors legacy
        ``TrainPipeline._next_batch`` iterator-change reset
        (train_pipeline.py:418), with an added drain-required guard
        (Codex MEDIUM 2026-04-26).

        Concurrency note: ``progress()`` is single-driver only — a
        single host thread should drive any one ``SchedulablePipeline``.
        The threaded executor parallelizes tasks **inside** a single
        ``progress()`` call; concurrent ``progress()`` invocations on
        the same instance race on ``_driving_iter``, ``_internal_iter``,
        ``_pulled``, and the ``BatchRing``.
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
        # Prefill absorption: first user call runs `max_offset`
        # internal iterations before returning the first steady
        # result. Each iter advances the ring so subsequent iters
        # see the right slot positions. If the dataloader is shorter
        # than the prefill requires (M < max_offset), the end check
        # inside `_run_one_internal_iter` raises StopIteration from
        # within the prefill loop — this propagates correctly: user's
        # first `progress()` call sees StopIteration (M=0 case).
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
    # Preset: vanilla training-step pipeline (SPEC §4.7)
    # ------------------------------------------------------------------

    @classmethod
    def basic(
        cls,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        # overlap knobs — enabled in V4
        prefetch: bool = False,
        memcpy_stream: bool = False,
        # execution strategy (Problem #3)
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

        Adoption bands (SPEC §4.7):
          - T1 vanilla: `SchedulablePipeline.basic(model, optimizer)` + `pipe.step(batch)` → ≤8-line diff
          - T2 AMP/clip/scheduler via escape kwargs              → ≤15-line diff

        V2 ships the single-stream path only. `prefetch=True` and
        `memcpy_stream=True` land in V4.
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

        # V4: prefetch moves H2D into the next-batch slot
        # (batch_offset=1) so it overlaps the current batch's
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
    """SPEC_p4 v2 user-facing declarative pipeline.

    Thin wrapper over :class:`SchedulablePipeline` that takes a flat
    list of tasks and constructs the underlying ``Schedule``
    internally. The user describes the per-batch DAG once via
    ``Task(...)`` declarations; ring depth, stream slots, and stage
    layout are derived.

    Differences from :class:`SchedulablePipeline` (the imperative API):

    - Single argument ``tasks=[Task(...), ...]``. No ``Stage`` /
      ``Schedule`` object construction at the call site.
    - ``stream_slots`` is the union of ``task.stream`` across all
      tasks (``"default"`` is always included so the engine has an
      anchor).
    - All tasks land in a single ``Stage``. Stage boundaries were
      already cosmetic — engine cross-stream wait inference is
      stage-agnostic per SPEC §4.2.
    - Ring depth is ``max(t.lookahead) + 1`` via the existing
      ``Schedule.in_flight_batches`` derivation; tasks can use either
      ``lookahead=...`` (SPEC_p4 v2) or ``batch_offset=...`` (legacy)
      since both alias the same field.

    The user still constructs and passes a :class:`StreamPool` —
    stream resources are device-specific and outside the engine's
    pure-Python scope.

    Adapter authors that need explicit multi-stage layout or other
    ``Schedule`` controls can keep using :class:`SchedulablePipeline`
    directly; nothing about this class makes that path go away.
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
