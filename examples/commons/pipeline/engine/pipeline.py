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

from typing import Any, Callable, Generic, Iterator, Optional, TypeVar

import torch

from .context import BatchRing, TaskContext
from .deps import infer_cross_stream_waits
from .schedule import Schedule, Stage
from .streams import StreamPool

__all__ = ["SchedulablePipeline"]


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
        nvtx: bool = True,
    ) -> None:
        self._schedule = schedule
        self._stream_pool = stream_pool
        self._nvtx = nvtx

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

        # Cross-stream wait_stream inference (SPEC §4.2 rule 8, §4.8
        # deps.py). Computed once at construction; applied before each
        # task run().
        self._cross_stream_waits = infer_cross_stream_waits(schedule)

        # SPEC §4.8 state: iter_count is the internal iteration
        # counter; pulled is the running count of batches pulled from
        # the iterator; exhausted flips True when next() raised.
        self._internal_iter: int = 0
        self._pulled: int = 0
        self._exhausted: bool = False
        self._prefill_done: bool = False

        # One-time init hook per task (HugeCTR parity).
        for task in schedule.all_tasks():
            task.init(self._ctx)

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
        # Pull next batch into the furthest-ahead slot if iterator
        # still has batches. Populates `batch_cpu` at
        # batch_offset=max_offset per SPEC §4.7 protocol.
        if not self._exhausted:
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
        if self._exhausted and self._internal_iter >= self._pulled + self._max_offset:
            raise StopIteration

        anchor_device = self._stream_pool.anchor_device
        iter_count = self._internal_iter

        for stage in self._schedule.stages:
            for task in stage.tasks:
                if not self._should_run(task, iter_count, self._pulled):
                    continue
                # Set the active offset so `ctx.slots` returns the
                # task's declared batch's slot store.
                self._ctx._active_offset = task.batch_offset
                self._ctx.iter_count = iter_count
                with self._stream_pool.use(task.stream):
                    waits = self._cross_stream_waits.get(task.name, ())
                    if waits and anchor_device is not None:
                        consumer_stream = torch.cuda.current_stream()
                        for producer_stream_name in waits:
                            producer_stream = self._stream_pool.get(
                                producer_stream_name
                            )
                            if producer_stream is None:
                                producer_stream = torch.cuda.default_stream(
                                    anchor_device
                                )
                            consumer_stream.wait_stream(producer_stream)
                    task.run(self._ctx)

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
        """
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

        result = self._ring.current().get(self.RETURN_SLOT, None)
        self._ring.advance()
        return result

    def step(self, batch) -> Optional[object]:
        """Convenience: run one iteration on a single batch.

        Equivalent to `self.progress(iter([batch]))`. Intended for the
        T1 adoption path where the user has a `for batch in loader:`
        loop and wants minimal intrusion — one line change per step.
        """
        return self.progress(iter([batch]))

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
        return cls(schedule, pool)
