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

"""Per-iteration context: SlotStore, BatchRing, TaskContext.

A Task body sees the pipeline state through `TaskContext`:

    def _forward(ctx):
        x = ctx.slots["batch_cpu"]
        ctx.slots.set("loss", model(x))

The store is scoped to one in-flight batch. V1 carries a single batch
(`BatchRing(n=1)`); V4 generalizes to N batches with prefill/drain.
"""

import threading
from typing import Any, Dict, Generic, TypeVar

__all__ = ["SlotStore", "BatchRing", "TaskContext"]


In = TypeVar("In")


class SlotStore:
    """Per-batch named value store.

    Keyed by slot name (str). V1 does not key on `(name, batch_offset)`
    because `in_flight_batches=1` → offset is always 0. V4 promotes
    this to a `(name, batch_offset)` map.

    Each SlotStore also carries a registry of **producer-completion
    events**, keyed by task name. After a task runs on its stream, the
    executor records a CUDA event on that stream and stores it here on
    the slot the task wrote (or, for cross-iter handoff, on the slot
    matching the task's batch_offset). A consumer task that reads from
    that slot can then `wait_event(producer_event)` instead of
    `wait_stream(producer_stream)` — fine-grained sync at task
    granularity, not stream granularity.

    Events are typed `Any` here so this module remains
    framework-agnostic; the executor (which already imports torch)
    populates them with `torch.cuda.Event` instances.
    """

    def __init__(self) -> None:
        self._data: Dict[str, Any] = {}
        self._events: Dict[str, Any] = {}

    def __contains__(self, name: str) -> bool:
        return name in self._data

    def __getitem__(self, name: str) -> Any:
        if name not in self._data:
            raise KeyError(
                f"Slot '{name}' not in store. Did an upstream task fail "
                f"to write it, or is the task ordering wrong?"
            )
        return self._data[name]

    def set(self, name: str, value: Any) -> None:
        self._data[name] = value

    def get(self, name: str, default: Any = None) -> Any:
        return self._data.get(name, default)

    def get_event(self, task_name: str) -> Any:
        """Producer-completion event for `task_name`, or None if no
        producer has recorded one on this slot yet."""
        return self._events.get(task_name)

    def set_event(self, task_name: str, event: Any) -> None:
        """Register the producer-completion event for `task_name`.

        Called by the executor right after the task body returns; the
        executor records the event on the task's CUDA stream first.
        Subsequent re-recording on the same `torch.cuda.Event` is
        intentional — `Event.record()` overwrites the prior record, so
        the same event object is reused across iterations as the slot
        rotates through the ring.
        """
        self._events[task_name] = event

    def has_event(self, task_name: str) -> bool:
        return task_name in self._events

    def clear(self) -> None:
        """Clear data only; events persist across `BatchRing.advance()`
        so the same `torch.cuda.Event` objects are reused (re-recorded
        in the next iteration when this slot rotates back to the high
        offset). Callers wanting full reset should use `clear_all()`.
        """
        self._data.clear()

    def clear_all(self) -> None:
        """Clear both data and the event registry. Used at shutdown
        or when discarding pipeline state — not in the hot loop."""
        self._data.clear()
        self._events.clear()


class BatchRing(Generic[In]):
    """N in-flight batches, each with its own SlotStore.

    The ring is indexed by `batch_offset`: index 0 is the CURRENT
    batch (the one about to be returned to the user), index 1 is
    the NEXT batch (prefetched one iteration ahead), ..., index
    N-1 is the furthest-ahead batch (prefetched N-1 iterations ahead).

    On `advance()`:
      - The slot store at index 0 is dropped (evicted); its
        tensors' refcounts decrease so CUDA memory can be freed.
      - All remaining stores shift toward index 0 (current batch
        slides out of the ring; the batch that was 1-ahead is
        now current; a fresh empty store is appended at index N-1).

    No cross-iteration carry-over in v1 — `carries_over` was cut
    from scope.
    """

    def __init__(self, n: int) -> None:
        if n < 1:
            raise ValueError(f"BatchRing needs at least 1 slot, got n={n}")
        self._n: int = n
        self._slots = [SlotStore() for _ in range(n)]

    @property
    def n(self) -> int:
        return self._n

    def at(self, batch_offset: int) -> SlotStore:
        """Slot store for the batch at `batch_offset` (0 = current)."""
        if batch_offset < 0 or batch_offset >= self._n:
            raise IndexError(f"batch_offset={batch_offset} out of range [0, {self._n})")
        return self._slots[batch_offset]

    def current(self) -> SlotStore:
        """Slot store for the current batch (batch_offset=0).

        Backward-compat shim for V1/V2/V3 callers. New code should
        prefer `at(batch_offset)` when offset > 0 is possible.
        """
        return self._slots[0]

    def advance(self) -> None:
        """End-of-iteration: shift ring toward offset=0.

        The slot at offset=0 is **recycled** — its data is cleared, but
        the SlotStore object itself (and its event registry) are
        preserved and rotated to the highest offset. This lets producer
        tasks at the high offset re-record onto the same
        `torch.cuda.Event` objects iteration after iteration, so we
        don't churn CUDA events. (See `SlotStore.set_event` for the
        re-record contract.)

        After advance:
          - old offset=0 slot is now empty at offset=N-1, ready to
            receive a freshly pulled batch;
          - old offset=k for k>0 slides to offset=k-1.

        Callers must not hold long-lived references to a SlotStore
        across `advance()` — the object identity is stable, but its
        offset (and therefore its semantic role) changes.
        """
        recycled = self._slots[0]
        recycled.clear()  # data cleared; events kept for re-record
        # Shift slots[1..n-1] to slots[0..n-2]
        self._slots[:-1] = self._slots[1:]
        # Recycled slot lands at the highest offset
        self._slots[-1] = recycled


class TaskContext(Generic[In]):
    """Handle passed to every `Task.run(ctx)` invocation.

    Exposes the active slot store, the stream pool, and the iteration
    counter. Framework-agnostic — imports only stdlib.

    For tasks with `batch_offset > 0` (e.g. prefetch H2D), the engine
    sets `ctx._active_offset` to the task's offset before calling
    `run(ctx)`, so `ctx.slots` transparently returns the right
    slot-store in the ring. Tasks authored for V1-V3 that used
    `ctx.slots` still work — their `batch_offset` is 0.
    """

    def __init__(self, ring: BatchRing, stream_pool) -> None:
        self._ring = ring
        self._stream_pool = stream_pool
        # Thread-local storage for _active_offset and iter_count so
        # that ThreadedExecutor (Problem #3) can set them per-thread
        # without races.  SequentialExecutor works identically — the
        # main thread's local state is used.
        self._local = threading.local()

    @property
    def _active_offset(self) -> int:
        return getattr(self._local, "offset", 0)

    @_active_offset.setter
    def _active_offset(self, value: int) -> None:
        self._local.offset = value

    @property
    def iter_count(self) -> int:
        return getattr(self._local, "iter_count", 0)

    @iter_count.setter
    def iter_count(self, value: int) -> None:
        self._local.iter_count = value

    @property
    def slots(self) -> SlotStore:
        """Slot store at the active task's `batch_offset`.

        V1-V3 tasks all have offset=0 so this returns the current
        batch's store. V4 tasks with offset>0 (e.g. prefetch H2D)
        read/write the future batch's store.
        """
        return self._ring.at(self._active_offset)

    def slots_at(self, batch_offset: int) -> SlotStore:
        """Explicit accessor for an arbitrary slot store in the ring.

        Tasks rarely need this — the `.slots` property honors the
        task's declared `batch_offset` automatically. Used by the
        engine internals (populating `batch_cpu` at
        `batch_offset=max_offset`).
        """
        return self._ring.at(batch_offset)

    @property
    def stream_pool(self):
        return self._stream_pool
