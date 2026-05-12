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

The store is scoped to N in-flight batches via `BatchRing`, with
prefill/drain to handle ring wraparound at the iteration boundary.
"""

import threading
from typing import Any, Dict, Generic, TypeVar

__all__ = ["SlotStore", "BatchRing", "TaskContext"]


In = TypeVar("In")


class SlotStore:
    """Per-batch named value store.

    Keyed by slot name (str); the `(name, batch_offset)` mapping lives
    in the enclosing `BatchRing`.

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
        """Convenience accessor for `batch_offset=0`; equivalent to
        `at(0)`. Prefer `at(batch_offset)` when `offset > 0` is
        possible.
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
    slot-store in the ring.
    """

    def __init__(self, ring: BatchRing, stream_pool) -> None:
        self._ring = ring
        self._stream_pool = stream_pool
        # Thread-local storage for _active_offset and iter_count so
        # that ThreadedExecutor can set them per-thread without races.
        # SequentialExecutor works identically — the main thread's
        # local state is used.
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

        Tasks with `batch_offset=0` see the current batch's store;
        tasks with `offset>0` (e.g. prefetch H2D) read/write the
        future batch's store.
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

    # ------------------------------------------------------------------
    # Explicit Event escape hatch
    # ------------------------------------------------------------------
    #
    # Most cross-stream sync is auto-inferred by the engine from the
    # schedule's slot dependencies (see ``deps.infer_cross_stream_event_deps``
    # + ``executor._apply_cross_stream_waits``). The escape hatch is for
    # the rare case where a task body needs to publish a partial-progress
    # event from inside its own work — e.g. the producer wants to signal
    # downstream consumers as soon as a kernel chain reaches a checkpoint,
    # before the task body finishes — or a consumer needs to wait on a
    # specific named event whose producer isn't a separate task.
    #
    # Naming collision with the engine's auto-recorded
    # producer-completion events (keyed by ``task.name``) is avoided by
    # prefixing user-supplied names with ``"user:"``. A user task that
    # picks a label colliding with another task's name still ends up in
    # a private namespace.

    _USER_EVENT_PREFIX = "user:"

    def record_event(self, name: str, *, batch_offset: int = 0) -> Any:
        """Record a torch.cuda.Event on the current task's CUDA stream
        and stash it in the slot at ``batch_offset`` (default: 0, i.e.
        the same slot the active task writes to).

        The Event object is reused across iterations as the slot rotates
        through the ring (matching engine-level event recycling).
        Returns the event object so the caller can hold a reference if
        needed.

        Raises ``ImportError`` if torch is unavailable (CPU-only test
        host) — the escape hatch is GPU-only by definition.
        """
        if not name:
            raise ValueError("record_event(name) requires a non-empty name")
        slot = self._ring.at(batch_offset)
        full_name = self._USER_EVENT_PREFIX + name
        event = slot.get_event(full_name)
        if event is None:
            import torch  # local import to keep this module framework-agnostic

            event = torch.cuda.Event()
            slot.set_event(full_name, event)
        event.record()
        return event

    def wait_event(self, name: str, *, batch_offset: int = 0) -> bool:
        """Make the current CUDA stream wait on the user-recorded event
        ``name`` stored on the slot at ``batch_offset``. Returns True if
        a wait was actually issued, False if no producer has recorded
        the event yet (typical first-iter case — caller can fall back
        to a coarser ``wait_stream`` or ignore).
        """
        if not name:
            raise ValueError("wait_event(name) requires a non-empty name")
        slot = self._ring.at(batch_offset)
        full_name = self._USER_EVENT_PREFIX + name
        event = slot.get_event(full_name)
        if event is None:
            return False
        event.wait()
        return True
