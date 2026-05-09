# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for SlotStore event registry + BatchRing in-place rotate.

These tests are pure-Python (no torch/cuda required). They validate
that:

  * `SlotStore` carries a per-task event registry alongside its data,
    independent of slot reads/writes.
  * `SlotStore.clear()` clears data but keeps events (the contract the
    in-place ring rotate relies on for event reuse).
  * `BatchRing.advance()` rotates SlotStore objects in-place rather
    than dropping/allocating, so the same SlotStore (and its event
    registry) cycles through every offset over `n` iterations.

Real `torch.cuda.Event` objects are stand-ins here — we use sentinel
strings; behavior at the SlotStore/BatchRing layer is event-type
agnostic.
"""

from commons.pipeline.engine.context import BatchRing, SlotStore

# ---------------------------------------------------------------------
# SlotStore
# ---------------------------------------------------------------------


def test_slotstore_events_independent_of_data() -> None:
    s = SlotStore()
    s.set("x", 1)
    s.set_event("forward", "ev_fwd")

    assert s["x"] == 1
    assert s.get_event("forward") == "ev_fwd"
    assert s.has_event("forward")
    assert not s.has_event("backward")


def test_slotstore_clear_keeps_events() -> None:
    s = SlotStore()
    s.set("x", 1)
    s.set_event("forward", "ev_fwd")

    s.clear()

    assert "x" not in s
    assert s.get_event("forward") == "ev_fwd"  # event survives clear()
    assert s.has_event("forward")


def test_slotstore_clear_all_drops_both() -> None:
    s = SlotStore()
    s.set("x", 1)
    s.set_event("forward", "ev_fwd")

    s.clear_all()

    assert "x" not in s
    assert s.get_event("forward") is None
    assert not s.has_event("forward")


def test_slotstore_set_event_overwrites() -> None:
    s = SlotStore()
    s.set_event("forward", "ev_fwd_v1")
    s.set_event("forward", "ev_fwd_v2")

    assert s.get_event("forward") == "ev_fwd_v2"


# ---------------------------------------------------------------------
# BatchRing in-place rotate
# ---------------------------------------------------------------------


def test_advance_recycles_slot_objects() -> None:
    """SlotStore objects are rotated, not allocated fresh each advance."""
    ring = BatchRing(n=3)
    s0 = ring.at(0)
    s1 = ring.at(1)
    s2 = ring.at(2)

    ring.advance()

    # After advance: old offset=0 is recycled to highest offset;
    # offsets 1,2 shift down by one.
    assert ring.at(0) is s1, "offset=1 should slide to offset=0"
    assert ring.at(1) is s2, "offset=2 should slide to offset=1"
    assert ring.at(2) is s0, "old offset=0 should be recycled to highest"


def test_advance_clears_data_keeps_events_on_recycled_slot() -> None:
    """Recycled slot has data cleared but its event registry intact —
    so a producer at the highest offset can re-record on the same
    event objects iteration after iteration."""
    ring = BatchRing(n=3)
    s0 = ring.at(0)

    s0.set("payload", "batch_K_data")
    s0.set_event("h2d", "ev_h2d_obj")
    s0.set_event("forward", "ev_fwd_obj")

    ring.advance()

    # s0 is now at offset=2 (highest), data cleared, events kept.
    assert ring.at(2) is s0
    assert "payload" not in s0
    assert s0.get_event("h2d") == "ev_h2d_obj"
    assert s0.get_event("forward") == "ev_fwd_obj"


def test_advance_full_cycle_returns_objects_to_origin() -> None:
    """After `n` advances, every slot object is back at its original
    offset — confirms rotate (not pop/append) semantics."""
    n = 4
    ring = BatchRing(n=n)
    originals = [ring.at(k) for k in range(n)]

    for _ in range(n):
        ring.advance()

    for k in range(n):
        assert ring.at(k) is originals[k], (
            f"after {n} advances, offset={k} should hold the same SlotStore "
            f"object that started at offset={k}"
        )


def test_event_persists_through_full_ring_cycle() -> None:
    """An event recorded at the highest offset is still queryable after
    the slot rotates all the way back to the same offset (covers the
    re-record contract: event objects live as long as the ring)."""
    ring = BatchRing(n=3)
    s = ring.at(2)
    s.set_event("h2d", "ev_h2d_persistent")

    for _ in range(3):
        ring.advance()

    # s is back at offset=2; event still present.
    assert ring.at(2) is s
    assert s.get_event("h2d") == "ev_h2d_persistent"


def test_advance_payload_is_not_visible_after_one_advance() -> None:
    """Data written at offset=0 is dropped (cleared) when that slot
    rotates to the highest offset — the slot's role changed, so its
    old payload must not leak into the new role."""
    ring = BatchRing(n=2)
    s = ring.at(0)
    s.set("payload", "batch_K_data")

    ring.advance()

    # s is now at offset=1 (highest, in n=2 ring)
    assert ring.at(1) is s
    assert "payload" not in s, "old payload must be cleared on recycle"
