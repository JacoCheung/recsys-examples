# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for SlotStore event retention across BatchRing rotation."""

import pytest
from commons.pipeline.engine.context import BatchRing, SlotStore, TaskContext


def test_slotstore_clear_keeps_events_clear_all_drops_them() -> None:
    slot = SlotStore()
    slot.set("payload", "batch")
    slot.set_event("forward", "event")

    slot.clear()
    assert "payload" not in slot
    assert slot.get_event("forward") == "event"

    slot.clear_all()
    assert slot.get_event("forward") is None


def test_batchring_advance_rotates_objects_and_clears_payloads() -> None:
    ring = BatchRing(n=3)
    original = [ring.at(i) for i in range(3)]
    original[0].set("payload", "old")
    original[0].set_event("h2d", "event")

    ring.advance()

    assert ring.at(0) is original[1]
    assert ring.at(1) is original[2]
    assert ring.at(2) is original[0]
    assert "payload" not in ring.at(2)
    assert ring.at(2).get_event("h2d") == "event"


def test_batchring_full_cycle_preserves_event_registry() -> None:
    ring = BatchRing(n=3)
    slot = ring.at(2)
    slot.set_event("producer", "event")

    for _ in range(3):
        ring.advance()

    assert ring.at(2) is slot
    assert ring.at(2).get_event("producer") == "event"


def test_taskcontext_user_event_namespace_isolated() -> None:
    ring = BatchRing(n=2)
    ctx = TaskContext(ring, stream_pool=None)
    ring.at(0).set_event("forward", "engine-event")

    assert ctx.wait_event("forward") is False
    assert ring.at(0).get_event("forward") == "engine-event"


def test_taskcontext_rejects_empty_user_event_names() -> None:
    ctx = TaskContext(BatchRing(n=2), stream_pool=None)

    with pytest.raises(ValueError, match="non-empty name"):
        ctx.record_event("")
    with pytest.raises(ValueError, match="non-empty name"):
        ctx.wait_event("")
