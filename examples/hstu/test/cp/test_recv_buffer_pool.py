# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the ring-P2P recv buffer pool used by
`_multi_gpu_forward_arbitrary` / `_multi_gpu_backward_arbitrary`.

The pool MUST be keyed by (dtype, device, slot) ONLY — NOT by shape.
A previous version (commit a051cef1) keyed by shape and leaked ~80 MB
per training step on the HSTU benchmark (100 unique cu_seqlens × 6
slots), eventually OOM'ing at iter ~400 into a `CUDA error: an illegal
memory access`. These tests pin the bounded-growth invariant.

Single-process, no torchrun, no GPU needed (CPU device works fine for
the storage management semantics).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    pytest.skip(
        "test_recv_buffer_pool is single-process; skipping under torchrun",
        allow_module_level=True,
    )

_HSTU_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = Path(__file__).resolve().parents[3]
for p in (_HSTU_ROOT, _EXAMPLES_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))


@pytest.fixture
def clear_pool():
    from context_parallel.hstu_attn_cp import cp_recv_buffer_pool_clear

    cp_recv_buffer_pool_clear()
    yield
    cp_recv_buffer_pool_clear()


def test_same_template_returns_view_into_same_storage(clear_pool):
    """Repeated calls with the same kwargs return tensors that share storage."""
    from context_parallel.hstu_attn_cp import _get_recv_buffer

    template = torch.empty(128, 8, 64, dtype=torch.bfloat16)
    a = _get_recv_buffer(template, slot="fwd_recv_k")
    b = _get_recv_buffer(template, slot="fwd_recv_k")
    assert a.data_ptr() == b.data_ptr(), "same slot must alias same storage"
    assert a.shape == template.shape


def test_distinct_slots_map_to_distinct_storage(clear_pool):
    """recv_k and recv_v are accessed concurrently within one ring step;
    they MUST come from distinct physical storages."""
    from context_parallel.hstu_attn_cp import _get_recv_buffer

    template = torch.empty(128, 8, 64, dtype=torch.bfloat16)
    k = _get_recv_buffer(template, slot="fwd_recv_k")
    v = _get_recv_buffer(template, slot="fwd_recv_v")
    assert k.data_ptr() != v.data_ptr(), (
        "two slots must be physically distinct — otherwise NCCL P2P for "
        "recv_k and recv_v would race on the same storage within a ring step"
    )


def test_different_shapes_same_slot_share_storage_when_fits(clear_pool):
    """The HSTU benchmark cycles 100 unique cu_seqlens, so total_tokens
    varies per batch. The pool MUST NOT create a new entry per shape —
    that's the round-10 leak that this test guards against.

    With a (dtype, device, slot)-keyed pool, two different-shape templates
    sharing the same slot reuse the SAME underlying storage as long as
    the smaller fits.
    """
    from context_parallel import hstu_attn_cp as _cp
    from context_parallel.hstu_attn_cp import _get_recv_buffer

    big = torch.empty(256, 8, 64, dtype=torch.bfloat16)
    a = _get_recv_buffer(big, slot="fwd_recv_k")
    keys_after_big = set(_cp._recv_buffer_pool.keys())

    small = torch.empty(128, 8, 64, dtype=torch.bfloat16)
    b = _get_recv_buffer(small, slot="fwd_recv_k")
    keys_after_small = set(_cp._recv_buffer_pool.keys())

    assert keys_after_small == keys_after_big, (
        "shape variation MUST NOT add new pool entries — "
        f"keys grew from {keys_after_big} to {keys_after_small}"
    )
    assert (
        a.data_ptr() == b.data_ptr()
    ), "smaller shape into existing storage must alias, not allocate fresh"
    assert b.shape == small.shape, "view must match requested template shape"


def test_growing_shape_grows_storage_in_place(clear_pool):
    """If a later call needs MORE elements than the cached storage holds,
    the pool reallocates to fit. Only ONE entry remains."""
    from context_parallel import hstu_attn_cp as _cp
    from context_parallel.hstu_attn_cp import _get_recv_buffer

    small = torch.empty(128, 8, 64, dtype=torch.bfloat16)
    a = _get_recv_buffer(small, slot="fwd_recv_k")

    big = torch.empty(512, 8, 64, dtype=torch.bfloat16)
    b = _get_recv_buffer(big, slot="fwd_recv_k")

    assert b.shape == big.shape, "view must match the bigger requested shape"
    assert b.numel() >= a.numel(), "storage grew to fit the bigger request"
    # Pool still has exactly one entry per (dtype, device, slot).
    assert (
        len(_cp._recv_buffer_pool) == 1
    ), f"shape growth must NOT add entries; got {len(_cp._recv_buffer_pool)}"


def test_pool_bounded_under_simulated_jagged_workload(clear_pool):
    """Simulates the HSTU benchmark's per-step varying cu_seqlens. After
    100 distinct shapes × 6 slots, pool size MUST be O(slots), not
    O(shapes × slots) — that was the round-10 leak."""
    from context_parallel import hstu_attn_cp as _cp
    from context_parallel.hstu_attn_cp import _get_recv_buffer

    slots = [
        "fwd_recv_k",
        "fwd_recv_v",
        "bwd_recv_k",
        "bwd_recv_v",
        "bwd_recv_dk",
        "bwd_recv_dv",
    ]
    for k in range(100):
        template = torch.empty(128 + k, 8, 64, dtype=torch.bfloat16)
        for slot in slots:
            _get_recv_buffer(template, slot=slot)

    assert len(_cp._recv_buffer_pool) == len(slots), (
        f"pool must hold one entry per slot regardless of shape variety; "
        f"got {len(_cp._recv_buffer_pool)} entries for {len(slots)} slots"
    )


def test_returned_view_is_contiguous(clear_pool):
    """NCCL P2P requires contiguous tensors. A view into a 1-D contiguous
    storage front-sliced to N elements and reshaped to the template shape
    must remain contiguous (no internal stride trickery that would force
    NCCL to allocate a staging buffer)."""
    from context_parallel.hstu_attn_cp import _get_recv_buffer

    template = torch.empty(128, 8, 64, dtype=torch.bfloat16)
    buf = _get_recv_buffer(template, slot="fwd_recv_k")
    assert buf.is_contiguous(), "recv buffer view must be contiguous for NCCL"
    assert buf.shape == template.shape
    assert buf.dtype == template.dtype
