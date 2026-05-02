# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Regression tests for the per-training-step `func` tensor cache used by
`_multi_gpu_forward_arbitrary` / `_multi_gpu_backward_arbitrary`.

The cache MUST be process-global (not thread-local). PyTorch autograd
runs `.backward()` on a worker thread, so a `threading.local()` cache
populated on the forward thread is invisible from the backward thread —
which produces the exact perf regression we saw on cw-dfw round-4
(forward cached → 16 backward rebuilds → 734 ms/step instead of ~70 ms).

These tests would fail against a `threading.local()` implementation and
pass against a module-level dict.

Single-process, no torchrun, no GPU strictly required (we only need a
device for the func tensor; CPU is fine for `localize_func_for_cp_step`).
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import pytest
import torch

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    pytest.skip(
        "test_func_cache.py is single-process; skipping under torchrun",
        allow_module_level=True,
    )

_HSTU_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = Path(__file__).resolve().parents[3]
for p in (_HSTU_ROOT, _EXAMPLES_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))


def _make_kwargs(*, device: torch.device) -> dict:
    """Minimal valid kwargs for `localize_func_for_cp_step`.

    Two samples of length 8 each, cp_size=2, cp_rank=0, no contextual /
    target prefix. Causal-only, NFUNC=3.
    """
    cu_seqlens_global = torch.tensor([0, 8, 16], dtype=torch.int32, device=device)
    return dict(
        cu_seqlens_global=cu_seqlens_global,
        cp_size=2,
        cp_rank=0,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        NFUNC=3,
        device=device,
    )


@pytest.fixture
def reset_cache_after_test():
    """Ensure each test starts and ends with an empty cache.

    Without this, leakage between tests would mask cache-visibility bugs
    (e.g. test B reads test A's residual cache and passes accidentally).
    """
    from context_parallel import cp_func_cache_scope_enter

    cp_func_cache_scope_enter()  # clear before
    yield
    cp_func_cache_scope_enter()  # clear after


def test_dedups_across_repeated_calls(reset_cache_after_test):
    """Repeated calls with the SAME args return the SAME tensor.

    Validates both the layer-stack reuse path (8 layers calling for the
    same step within one HSTUBlock.forward) AND the cross-iteration
    reuse (training step N+k calling with the same cu_seqlens as step N).
    Both fold to the same cache hit.
    """
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    kw = _make_kwargs(device=torch.device("cpu"))

    first = _cached_localize_func_for_cp_step(step=0, **kw)
    second = _cached_localize_func_for_cp_step(step=0, **kw)
    third = _cached_localize_func_for_cp_step(step=1, **kw)

    assert first is second, "same kwargs must return cached tensor"
    assert first is not third, "different `step` must not collide"


def test_cache_visible_from_other_thread(reset_cache_after_test):
    """The cache populated on thread A must be visible to thread B.

    This is the ROOT BUG of cw-dfw round-4: PyTorch autograd backward
    runs on a worker thread, NOT the forward thread. With
    `threading.local()`, the worker thread sees an EMPTY cache and
    rebuilds every layer's func tensor (16 rebuilds × ~40 ms ≈ 640 ms
    of wasted Python+H2D per training step).

    With a module-level dict, the worker thread reads the same
    dict that the main thread populated.
    """
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    kw = _make_kwargs(device=torch.device("cpu"))

    # Populate from MAIN thread (mirrors HSTUBlock.forward).
    main_tensor = _cached_localize_func_for_cp_step(step=0, **kw)

    # Read from a DIFFERENT thread (mirrors autograd worker thread
    # running the backward `apply()` of the autograd.Function).
    other: dict[str, torch.Tensor] = {}

    def worker():
        other["t"] = _cached_localize_func_for_cp_step(step=0, **kw)

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert "t" in other, "worker thread did not return a tensor"
    assert other["t"] is main_tensor, (
        "cache populated by main thread must be visible to autograd "
        "worker thread — see commit message; this assertion fails with "
        "threading.local() and passes with module-level dict"
    )


def test_cache_keys_on_cu_seqlens_content(reset_cache_after_test):
    """Two distinct cu_seqlens values produce distinct cache entries.

    Cross-iteration caching requires content-keyed lookup. If we only
    keyed on `step`, the cache would silently return a STALE func built
    against a different batch's cu_seqlens — wrong mask = wrong grads
    = silent training corruption. This is the most important regression
    guard for the cross-step caching change.
    """
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    cu_a = torch.tensor([0, 8, 16], dtype=torch.int32, device="cpu")
    cu_b = torch.tensor([0, 4, 16], dtype=torch.int32, device="cpu")  # 4+12 split
    common_kw = dict(
        cp_size=2,
        cp_rank=0,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        NFUNC=3,
        device=torch.device("cpu"),
    )

    func_a = _cached_localize_func_for_cp_step(
        step=0, cu_seqlens_global=cu_a, **common_kw
    )
    func_b = _cached_localize_func_for_cp_step(
        step=0, cu_seqlens_global=cu_b, **common_kw
    )
    func_a_again = _cached_localize_func_for_cp_step(
        step=0, cu_seqlens_global=cu_a, **common_kw
    )

    assert func_a is not func_b, (
        "different cu_seqlens MUST produce different cache entries — "
        "otherwise we'd return a stale func and silently corrupt grads"
    )
    assert func_a_again is func_a, (
        "same cu_seqlens content (even via a fresh tensor object) must "
        "hit the cache — this is the cross-iteration win"
    )


def test_caller_provided_tuple_keys_skip_internal_sync(reset_cache_after_test):
    """Caller fast-path: pre-computed tuple keys skip the internal
    `_hash_int_tensor_or_none` GPU→CPU sync.

    The caller's contract is: `cu_seqlens_global_tuple` MUST equal
    `tuple(cu_seqlens_global.tolist())`. When provided, the cache key
    is built from the tuple alone — `cu_seqlens_global` itself is
    only forwarded to `localize_func_for_cp_step` for the actual
    build (cold path), never hashed.

    The behavioural test: two calls with the SAME `cu_seqlens_global`
    tensor but DIFFERENT `cu_seqlens_global_tuple` values must hit
    DIFFERENT cache entries — proves the tuple drives the key.
    Two calls with different tensors but the SAME tuple must hit the
    SAME entry — proves the tensor doesn't sneak into the key.
    """
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    cu = torch.tensor([0, 8, 16], dtype=torch.int32, device="cpu")
    common_kw = dict(
        cp_size=2,
        cp_rank=0,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        NFUNC=3,
        device=torch.device("cpu"),
    )

    a = _cached_localize_func_for_cp_step(
        step=0,
        cu_seqlens_global=cu,
        cu_seqlens_global_tuple=(0, 8, 16),
        **common_kw,
    )
    b = _cached_localize_func_for_cp_step(
        step=0,
        cu_seqlens_global=cu,
        cu_seqlens_global_tuple=(0, 4, 16),  # different tuple
        **common_kw,
    )
    assert a is not b, (
        "different `cu_seqlens_global_tuple` MUST produce distinct "
        "cache entries — caller's tuple drives the key"
    )

    # Different fresh tensor, same tuple → cache hit on first key.
    cu2 = torch.tensor([99, 99, 99], dtype=torch.int32, device="cpu")
    a_again = _cached_localize_func_for_cp_step(
        step=0,
        cu_seqlens_global=cu2,
        cu_seqlens_global_tuple=(0, 8, 16),
        **common_kw,
    )
    assert a_again is a, (
        "same `cu_seqlens_global_tuple` MUST hit the same entry, "
        "regardless of which tensor was passed in"
    )


def test_cache_caps_growth(reset_cache_after_test):
    """The cache must evict when it grows beyond `_CP_FUNC_CACHE_MAX`.

    Bounding cache size matters for production workloads with truly
    random per-step batches (memory leak otherwise).
    """
    from context_parallel import hstu_attn_cp as _cp
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    common_kw = dict(
        cp_size=2,
        cp_rank=0,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        NFUNC=3,
        device=torch.device("cpu"),
    )

    saved_max = _cp._CP_FUNC_CACHE_MAX
    _cp._CP_FUNC_CACHE_MAX = 3
    try:
        for k in range(5):
            cu = torch.tensor([0, 4, 8 + k], dtype=torch.int32, device="cpu")
            _cached_localize_func_for_cp_step(step=0, cu_seqlens_global=cu, **common_kw)
        assert len(_cp._cp_func_cache) <= 3, (
            f"cache must not exceed _CP_FUNC_CACHE_MAX={3}; "
            f"got {len(_cp._cp_func_cache)} entries"
        )
    finally:
        _cp._CP_FUNC_CACHE_MAX = saved_max


def test_scope_exit_keeps_cache_alive_for_backward(reset_cache_after_test):
    """`scope_exit()` MUST be a no-op (or leave the cache populated).

    The autograd backward runs AFTER HSTUBlock.forward returns. If
    scope_exit() cleared the dict, backward would see an empty cache
    and rebuild all 16 funcs (the cw-dfw round-4 regression).
    """
    from context_parallel import cp_func_cache_scope_enter, cp_func_cache_scope_exit
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    kw = _make_kwargs(device=torch.device("cpu"))
    cp_func_cache_scope_enter()
    fwd = _cached_localize_func_for_cp_step(step=0, **kw)
    cp_func_cache_scope_exit()
    # Mimic a backward call AFTER forward's scope_exit.
    bwd = _cached_localize_func_for_cp_step(step=0, **kw)
    assert bwd is fwd, (
        "scope_exit must leave the cache populated for backward; "
        "clearing it would re-introduce the round-4 regression"
    )
