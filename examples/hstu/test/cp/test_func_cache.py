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
    """Ensure each test starts and ends with no active cache scope.

    Without this, leakage between tests would mask cache-visibility bugs
    (e.g. test B reads test A's residual cache and passes accidentally).
    """
    from context_parallel import hstu_attn_cp as _cp

    _cp._cp_func_cache = None
    yield
    _cp._cp_func_cache = None


def test_no_scope_falls_through_no_caching(reset_cache_after_test):
    """When no scope is active, each call rebuilds the func tensor.

    Regression guard: it must be possible to call
    `_cached_localize_func_for_cp_step` outside any scope (e.g. in unit
    tests, or in code paths that haven't entered the scope yet) and get
    a correct result without crashing on a None lookup.
    """
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    kw = _make_kwargs(device=torch.device("cpu"))
    a = _cached_localize_func_for_cp_step(step=0, **kw)
    b = _cached_localize_func_for_cp_step(step=0, **kw)
    # Both correct, but DIFFERENT objects (no cache).
    assert torch.equal(a, b)
    assert a.data_ptr() != b.data_ptr(), (
        "outside scope, _cached_localize_func should fall through and "
        "build a fresh tensor on every call"
    )


def test_scope_dedups_across_main_thread_calls(reset_cache_after_test):
    """Inside a scope, repeated calls for the same step return the SAME tensor.

    This validates the layer-stack reuse path (8 layers calling for the
    same step within one HSTUBlock.forward).
    """
    from context_parallel import cp_func_cache_scope_enter, cp_func_cache_scope_exit
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    kw = _make_kwargs(device=torch.device("cpu"))

    cp_func_cache_scope_enter()
    try:
        first = _cached_localize_func_for_cp_step(step=0, **kw)
        second = _cached_localize_func_for_cp_step(step=0, **kw)
        third = _cached_localize_func_for_cp_step(step=1, **kw)
    finally:
        cp_func_cache_scope_exit()

    assert first is second, "same step in same scope must return cached tensor"
    assert first is not third, "different step keys must not collide"


def test_scope_visible_from_other_thread(reset_cache_after_test):
    """The cache populated on thread A must be visible to thread B.

    This is the ROOT BUG of cw-dfw round-4: PyTorch autograd backward
    runs on a worker thread, NOT the forward thread. With
    `threading.local()`, the worker thread sees an EMPTY cache and
    rebuilds every layer's func tensor (16 rebuilds × ~40 ms ≈ 640 ms
    of wasted Python+H2D per training step).

    With a module-level dict, the worker thread reads the same
    dict that the main thread populated.
    """
    from context_parallel import cp_func_cache_scope_enter, cp_func_cache_scope_exit
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    kw = _make_kwargs(device=torch.device("cpu"))

    cp_func_cache_scope_enter()
    try:
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
    finally:
        cp_func_cache_scope_exit()

    assert "t" in other, "worker thread did not return a tensor"
    assert other["t"] is main_tensor, (
        "cache populated by main thread must be visible to autograd "
        "worker thread — see commit message; this assertion fails with "
        "threading.local() and passes with module-level dict"
    )


def test_scope_enter_drops_previous_step_tensors(reset_cache_after_test):
    """`scope_enter()` MUST replace the dict so step N's tensors get
    GC'd before step N+1 builds new ones — otherwise we leak `cp_size`
    func tensors per training step indefinitely.
    """
    import weakref

    from context_parallel import cp_func_cache_scope_enter
    from context_parallel.hstu_attn_cp import _cached_localize_func_for_cp_step

    kw = _make_kwargs(device=torch.device("cpu"))

    cp_func_cache_scope_enter()
    step_n = _cached_localize_func_for_cp_step(step=0, **kw)
    ref = weakref.ref(step_n)
    # Drop the local reference, then start the next step.
    del step_n
    cp_func_cache_scope_enter()
    # The previous step's tensor must now be unreferenced.
    assert ref() is None, (
        "scope_enter() must drop the previous step's tensors so "
        "memory does not grow unboundedly across training steps"
    )


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
