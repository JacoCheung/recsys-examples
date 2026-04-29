# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for `context_parallel._mask_func.build_global_mask_func`.

Strategy: for a fixed (q, k, v, cu_seqlens) and a chosen mask spec,
compute the HSTU attention output two ways:

  A. Pass the discrete (num_contexts, num_targets, target_group_size,
     window_size) 4-tuple to `hstu_attn_varlen_func` (kernel handles
     mask internally).
  B. Translate the same 4-tuple to a `(B, 1, NFUNC, max_seqlen_q)`
     int32 `func` tensor, pass `func` to `hstu_attn_varlen_func` with
     structured-mask params disabled
     (`num_contexts=None, num_targets=None, target_group_size=1,
       window_size=(-1, -1)`).

Both calls must produce the same output (bit-exact at fp32, within
bf16 tolerance otherwise) — which proves the translator faithfully
encodes the kernel's mask logic.

Single-GPU only, auto-skipped under torchrun. Requires the FBGEMM
hstu kernel to be built with `HSTU_ARBITRARY_NFUNC=3` (see
`docs/cp/het_mask_design.md` open question A).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import pytest
import torch

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    pytest.skip(
        "test_mask_func.py is single-GPU-only; skipping under torchrun",
        allow_module_level=True,
    )

_HSTU_ROOT = Path(__file__).resolve().parents[2]
if str(_HSTU_ROOT) not in sys.path:
    sys.path.append(str(_HSTU_ROOT))


# Skip the whole module if the installed hstu kernel does not have
# arbitrary-mask support compiled in. This lets the test be cheap to
# discover on any install but only run on builds where it can succeed.
def _kernel_has_arbitrary() -> bool:
    try:
        from hstu import hstu_attn_varlen_func  # noqa: WPS433
    except Exception:
        return False
    device = torch.device("cuda:0") if torch.cuda.is_available() else None
    if device is None:
        return False
    try:
        S = 4
        q = torch.randn(S, 1, 32, dtype=torch.bfloat16, device=device)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        cu = torch.tensor([0, S], dtype=torch.int32, device=device)
        func = torch.zeros((1, 1, 3, S), dtype=torch.int32, device=device)
        for q_pos in range(S):
            func[0, 0, 0, q_pos] = q_pos + 1
        hstu_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=S,
            max_seqlen_k=S,
            scaling_seqlen=S,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, -1),
            alpha=1.0 / (32**0.5),
            func=func,
        )
        return True
    except Exception:
        return False


_HAS_ARBITRARY = _kernel_has_arbitrary()


@pytest.fixture(scope="module")
def cuda_device() -> Iterator[torch.device]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not _HAS_ARBITRARY:
        pytest.skip(
            "installed hstu kernel does not have arbitrary-mask support "
            "(rebuild with HSTU_ARBITRARY_NFUNC=3); see "
            "docs/cp/het_mask_design.md §4.A"
        )
    yield torch.device("cuda:0")


def _build_packed_inputs(
    seqlens: list[int],
    *,
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    g = torch.Generator(device=device).manual_seed(seed)
    cu = [0]
    for L in seqlens:
        cu.append(cu[-1] + L)
    total = cu[-1]
    q = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    return q, k, v, cu_t, max(seqlens)


def _kernel(
    q,
    k,
    v,
    cu,
    *,
    max_seqlen,
    num_contexts=None,
    num_targets=None,
    target_group_size=1,
    window_size=(-1, 0),
    func=None,
):
    from hstu import hstu_attn_varlen_func

    head_dim = q.shape[-1]
    return hstu_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        scaling_seqlen=max_seqlen,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
        window_size=window_size,
        alpha=1.0 / (head_dim**0.5),
        func=func,
    )


# ----------------------------------------------------------------------------
# 1. Pure causal — single interval [0, q+1), simplest sanity.
# ----------------------------------------------------------------------------
def test_pure_causal_translator_matches_4tuple(cuda_device: torch.device) -> None:
    from context_parallel._mask_func import build_global_mask_func

    seqlens = [16, 32]
    q, k, v, cu, max_s = _build_packed_inputs(
        seqlens, num_heads=2, head_dim=32, device=cuda_device, seed=0
    )
    out_4tuple = _kernel(q, k, v, cu, max_seqlen=max_s)  # default causal
    func = build_global_mask_func(
        cu_seqlens_q=cu,
        max_seqlen_q=max_s,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
    )
    out_func = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, -1),  # disable structured causal — `func` carries it
        func=func,
    )
    torch.testing.assert_close(out_func, out_4tuple, rtol=2e-2, atol=2e-2)


# ----------------------------------------------------------------------------
# 2. Sliding-causal with window_size=(w, 0).
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("w", [4, 8])
def test_sliding_causal_translator_matches_4tuple(
    cuda_device: torch.device, w: int
) -> None:
    from context_parallel._mask_func import build_global_mask_func

    seqlens = [32, 24]
    q, k, v, cu, max_s = _build_packed_inputs(
        seqlens, num_heads=2, head_dim=32, device=cuda_device, seed=1
    )
    out_4tuple = _kernel(q, k, v, cu, max_seqlen=max_s, window_size=(w, 0))
    func = build_global_mask_func(
        cu_seqlens_q=cu,
        max_seqlen_q=max_s,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(w, 0),
    )
    out_func = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        window_size=(-1, -1),
        func=func,
    )
    torch.testing.assert_close(out_func, out_4tuple, rtol=2e-2, atol=2e-2)


# ----------------------------------------------------------------------------
# 3. Targets with target_group_size=1 (each target attends to history + self only).
# ----------------------------------------------------------------------------
def test_targets_g1_translator_matches_4tuple(cuda_device: torch.device) -> None:
    from context_parallel._mask_func import build_global_mask_func

    seqlens = [16, 24]
    q, k, v, cu, max_s = _build_packed_inputs(
        seqlens, num_heads=2, head_dim=32, device=cuda_device, seed=2
    )
    num_targets = torch.tensor([4, 6], dtype=torch.int32, device=cuda_device)
    out_4tuple = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        num_targets=num_targets,
        target_group_size=1,
    )
    func = build_global_mask_func(
        cu_seqlens_q=cu,
        max_seqlen_q=max_s,
        num_contexts=None,
        num_targets=num_targets,
        target_group_size=1,
        window_size=(-1, 0),
    )
    out_func = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        window_size=(-1, -1),
        func=func,
    )
    torch.testing.assert_close(out_func, out_4tuple, rtol=2e-2, atol=2e-2)


# ----------------------------------------------------------------------------
# 4. Targets with target_group_size=2 (group-causal within target region).
# ----------------------------------------------------------------------------
def test_targets_g2_translator_matches_4tuple(cuda_device: torch.device) -> None:
    from context_parallel._mask_func import build_global_mask_func

    seqlens = [16, 16]  # multiples of 2 so target groups divide cleanly
    q, k, v, cu, max_s = _build_packed_inputs(
        seqlens, num_heads=2, head_dim=32, device=cuda_device, seed=3
    )
    num_targets = torch.tensor([6, 4], dtype=torch.int32, device=cuda_device)
    out_4tuple = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        num_targets=num_targets,
        target_group_size=2,
    )
    func = build_global_mask_func(
        cu_seqlens_q=cu,
        max_seqlen_q=max_s,
        num_contexts=None,
        num_targets=num_targets,
        target_group_size=2,
        window_size=(-1, 0),
    )
    out_func = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        window_size=(-1, -1),
        func=func,
    )
    torch.testing.assert_close(out_func, out_4tuple, rtol=2e-2, atol=2e-2)


# ----------------------------------------------------------------------------
# 5. Contextual prefix with no targets.
# ----------------------------------------------------------------------------
def test_contextual_prefix_translator_matches_4tuple(
    cuda_device: torch.device,
) -> None:
    from context_parallel._mask_func import build_global_mask_func

    seqlens = [16, 24]
    q, k, v, cu, max_s = _build_packed_inputs(
        seqlens, num_heads=2, head_dim=32, device=cuda_device, seed=4
    )
    num_contexts = torch.tensor([3, 5], dtype=torch.int32, device=cuda_device)
    out_4tuple = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        num_contexts=num_contexts,
    )
    func = build_global_mask_func(
        cu_seqlens_q=cu,
        max_seqlen_q=max_s,
        num_contexts=num_contexts,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
    )
    out_func = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        window_size=(-1, -1),
        func=func,
    )
    torch.testing.assert_close(out_func, out_4tuple, rtol=2e-2, atol=2e-2)


# ----------------------------------------------------------------------------
# 6. Combined: contextual + targets + group_size=2. Worst-case 2-interval mask.
# ----------------------------------------------------------------------------
def test_full_combination_translator_matches_4tuple(
    cuda_device: torch.device,
) -> None:
    from context_parallel._mask_func import build_global_mask_func

    seqlens = [24, 32]
    q, k, v, cu, max_s = _build_packed_inputs(
        seqlens, num_heads=2, head_dim=32, device=cuda_device, seed=5
    )
    num_contexts = torch.tensor([3, 5], dtype=torch.int32, device=cuda_device)
    num_targets = torch.tensor([6, 8], dtype=torch.int32, device=cuda_device)
    out_4tuple = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=2,
    )
    func = build_global_mask_func(
        cu_seqlens_q=cu,
        max_seqlen_q=max_s,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=2,
        window_size=(-1, 0),
    )
    out_func = _kernel(
        q,
        k,
        v,
        cu,
        max_seqlen=max_s,
        window_size=(-1, -1),
        func=func,
    )
    torch.testing.assert_close(out_func, out_4tuple, rtol=2e-2, atol=2e-2)


# ============================================================================
# Per-step CP localiser tests (no kernel call; pure index-arithmetic checks).
# ============================================================================
def _decode_func_to_bool_mask(
    func: torch.Tensor, *, total_q: int, total_k: int
) -> torch.Tensor:
    """Decode a (1, NFUNC, total_q) func tensor back into a
    (total_q, total_k) boolean allowed-mask. Used by tests to verify
    the localiser packs the right intervals."""
    NFUNC = func.shape[1]
    n_pair = NFUNC // 2
    mask = torch.zeros((total_q, total_k), dtype=torch.bool, device=func.device)
    for q in range(total_q):
        # First interval: [0, slot_0)
        slot_0 = int(func[0, 0, q].item())
        if slot_0 > 0:
            mask[q, : min(slot_0, total_k)] = True
        # Subsequent pairs: (slot_{2k-1}, slot_{2k}) → interval [lo, hi)
        for pair in range(1, n_pair + 1):
            lo = int(func[0, 2 * pair - 1, q].item())
            hi = int(func[0, 2 * pair, q].item())
            if lo < hi:
                mask[q, lo : min(hi, total_k)] = True
    return mask


def _global_mask_for_sample(
    *, L: int, nc: int, nt: int, g: int, window_size: tuple[int, int]
) -> torch.Tensor:
    """Brute-force build the (L, L) bool mask for one sample using the
    analytical predicate. Used as the oracle for both the global builder
    and the per-step localiser."""
    from context_parallel._mask_func import _per_sample_intervals

    intervals = _per_sample_intervals(
        L=L,
        nc=nc,
        nt=nt,
        g=g,
        w_left=window_size[0],
        w_right=window_size[1],
    )
    mask = torch.zeros((L, L), dtype=torch.bool)
    for q in range(L):
        for lo, hi in intervals[q]:
            mask[q, lo:hi] = True
    return mask


@pytest.mark.parametrize("cp_size", [2, 4])
def test_localiser_partitions_global_mask(
    cuda_device: torch.device, cp_size: int
) -> None:
    """For a single sample of length L = 4 * 2 * cp_size, build the
    global mask via the analytical predicate, then for every (rank,
    step) build the localised func, decode it back to a local bool
    mask, and verify it equals the corresponding sub-block of the
    global mask gathered with DualChunkSwap indices."""
    from context_parallel._mask_func import localize_func_for_cp_step

    chunks_per_seq = 2 * cp_size
    c = 4
    L = c * chunks_per_seq
    cu_global = torch.tensor([0, L], dtype=torch.int32, device=cuda_device)
    nc, nt, g = 2, 4, 2
    window_size = (-1, 0)
    num_contexts = torch.tensor([nc], dtype=torch.int32, device=cuda_device)
    num_targets = torch.tensor([nt], dtype=torch.int32, device=cuda_device)

    # Oracle: global (L, L) mask.
    global_mask = _global_mask_for_sample(
        L=L, nc=nc, nt=nt, g=g, window_size=window_size
    )

    for cp_rank in range(cp_size):
        for step in range(cp_size):
            peer = (cp_rank - step) % cp_size
            own_q_chunks = (cp_rank, chunks_per_seq - 1 - cp_rank)
            own_k_chunks = (peer, chunks_per_seq - 1 - peer)
            # Build l2g_q and l2g_k for this step.
            l2g_q = list(range(own_q_chunks[0] * c, (own_q_chunks[0] + 1) * c)) + list(
                range(own_q_chunks[1] * c, (own_q_chunks[1] + 1) * c)
            )
            l2g_k = list(range(own_k_chunks[0] * c, (own_k_chunks[0] + 1) * c)) + list(
                range(own_k_chunks[1] * c, (own_k_chunks[1] + 1) * c)
            )
            l2g_q_t = torch.tensor(l2g_q, dtype=torch.int64, device=cuda_device)
            l2g_k_t = torch.tensor(l2g_k, dtype=torch.int64, device=cuda_device)
            # Oracle local mask: gather global_mask rows by l2g_q, cols by l2g_k.
            oracle_local = global_mask[l2g_q_t.cpu()][:, l2g_k_t.cpu()]

            func = localize_func_for_cp_step(
                cu_seqlens_global=cu_global,
                cp_size=cp_size,
                cp_rank=cp_rank,
                step=step,
                num_contexts=num_contexts,
                num_targets=num_targets,
                target_group_size=g,
                window_size=window_size,
                NFUNC=3,
            )
            decoded = _decode_func_to_bool_mask(func, total_q=2 * c, total_k=2 * c)
            assert torch.equal(decoded.cpu(), oracle_local), (
                f"cp_size={cp_size} cp_rank={cp_rank} step={step}: "
                f"localiser mask differs from oracle"
            )


def test_localiser_step_zero_matches_global_self_chunks(
    cuda_device: torch.device,
) -> None:
    """Sanity: at step=0 (peer=cp_rank), the local Q × local K layout
    mirrors the diagonal `[chunk_r; chunk_(2cp-1-r)]` × itself sub-block
    of the global mask."""
    from context_parallel._mask_func import localize_func_for_cp_step

    cp_size = 4
    c = 4
    L = c * 2 * cp_size  # = 32
    cu_global = torch.tensor([0, L], dtype=torch.int32, device=cuda_device)
    global_mask = _global_mask_for_sample(L=L, nc=0, nt=0, g=1, window_size=(-1, 0))
    for cp_rank in range(cp_size):
        chunks = (cp_rank, 2 * cp_size - 1 - cp_rank)
        l2g = list(range(chunks[0] * c, (chunks[0] + 1) * c)) + list(
            range(chunks[1] * c, (chunks[1] + 1) * c)
        )
        l2g_t = torch.tensor(l2g, dtype=torch.int64)
        oracle = global_mask[l2g_t][:, l2g_t]
        func = localize_func_for_cp_step(
            cu_seqlens_global=cu_global,
            cp_size=cp_size,
            cp_rank=cp_rank,
            step=0,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, 0),
            NFUNC=3,
        )
        decoded = _decode_func_to_bool_mask(func, total_q=2 * c, total_k=2 * c)
        assert torch.equal(decoded.cpu(), oracle), f"step=0 cp_rank={cp_rank}"


def test_intervals_to_slots_nfunc_overflow_raises(
    cuda_device: torch.device,
) -> None:
    """If a Q row needs more disjoint intervals than NFUNC capacity,
    `_intervals_to_slots` must raise rather than silently truncating —
    truncation would silently drop in-window K cells from attention."""
    from context_parallel._mask_func import _intervals_to_slots

    # NFUNC=3 → max 2 disjoint intervals.  Three intervals → must raise.
    too_many = [(0, 1), (3, 5), (8, 10)]
    with pytest.raises(ValueError, match="NFUNC"):
        _intervals_to_slots(too_many, NFUNC=3)


def test_localiser_rejects_invalid_nc_nt_split(cuda_device: torch.device) -> None:
    """`localize_func_for_cp_step` must raise when nc + nt > L for any
    sample (Codex round-2 IMPORTANT regression-guard).

    The vectorised predicate (`_per_sample_intervals_array`) writes
    contextual rows AND target rows independently using `q < nc` /
    `q >= L - nt` masks. When `nc + nt > L` those masks overlap and
    target writes silently overwrite contextual rows, diverging from
    the scalar reference (`_per_sample_intervals`, which has an
    explicit `if q < nc: ... continue` guard). The localiser must
    reject malformed splits up-front rather than build a divergent
    mask. Mirrors the same invariant `build_global_mask_func`
    enforces.
    """
    from context_parallel._mask_func import localize_func_for_cp_step

    # cp_size=2, L=16: nc=10, nt=10 → nc+nt=20 > L=16 → must raise.
    cu = torch.tensor([0, 16], dtype=torch.int32, device=cuda_device)
    num_contexts = torch.tensor([10], dtype=torch.int32, device=cuda_device)
    num_targets = torch.tensor([10], dtype=torch.int32, device=cuda_device)
    with pytest.raises(ValueError, match="invalid heterogeneous mask split"):
        localize_func_for_cp_step(
            cu_seqlens_global=cu,
            cp_size=2,
            cp_rank=0,
            step=0,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=1,
            window_size=(-1, 0),
            NFUNC=3,
            device=cuda_device,
        )

    # Negative nc must also raise.
    bad_nc = torch.tensor([-1], dtype=torch.int32, device=cuda_device)
    with pytest.raises(ValueError, match="invalid heterogeneous mask split"):
        localize_func_for_cp_step(
            cu_seqlens_global=cu,
            cp_size=2,
            cp_rank=0,
            step=0,
            num_contexts=bad_nc,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, 0),
            NFUNC=3,
            device=cuda_device,
        )


def test_global_builder_nfunc_overflow_raises(cuda_device: torch.device) -> None:
    """If the global mask requires more intervals than NFUNC supports,
    `build_global_mask_func` must raise.  Constructed by deliberately
    passing NFUNC=1 (single implicit interval [0, slot_0)) with target Q
    rows that require 2 disjoint intervals (history-band +
    target-group-band)."""
    from context_parallel._mask_func import build_global_mask_func

    cu = torch.tensor([0, 16], dtype=torch.int32, device=cuda_device)
    num_targets = torch.tensor([4], dtype=torch.int32, device=cuda_device)
    # Target rows q ∈ {12, 13, 14, 15} (last 4 of L=16, target_group_size=1).
    # `_per_sample_intervals` produces:
    #   q=12: [(0, 12)] then [(12, 13)] — adjacent, merges to [(0, 13)] (1 interval, fits NFUNC=1).
    #   q=13: [(0, 12)] then [(13, 14)] — non-adjacent, stays 2 disjoint intervals.
    # So the overflow fires at q=13 (or any later target row), not q=12.
    with pytest.raises(ValueError, match="NFUNC"):
        build_global_mask_func(
            cu_seqlens_q=cu,
            max_seqlen_q=16,
            num_contexts=None,
            num_targets=num_targets,
            target_group_size=1,
            window_size=(-1, 0),
            NFUNC=1,
        )


def test_localiser_union_across_steps_equals_full_q_row(
    cuda_device: torch.device,
) -> None:
    """For each Q row, the union of allowed K positions across all ring
    steps must equal the global allowed K set for that Q row."""
    from context_parallel._mask_func import localize_func_for_cp_step

    cp_size = 4
    c = 4
    L = c * 2 * cp_size
    cu_global = torch.tensor([0, L], dtype=torch.int32, device=cuda_device)
    nc, nt, g = 2, 4, 2
    num_contexts = torch.tensor([nc], dtype=torch.int32, device=cuda_device)
    num_targets = torch.tensor([nt], dtype=torch.int32, device=cuda_device)

    global_mask = _global_mask_for_sample(L=L, nc=nc, nt=nt, g=g, window_size=(-1, 0))

    cp_rank = 1
    own_q_chunks = (cp_rank, 2 * cp_size - 1 - cp_rank)
    l2g_q = list(range(own_q_chunks[0] * c, (own_q_chunks[0] + 1) * c)) + list(
        range(own_q_chunks[1] * c, (own_q_chunks[1] + 1) * c)
    )
    l2g_q_t = torch.tensor(l2g_q, dtype=torch.int64)

    # Union mask per Q row across all steps, accumulated in global K coords.
    union = torch.zeros((2 * c, L), dtype=torch.bool)
    for step in range(cp_size):
        peer = (cp_rank - step) % cp_size
        own_k_chunks = (peer, 2 * cp_size - 1 - peer)
        l2g_k = list(range(own_k_chunks[0] * c, (own_k_chunks[0] + 1) * c)) + list(
            range(own_k_chunks[1] * c, (own_k_chunks[1] + 1) * c)
        )
        func = localize_func_for_cp_step(
            cu_seqlens_global=cu_global,
            cp_size=cp_size,
            cp_rank=cp_rank,
            step=step,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=g,
            window_size=(-1, 0),
            NFUNC=3,
        )
        decoded_local = _decode_func_to_bool_mask(func, total_q=2 * c, total_k=2 * c)
        # Map local K → global K via l2g_k.
        for k_local, k_global in enumerate(l2g_k):
            union[:, k_global] |= decoded_local[:, k_local].cpu()

    oracle_q_rows = global_mask[l2g_q_t]
    assert torch.equal(union, oracle_q_rows)
