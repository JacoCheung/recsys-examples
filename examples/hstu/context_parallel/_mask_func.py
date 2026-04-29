# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HSTU heterogeneous-mask → arbitrary-mask `func` tensor translator.

Tracks `docs/cp/het_mask_design.md` Step 2 — the cp_size==1 path
forwards the discrete `(num_contexts, num_targets, target_group_size,
window_size)` 4-tuple to the kernel which handles them natively, but
the cp_size>1 ring needs to express the same mask using the kernel's
arbitrary-mask facility (`func` tensor) so each ring step can encode
its peer-K-restricted view of the global predicate.

This module provides the analytical translator that produces a
`(B, 1, NFUNC, max_seqlen_q)` int32 tensor encoding the same per-Q-row
K interval that the FBGEMM kernel would compute internally given the
4-tuple. Verified against the PT reference at
`examples/hstu/ops/pt_ops/pt_hstu_attention.py::_get_valid_attn_mask`.

Encoding (matches FBGEMM
`third_party/FBGEMM/fbgemm_gpu/experimental/hstu/src/hstu_ampere/hstu_fwd.h:155-168`):

  - Slot 0: MaxFunc[0] = upper of interval 0 → keeps `[0, slot_0)`
  - Slot 1: MinFunc[0] = lower of interval 1
  - Slot 2: MaxFunc[1] = upper of interval 1 → keeps `[slot_1, slot_2)`
  - Slot 3: MinFunc[1] = lower of interval 2 (NFUNC ≥ 5 only)
  - ...

NFUNC must be odd. NFUNC=3 supports up to 2 disjoint intervals per
row, which suffices for HSTU's standard mask families
(history-causal + target-group; see `docs/cp/het_mask_design.md` §1).

This module is **internal** to the CP wrapper; it imports nothing from
the training stack and has no torch.distributed dependencies.
"""

from __future__ import annotations

from typing import Optional

import torch


def _per_sample_intervals(
    *,
    L: int,
    nc: int,
    nt: int,
    g: int,
    w_left: int,
    w_right: int,
) -> list[list[tuple[int, int]]]:
    """Compute the allowed-K interval list for each Q row of one sample.

    Returns a list of length L; entry q is a list of (lo_inclusive,
    hi_exclusive) intervals encoding the K positions allowed to
    contribute to Q[q]. This is the readable scalar form, used by
    tests as the oracle. The hot path uses
    `_per_sample_intervals_array` (numpy-vectorised, same predicate)
    which is functionally identical but ~100× faster.

    Reproduces the predicate from
    `examples/hstu/ops/pt_ops/pt_hstu_attention.py::_get_valid_attn_mask`.
    `(w_left, w_right)` follows the kernel's `window_size` convention:
    `(-1, 0)` ⇒ pure causal (no sliding); `w_left ≥ 0` ⇒
    sliding-causal with span `w_left`. We currently support
    `w_right == 0` only; `w_right > 0` (lookahead) is rejected before
    reaching this function.
    """
    is_sliding = w_left >= 0
    is_causal = w_right == 0
    history_end = L - nt
    target_start = L - nt

    out: list[list[tuple[int, int]]] = []
    for q in range(L):
        causal_hi = q + 1 if is_causal else L
        sliding_lo = max(0, q - w_left) if is_sliding and is_causal else 0

        if q < nc:
            lo = sliding_lo if (is_sliding and is_causal) else 0
            hi = min(history_end, L)
            out.append([(lo, hi)] if lo < hi else [])
            continue

        if q < history_end:
            hi = min(causal_hi, history_end)
            lo = sliding_lo
            out.append([(lo, hi)] if lo < hi else [])
            continue

        # Target Q: q ∈ [target_start, L).
        group_q = (q - target_start) // g
        target_group_lo = target_start + group_q * g
        target_lo = (
            max(target_group_lo, sliding_lo)
            if is_sliding and is_causal
            else target_group_lo
        )
        target_hi = min(causal_hi, L) if is_causal else L
        history_for_target_lo = sliding_lo
        history_for_target_hi = history_end

        intervals: list[tuple[int, int]] = []
        if history_for_target_lo < history_for_target_hi:
            intervals.append((history_for_target_lo, history_for_target_hi))
        if target_lo < target_hi:
            if intervals and intervals[-1][1] == target_lo:
                intervals[-1] = (intervals[-1][0], target_hi)
            else:
                intervals.append((target_lo, target_hi))
        out.append(intervals)

    return out


def _per_sample_intervals_array(
    *,
    L: int,
    nc: int,
    nt: int,
    g: int,
    w_left: int,
    w_right: int,
):
    """Numpy-vectorised version of `_per_sample_intervals`.

    Returns a `(L, 2, 2)` int32 array where entry `[q, k, :]` is the
    `(lo, hi)` of the k-th interval for Q row q. Empty intervals are
    encoded as `(0, 0)` (the kernel's `[lo, hi)` semantics make any
    `lo == hi` interval a no-op). This 2-interval cap matches HSTU's
    standard mask families (history-band + target-group-band); if a
    future predicate produces > 2 intervals per row, this representation
    must be widened.
    """
    import numpy as np

    is_sliding = w_left >= 0
    is_causal = w_right == 0
    history_end = L - nt

    q = np.arange(L, dtype=np.int64)
    causal_hi = (q + 1) if is_causal else np.full(L, L, dtype=np.int64)
    sliding_lo = (
        np.maximum(0, q - w_left)
        if (is_sliding and is_causal)
        else np.zeros(L, dtype=np.int64)
    )

    intervals = np.zeros((L, 2, 2), dtype=np.int32)

    # Region masks.
    is_contextual = q < nc
    is_target = q >= history_end
    is_history = ~is_contextual & ~is_target

    # Contextual Q: single interval [lo, history_end).
    ctx_lo = sliding_lo if (is_sliding and is_causal) else np.zeros(L, dtype=np.int64)
    ctx_hi = np.full(L, history_end, dtype=np.int64)
    ctx_valid = ctx_lo < ctx_hi
    sel = is_contextual & ctx_valid
    intervals[sel, 0, 0] = ctx_lo[sel]
    intervals[sel, 0, 1] = ctx_hi[sel]

    # History Q: single interval [sliding_lo, min(q+1, history_end)).
    hist_lo = sliding_lo
    hist_hi = np.minimum(causal_hi, history_end)
    hist_valid = hist_lo < hist_hi
    sel = is_history & hist_valid
    intervals[sel, 0, 0] = hist_lo[sel]
    intervals[sel, 0, 1] = hist_hi[sel]

    # Target Q: two intervals.
    target_start = history_end
    group_q = (q - target_start) // g  # only defined for q >= target_start
    target_group_lo_full = target_start + group_q * g
    target_lo = (
        np.maximum(target_group_lo_full, sliding_lo)
        if is_sliding and is_causal
        else target_group_lo_full
    )
    target_hi = np.minimum(causal_hi, L) if is_causal else np.full(L, L, dtype=np.int64)
    history_band_lo = sliding_lo
    history_band_hi = np.full(L, history_end, dtype=np.int64)

    # Target Q with non-empty history band.
    history_valid = is_target & (history_band_lo < history_band_hi)
    intervals[history_valid, 0, 0] = history_band_lo[history_valid]
    intervals[history_valid, 0, 1] = history_band_hi[history_valid]

    # Target Q with non-empty target band.
    target_valid = is_target & (target_lo < target_hi)
    # Merge with the history band when adjacent (history_band_hi ==
    # target_lo). When merged, just extend the first interval's upper
    # to target_hi and keep the second slot empty.
    can_merge = history_valid & target_valid & (history_band_hi == target_lo)
    intervals[can_merge, 0, 1] = target_hi[can_merge]

    # Non-mergeable target band → goes into slot 1.
    needs_slot1 = target_valid & ~can_merge
    # Two sub-cases: history band was non-empty (slot 0 already used) →
    # target band goes to slot 1. History band was empty → target band
    # goes to slot 0.
    has_hist0 = needs_slot1 & history_valid
    no_hist0 = needs_slot1 & ~history_valid
    intervals[has_hist0, 1, 0] = target_lo[has_hist0]
    intervals[has_hist0, 1, 1] = target_hi[has_hist0]
    intervals[no_hist0, 0, 0] = target_lo[no_hist0]
    intervals[no_hist0, 0, 1] = target_hi[no_hist0]

    return intervals


def _intervals_to_slots(intervals: list[tuple[int, int]], NFUNC: int) -> list[int]:
    """Pack a list of (lo, hi) intervals into a length-NFUNC slot list.

    Encoding: slot[0] = upper of interval 0 (implicit lower = 0).
    Pair (slot[2k-1], slot[2k]) = (lower, upper) of interval k for
    k ≥ 1.  When the first interval's `lo > 0`, slot[0] is set to 0
    (empty implicit interval) and the first interval is encoded via
    the (slot[1], slot[2]) pair, leaving (NFUNC // 2 + 1 - 1) more
    interval slots available.
    """
    slots = [0] * NFUNC
    if not intervals:
        return slots
    lo0, hi0 = intervals[0]
    if lo0 > 0:
        if NFUNC < 3:
            raise ValueError(
                f"first interval starts at lo0={lo0}>0 requires NFUNC≥3, "
                f"got {NFUNC}"
            )
        slots[0] = 0
        slots[1] = lo0
        slots[2] = hi0
        next_pair = 1  # pair 1 already used; next is pair 2 → slots 3, 4
    else:
        slots[0] = hi0
        next_pair = 0  # next is pair 1 → slots 1, 2
    for lo, hi in intervals[1:]:
        next_pair += 1
        slot_lo = 2 * next_pair - 1
        slot_hi = 2 * next_pair
        if slot_hi >= NFUNC:
            raise ValueError(
                f"ran out of NFUNC slots (NFUNC={NFUNC}); needs " f"≥ {slot_hi + 1}"
            )
        slots[slot_lo] = lo
        slots[slot_hi] = hi
    return slots


def _intersect_and_merge_to_local(
    global_intervals: list[tuple[int, int]],
    *,
    chunk_first: tuple[int, int],
    chunk_second: tuple[int, int],
    c: int,
) -> list[tuple[int, int]]:
    """Project global K intervals onto a CP-step's local K layout.

    Local K layout per sample is `[chunk_first; chunk_second]` (two
    chunks of size `c` concatenated).  Each global interval gets
    intersected with each chunk and remapped into local K coordinates;
    adjacent runs are merged.

    Args:
        global_intervals: list of (lo, hi) global K intervals.
        chunk_first: (lo, hi) global K range of the first local chunk.
        chunk_second: (lo, hi) global K range of the second local chunk.
        c: per-chunk size (each chunk has length `c`; first half of
            local K is chunk_first → [0, c), second half is chunk_second
            → [c, 2c)).

    Returns:
        list of (lo, hi) local K intervals in [0, 2c) range.
    """
    local: list[tuple[int, int]] = []
    cf_lo, cf_hi = chunk_first
    cs_lo, cs_hi = chunk_second
    for g_lo, g_hi in global_intervals:
        if g_lo >= g_hi:
            continue
        # Intersect with chunk_first → maps to local [0, c).
        i1_lo = max(g_lo, cf_lo)
        i1_hi = min(g_hi, cf_hi)
        if i1_lo < i1_hi:
            local.append((i1_lo - cf_lo, i1_hi - cf_lo))
        # Intersect with chunk_second → maps to local [c, 2c).
        i2_lo = max(g_lo, cs_lo)
        i2_hi = min(g_hi, cs_hi)
        if i2_lo < i2_hi:
            local.append((c + i2_lo - cs_lo, c + i2_hi - cs_lo))
    # Sort and merge adjacent/overlapping local intervals.
    local.sort()
    merged: list[tuple[int, int]] = []
    for lo, hi in local:
        if merged and merged[-1][1] >= lo:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def localize_func_for_cp_step(
    *,
    cu_seqlens_global: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    step: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
    NFUNC: int = 3,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build the local-Q × peer-K `func` tensor for one CP ring step.

    DualChunkSwap layout:
        - Each sample of global length L_b has chunks of size
          `c_b = L_b / (2 * cp_size)`.
        - Rank `r` owns Q chunks `{r, 2*cp_size - 1 - r}` per sample,
          stored locally as `[chunk_r; chunk_(2cp-1-r)]`.
        - At ring step `s`, this rank's K is the peer rank's chunks,
          where `peer = (cp_rank - s) % cp_size` (forward direction).
        - Local Q layout: `[chunk_(cp_rank); chunk_(2cp-1-cp_rank)]`.
        - Local K layout: `[chunk_(peer); chunk_(2cp-1-peer)]`.

    For each (sample b, local_q_pos), compute the global Q index, the
    set of allowed global K positions via `_per_sample_intervals`, then
    project onto the local K layout (chunk_peer + chunk_(2cp-1-peer))
    and pack into NFUNC slots.

    Returns: int32 `(1, NFUNC, total_local_q)` jagged-along-total_local_q
    matching the `build_global_mask_func` layout convention.
    """
    if cp_size < 1:
        raise ValueError(f"cp_size must be ≥ 1; got {cp_size}")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}); got {cp_rank}")
    if not 0 <= step < cp_size:
        raise ValueError(f"step must be in [0, {cp_size}); got {step}")
    if NFUNC < 1 or NFUNC % 2 == 0:
        raise ValueError(f"NFUNC must be odd and ≥ 1; got {NFUNC}")

    if device is None:
        device = cu_seqlens_global.device
    B = cu_seqlens_global.numel() - 1
    cu_global_list = cu_seqlens_global.tolist()
    if num_contexts is not None:
        nc_list = num_contexts.tolist()
    else:
        nc_list = [0] * B
    if num_targets is not None:
        nt_list = num_targets.tolist()
    else:
        nt_list = [0] * B
    g = int(target_group_size)
    if g < 1:
        raise ValueError(f"target_group_size must be ≥ 1; got {g}")
    w_left, w_right = window_size

    chunks_per_seq = 2 * cp_size
    own_q_chunks = (cp_rank, chunks_per_seq - 1 - cp_rank)
    peer = (cp_rank - step) % cp_size
    own_k_chunks = (peer, chunks_per_seq - 1 - peer)

    # Compute per-sample local-Q starting offsets in the jagged
    # local-Q axis (= total_local_q).
    local_lens: list[int] = []
    total_local = 0
    for b in range(B):
        L_b = cu_global_list[b + 1] - cu_global_list[b]
        if L_b % chunks_per_seq != 0:
            raise ValueError(
                f"sample {b}: seqlen {L_b} not divisible by "
                f"chunks_per_seq={chunks_per_seq}"
            )
        local_lens.append(L_b // cp_size)  # 2 chunks per sample
        total_local += L_b // cp_size

    # Build the func tensor on the CPU as a numpy array first, then move
    # to the target device once. Writing element-by-element to a CUDA
    # tensor inside a Python loop incurs ~us-scale per-write
    # synchronisation overhead which dominates wall-clock for typical
    # shapes (measured ~277× slowdown vs the structured-mask path on
    # cp=4 + s=2048 het-mask before this change). Building on CPU with
    # vectorised numpy (`_per_sample_intervals_array`) then one bulk H2D
    # copy is ~2 orders of magnitude faster than the per-Q-row Python
    # loop variant.
    import numpy as np

    func_cpu = np.zeros((1, NFUNC, total_local), dtype=np.int32)
    local_offset = 0
    for b in range(B):
        L_b = cu_global_list[b + 1] - cu_global_list[b]
        if L_b == 0:
            continue
        c_b = L_b // chunks_per_seq
        nc = int(nc_list[b])
        nt = int(nt_list[b])
        # Mirror the invariant `build_global_mask_func` enforces: with
        # nc + nt > L_b the contextual / target predicates overlap, and
        # the vectorised builder (`_per_sample_intervals_array`)
        # silently double-writes the same row instead of raising the
        # way the scalar reference does (`_per_sample_intervals` skips
        # via `if q < nc: continue`). Validate up-front so a malformed
        # split fails loudly here rather than producing a divergent
        # mask.
        if nc < 0 or nt < 0 or nc + nt > L_b:
            raise ValueError(
                f"sample {b}: invalid heterogeneous mask split nc={nc}, "
                f"nt={nt}, L={L_b}"
            )
        # Get all L_b global intervals at once via vectorised numpy.
        intervals_global = _per_sample_intervals_array(
            L=L_b, nc=nc, nt=nt, g=g, w_left=w_left, w_right=w_right
        )  # shape (L_b, 2, 2)

        # Restrict to the local Q indices.  Layout: half h ∈ {0, 1}
        # corresponds to Q chunk own_q_chunks[h]; per-half local rows
        # span [h*c_b, (h+1)*c_b) and map to global rows
        # [own_q_chunks[h]*c_b, (own_q_chunks[h]+1)*c_b).
        local_indices = np.empty(2 * c_b, dtype=np.int64)
        for half in (0, 1):
            q_chunk_idx = own_q_chunks[half]
            local_indices[half * c_b : (half + 1) * c_b] = np.arange(
                q_chunk_idx * c_b, (q_chunk_idx + 1) * c_b, dtype=np.int64
            )
        local_intervals = intervals_global[local_indices]  # (2*c_b, 2, 2)

        # Project each global interval onto the local K layout.
        # Local K layout per sample is [chunk_peer; chunk_(2cp-1-peer)],
        # i.e. global rows [own_k_chunks[0]*c_b, (own_k_chunks[0]+1)*c_b)
        # at local [0, c_b), and [own_k_chunks[1]*c_b, (own_k_chunks[1]+1)*c_b)
        # at local [c_b, 2*c_b).
        k_chunk_first_lo, k_chunk_first_hi = (
            own_k_chunks[0] * c_b,
            (own_k_chunks[0] + 1) * c_b,
        )
        k_chunk_second_lo, k_chunk_second_hi = (
            own_k_chunks[1] * c_b,
            (own_k_chunks[1] + 1) * c_b,
        )

        # For each (row, interval_idx), compute up to two local sub-intervals
        # (one per chunk). Empty when intersection is empty.
        n_rows = 2 * c_b
        # First-chunk intersection.
        g_lo = local_intervals[:, :, 0]  # (n_rows, 2)
        g_hi = local_intervals[:, :, 1]
        # Empty global intervals (lo == hi == 0) trivially produce empty subs.
        i1_lo = np.maximum(g_lo, k_chunk_first_lo) - k_chunk_first_lo
        i1_hi = np.minimum(g_hi, k_chunk_first_hi) - k_chunk_first_lo
        i1_valid = (i1_lo < i1_hi) & (g_lo < g_hi)
        i1_lo = np.where(i1_valid, i1_lo, 0)
        i1_hi = np.where(i1_valid, i1_hi, 0)
        # Second-chunk intersection (mapped to local [c_b, 2*c_b)).
        i2_lo = c_b + np.maximum(g_lo, k_chunk_second_lo) - k_chunk_second_lo
        i2_hi = c_b + np.minimum(g_hi, k_chunk_second_hi) - k_chunk_second_lo
        i2_valid = (
            np.maximum(g_lo, k_chunk_second_lo) < np.minimum(g_hi, k_chunk_second_hi)
        ) & (g_lo < g_hi)
        i2_lo = np.where(i2_valid, i2_lo, 0)
        i2_hi = np.where(i2_valid, i2_hi, 0)

        # Pack into slots. Layout: slot 0 = upper of first interval,
        # pair (slot_{2k-1}, slot_{2k}) = (lo, hi) of interval k for
        # k ≥ 1. With at most 2 global intervals × 2 chunks = 4
        # sub-intervals max per row, but adjacent sub-intervals
        # (chunk_first followed by chunk_second of the same global
        # interval) merge if they are contiguous (only when the two
        # K chunks happen to be consecutive in global K, which is
        # rarely the case for DualChunkSwap — chunks {peer, 2cp-1-peer}
        # are usually far apart). So expect ≤ 4 sub-intervals; needs
        # NFUNC ≥ 7 for full safety. NFUNC=3 (max 2 intervals) covers
        # the common production case (single global history-band
        # intersected with 2 K chunks → 2 sub-intervals). Fall through
        # to a Python row-by-row pack for the rare > NFUNC // 2 + 1
        # case so we get a clean `ValueError` rather than silent truncation.
        max_intervals = NFUNC // 2 + 1
        # Sort sub-intervals per row by lower bound, then merge adjacent.
        # For HSTU's standard masks, sub-intervals from chunk_first map to
        # local [0, c_b) and from chunk_second map to local [c_b, 2*c_b),
        # so they are already in order.
        # Stack into a (n_rows, 4, 2) array of (lo, hi) pairs:
        # [interval_0 chunk_first, interval_0 chunk_second,
        #  interval_1 chunk_first, interval_1 chunk_second].
        all_lo = np.stack(
            [i1_lo[:, 0], i2_lo[:, 0], i1_lo[:, 1], i2_lo[:, 1]], axis=1
        )  # (n_rows, 4)
        all_hi = np.stack([i1_hi[:, 0], i2_hi[:, 0], i1_hi[:, 1], i2_hi[:, 1]], axis=1)

        # Per-row sort + merge. Vectorising the merge across rows is
        # awkward; per-row Python loop with tiny per-row cost (~few us
        # for 4 entries) is fine.
        for r in range(n_rows):
            row_lo = all_lo[r]
            row_hi = all_hi[r]
            # Filter out empties (lo == hi).
            valid_mask = row_lo < row_hi
            valids = list(zip(row_lo[valid_mask].tolist(), row_hi[valid_mask].tolist()))
            valids.sort()
            merged: list[tuple[int, int]] = []
            for lo, hi in valids:
                if merged and merged[-1][1] >= lo:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
                else:
                    merged.append((lo, hi))
            if len(merged) > max_intervals:
                raise ValueError(
                    f"sample {b} q_local {r}: produced {len(merged)} K "
                    f"intervals which exceeds NFUNC={NFUNC} capacity "
                    f"(max {max_intervals}). Bump HSTU_ARBITRARY_NFUNC."
                )
            slots = _intervals_to_slots(merged, NFUNC)
            func_cpu[0, :, local_offset + r] = slots
        local_offset += local_lens[b]

    return torch.from_numpy(func_cpu).to(device)


def build_global_mask_func(
    *,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
    NFUNC: int = 3,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a `(1, NFUNC, total_q)` int32 tensor encoding the HSTU global
    mask in the FBGEMM arbitrary-mask format.

    The kernel layout is JAGGED along `total_q` (samples concatenated;
    sample b occupies indices `[cu_seqlens_q[b], cu_seqlens_q[b+1])`).
    The kernel reads
    `func_ptr + binfo.sum_s_q + (j * func_ids_stride) + q_local` which
    only resolves correctly when the trailing axis is `total_q` not
    `max_seqlen_q` (verified empirically against
    `hstu_attn_varlen_func`'s causal output for non-uniform batches —
    a `(B, H, NFUNC, max_seqlen_q)` padded layout silently produces
    wrong outputs when sample lengths differ from `max_seqlen_q`).

    Heads are uniform: the kernel reads only `h=0` (see
    `hstu_fwd.h:163` `mMaxFunc(Int<0>{}, _, _)`), so we keep the head
    axis at 1 and broadcast.

    Caller contract (matches the FBGEMM kernel's intersection
    semantics): pass `window_size=(-1, -1)`, `num_contexts=None`,
    `num_targets=None`, `target_group_size=1` to disable the structured
    mask layer; the `func` tensor carries every constraint.

    Args:
        cu_seqlens_q: int32 (B+1,) cumulative seqlens.
        max_seqlen_q: max per-sample seqlen (used only for guard checks).
        num_contexts: int32 (B,) per-sample contextual prefix length, or None.
        num_targets: int32 (B,) per-sample target region length, or None.
        target_group_size: int.  When `num_targets is None`, ignored.
        window_size: (left, right).  v1 supports (-1, 0) (causal, no
            sliding) and (w, 0) (sliding-causal).  Other shapes raise.
        NFUNC: arbitrary-mask compile-time constant.  Must be odd and
            ≥ 1; max disjoint intervals = NFUNC // 2 + 1.

    Returns:
        int32 tensor `(1, NFUNC, total_q)`.  For global token index
        `g = cu_seqlens_q[b] + q_local`, slot 0 encodes interval 0's
        upper, and pairs (1, 2), (3, 4), … encode subsequent intervals'
        (lower, upper).  Unused slots are 0.
    """
    if NFUNC < 1 or NFUNC % 2 == 0:
        raise ValueError(f"NFUNC must be odd and ≥ 1; got {NFUNC}")
    w_left, w_right = window_size
    if w_right != 0 and w_right != -1:
        raise ValueError(
            f"build_global_mask_func: window_size={window_size} unsupported "
            f"(w_right must be 0 (causal) or -1 (no causal limit); lookahead "
            f"is out of v0.5 scope)"
        )
    g = int(target_group_size)
    if g < 1:
        raise ValueError(f"target_group_size must be ≥ 1; got {g}")

    if device is None:
        device = cu_seqlens_q.device
    B = cu_seqlens_q.numel() - 1
    cu_list = cu_seqlens_q.tolist()
    total_q = cu_list[-1]
    if num_contexts is not None:
        nc_list = num_contexts.tolist()
    else:
        nc_list = [0] * B
    if num_targets is not None:
        nt_list = num_targets.tolist()
    else:
        nt_list = [0] * B

    max_intervals = NFUNC // 2 + 1
    # Build on CPU (numpy), then bulk H2D copy at the end. See
    # `localize_func_for_cp_step` for rationale (per-element CUDA writes
    # in a Python loop are O(us) each → dominate wall-clock).
    import numpy as np

    func_cpu = np.zeros((1, NFUNC, total_q), dtype=np.int32)

    for b in range(B):
        L_b = cu_list[b + 1] - cu_list[b]
        if L_b <= 0:
            continue
        if L_b > max_seqlen_q:
            raise ValueError(
                f"sample {b}: seqlen {L_b} exceeds declared max_seqlen_q "
                f"{max_seqlen_q}"
            )
        nc = int(nc_list[b])
        nt = int(nt_list[b])
        if nc < 0 or nt < 0 or nc + nt > L_b:
            raise ValueError(
                f"sample {b}: invalid heterogeneous mask split nc={nc}, "
                f"nt={nt}, L={L_b}"
            )
        intervals_per_q = _per_sample_intervals(
            L=L_b, nc=nc, nt=nt, g=g, w_left=w_left, w_right=w_right
        )
        sample_offset = cu_list[b]
        for q_local, intervals in enumerate(intervals_per_q):
            if len(intervals) > max_intervals:
                raise ValueError(
                    f"sample {b} q_pos {q_local}: produced {len(intervals)} K "
                    f"intervals which exceeds NFUNC={NFUNC} capacity "
                    f"(max {max_intervals}). Bump HSTU_ARBITRARY_NFUNC or "
                    f"reduce mask shape complexity."
                )
            slots = _intervals_to_slots(intervals, NFUNC)
            global_idx = sample_offset + q_local
            func_cpu[0, :, global_idx] = slots

    return torch.from_numpy(func_cpu).to(device)
