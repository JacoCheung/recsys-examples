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
    contribute to Q[q].

    Reproduces the predicate from
    `examples/hstu/ops/pt_ops/pt_hstu_attention.py::_get_valid_attn_mask`
    in scalar form. `(w_left, w_right)` is the kernel's `window_size`
    convention: `(-1, 0)` ⇒ pure causal (no sliding); `w_left ≥ 0` ⇒
    sliding-causal with span `w_left`. We currently support
    `w_right == 0` only (sliding-causal); `w_right > 0` (lookahead) is
    out of v0/v0.5 scope and rejected before reaching this function.
    """
    is_sliding = w_left >= 0
    is_causal = w_right == 0  # FBGEMM convention: right=0 ⇒ causal limit

    # Region boundaries in raw (non-renormalised) Q/K position space.
    # Contextual prefix occupies [0, nc).  History occupies [nc, L-nt).
    # Targets occupy [L-nt, L), partitioned into groups of size g.
    history_end = L - nt  # exclusive
    target_start = L - nt  # inclusive

    out: list[list[tuple[int, int]]] = []
    for q in range(L):
        # Causal upper bound on K (post-self) ignoring all extra mask logic.
        # `causal_hi` = q + 1 if causal else L (unconstrained).
        causal_hi = q + 1 if is_causal else L

        # Sliding lower bound (post-window). For non-sliding, no extra cut.
        sliding_lo = max(0, q - w_left) if is_sliding and is_causal else 0

        if q < nc:
            # Contextual Q: per PT reference line 99-104, allowed K is
            # `[0, max_ids_after_contextual_decrement) = [0, L - nt)`.
            # The target-group constraint already passes (because target
            # K's group_col_id is -1, and contextual Q's group_row_id is
            # also -1, so target_group_mask is True). The eye+causal
            # `row_col_dist > 0` only allows contextuals (renormalised id=0)
            # to attend to other id=0 cells (other contextuals); the
            # extra OR rule extends this to history.  Net: contextuals
            # see contextuals + history, NOT targets.
            lo = sliding_lo if (is_sliding and is_causal) else 0
            hi = min(history_end, L)
            if lo < hi:
                out.append([(lo, hi)])
            else:
                out.append([])
            continue

        if q < history_end:
            # History Q: causal lower-tri up to self, restricted to
            # non-target K.  Sliding (if any) already cuts the lower edge.
            # Target K is dropped because:
            #   - causal blocks K > q (and target K > history Q is always
            #     above q),
            #   - target_group_mask blocks history Q × target K when the
            #     target_group_col is ≥ 0 and target_group_row is -1.
            # So the valid K range is [sliding_lo, q+1).
            hi = min(causal_hi, history_end)
            lo = sliding_lo
            if lo < hi:
                out.append([(lo, hi)])
            else:
                out.append([])
            continue

        # Target Q: q ∈ [target_start, L).
        group_q = (q - target_start) // g
        target_group_lo = target_start + group_q * g
        # Self-and-prior in same target group.  Causal cuts at q+1.
        target_lo = (
            max(target_group_lo, sliding_lo)
            if is_sliding and is_causal
            else target_group_lo
        )
        target_hi = min(causal_hi, L) if is_causal else L

        # And history (non-target) K — target Q sees ALL non-target K
        # (contextual + history) as long as causal+sliding hold.  The
        # target_group_mask passes because col_group_id is -1 there.
        history_for_target_lo = sliding_lo
        history_for_target_hi = history_end

        intervals: list[tuple[int, int]] = []
        if history_for_target_lo < history_for_target_hi:
            intervals.append((history_for_target_lo, history_for_target_hi))
        if target_lo < target_hi:
            # If the target interval starts exactly at history_end and
            # the previous interval ended at history_end, merge them.
            if intervals and intervals[-1][1] == target_lo:
                intervals[-1] = (intervals[-1][0], target_hi)
            else:
                intervals.append((target_lo, target_hi))
        out.append(intervals)

    return out


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
    func = torch.zeros((1, NFUNC, total_q), dtype=torch.int32, device=device)

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
            global_idx = sample_offset + q_local
            if len(intervals) == 0:
                # No allowed K — leave slots at 0; encodes empty range.
                continue
            lo0, hi0 = intervals[0]
            # Slot 0 is the upper bound of an implicit interval starting at 0.
            # If our first interval starts at lo0 > 0, encode it as the
            # second interval (slot 1 = lo0, slot 2 = hi0) and zero slot 0
            # (empty implicit first interval).
            if lo0 > 0:
                if NFUNC < 3:
                    raise ValueError(
                        f"sample {b} q_pos {q_local}: interval starts at "
                        f"lo0={lo0}>0 requires NFUNC≥3, got {NFUNC}"
                    )
                func[0, 0, global_idx] = 0
                func[0, 1, global_idx] = lo0
                func[0, 2, global_idx] = hi0
                next_pair_idx = 1
            else:
                func[0, 0, global_idx] = hi0
                next_pair_idx = 0
            for interval in intervals[1:]:
                lo, hi = interval
                next_pair_idx += 1
                slot_lo = 2 * next_pair_idx - 1
                slot_hi = 2 * next_pair_idx
                if slot_hi >= NFUNC:
                    raise ValueError(
                        f"sample {b} q_pos {q_local}: ran out of NFUNC slots "
                        f"({NFUNC}); needs ≥ {slot_hi + 1} for "
                        f"{len(intervals)} intervals"
                    )
                func[0, slot_lo, global_idx] = lo
                func[0, slot_hi, global_idx] = hi

    return func
