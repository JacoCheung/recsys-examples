# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
HSTU Context-Parallel attention wrapper (Slice 3 — multi-GPU forward).

Public entry point: `hstu_attn_varlen_cp_func`.

What this module provides for v0
================================
- A drop-in callable users can swap in for `hstu_attn_varlen_func`.
  Signature mirrors the installed kernel exactly plus four CP arguments.
- Hard guards rejecting v0+ modes with `ValueError` (per SPEC §2 / plan T3.1).
- `cp_size == 1` short-circuit: direct delegation to `hstu_attn_varlen_func`,
  no autograd wrap, no comm. Guards still run uniformly per plan T3.1
  (cost is a few Python conditionals — well within plan §Global rule 3
  cp=1 perf budget).
- DualChunkSwap dispatch helper `get_batch_on_this_cp_rank_for_hstu`
  (pure permutation; T3.2) plus testing-only `gather_global_from_cp_rank`.
- Multi-GPU forward path is implemented (T3.3): single CUDA stream,
  sequential ring P2P via `dist.batch_isend_irecv`, plain-sum reduction
  in fp32 across the (rank, step) classification grid (diagonal /
  lower-triangle / upper-triangle).
- Backward (T4.2): explicit reverse-direction ring. dQ stays local; dK/dV
  partials ride the reverse ring back to their owning rank with
  copy-on-first-receive / add-after semantics.

What this module does NOT do (v0 / SPEC §2)
===========================================
- Sliding-causal, `rab`, heterogeneous mask (`num_contexts`,
  `num_targets`, `target_group_size > 1`), FP8, KV-cache, Ulysses,
  comm/compute overlap (Slice 5), training-loop integration (Slice 6).
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.distributed as dist

# Use the *installed* `hstu.hstu_attn_varlen_func` as the runtime kernel
# (Global rule 6 — runtime authority). The in-tree `.hstu_attn_interface`
# is built from a different C-extension and is unavailable in production
# containers; importing it here would crash before the wrapper is even
# loaded. The kernel signature pin in `examples/hstu/test/cp/conftest.py`
# guards against signature drift.
from hstu import hstu_attn_varlen_func

__all__ = [
    "hstu_attn_varlen_cp_func",
    "get_batch_on_this_cp_rank_for_hstu",
    "gather_global_from_cp_rank",
    "apply_dualchunkswap_to_jagged",
    "gather_jagged_from_cp_rank",
    "GuardError",
]


# ----------------------------------------------------------------------------
# Errors. We reuse `ValueError` for guard rejections (matches SPEC §7) but
# expose a typed alias so tests / callers can `except GuardError`.
# ----------------------------------------------------------------------------
class GuardError(ValueError):
    """Raised when an input doesn't satisfy v0 contract (SPEC §2)."""


# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
_SUPPORTED_HEAD_DIMS = (32, 64, 128, 256)
_SUPPORTED_WINDOW_SIZE = (-1, 0)  # pure causal only
_SPEC_REF = "see SPEC §2 (out-of-scope) and plan T3.1 (hard-guard list)"


# ----------------------------------------------------------------------------
# Slice 5 — secondary CUDA stream used to overlap NCCL P2P with HSTU
# attention compute. One stream per CUDA device per process, lazily created.
# ----------------------------------------------------------------------------
_default_cp_streams: dict[int, "torch.cuda.Stream"] = {}


# ----------------------------------------------------------------------------
# Ring-P2P recv buffer pool — keyed by (dtype, device, slot) ONLY.
#
# `_multi_gpu_forward_arbitrary` and `_multi_gpu_backward_arbitrary` used to
# allocate fresh `torch.empty_like(k_local)` every layer for the recv side of
# the ring P2P (recv_k, recv_v, and in backward also recv_dk, recv_dv).
# Across 8 layers × {fwd, bwd KV-reload, bwd reverse-ring}, that's ~32 fresh
# ~56 MB allocations per training step.
#
# We pool one storage per (dtype, device, slot). Different cu_seqlens between
# training steps give different total_tokens — and thus different
# `k_local.shape[0]` — so we DO NOT key by shape (a previous attempt did,
# leaking ~80 MB/step until OOM at iter ~400 — see commit log "fix(cp):
# pool recv buffers across layers"). Instead we keep a 1-D max-size storage
# per slot, grow it on demand, and return a `view(template.shape)` carved
# out of the front. The view is contiguous because the storage was allocated
# contiguous and we slice the leading 1-D, which is what NCCL P2P needs.
#
# Stream-correctness: each ring step ends with
# `default_stream.wait_stream(comm_stream)` followed by reading the recv
# buffer; the next layer's `comm_stream.wait_stream(default_stream)` before
# the next NCCL P2P guarantees the previous layer's last read on the same
# buffer has completed before NCCL writes. Reusing the storage across layers
# is therefore safe even though comm and compute live on separate streams.
# ----------------------------------------------------------------------------
_recv_buffer_pool: dict[tuple, torch.Tensor] = {}


def _get_recv_buffer(template: torch.Tensor, slot: str) -> torch.Tensor:
    """Return a `template.shape`-sized contiguous view onto a pooled storage.

    Pool key is (template.dtype, template.device, slot). The storage is
    allocated 1-D with `template.numel()` elements on first use and grown
    in place if a later call's `template.numel()` exceeds the cached size.

    `slot` is a logical name ("fwd_recv_k" etc.) — distinct slots map to
    distinct physical storages so a single layer can hold two recv tensors
    (recv_k AND recv_v) simultaneously without aliasing.
    """
    key = (template.dtype, template.device, slot)
    needed = template.numel()
    cached = _recv_buffer_pool.get(key)
    if cached is None or cached.numel() < needed:
        cached = torch.empty(needed, dtype=template.dtype, device=template.device)
        _recv_buffer_pool[key] = cached
    # Slice the front then reshape to template.shape. Front-slice on a
    # contiguous 1-D tensor stays contiguous; .view(shape) is a no-copy
    # reshape that NCCL P2P consumes natively.
    return cached.narrow(0, 0, needed).view(template.shape)


def cp_recv_buffer_pool_clear() -> None:
    """Free the pooled recv buffers. Tests / shutdown only."""
    global _recv_buffer_pool
    _recv_buffer_pool = {}


# ----------------------------------------------------------------------------
# Cross-training-step `func` tensor cache.
#
# `localize_func_for_cp_step` is invoked PER LAYER PER RING STEP inside
# `_multi_gpu_forward_arbitrary` / `_multi_gpu_backward_arbitrary`. The
# resulting `func` is a pure function of (cu_seqlens_global, cp_size,
# cp_rank, step, num_contexts, num_targets, target_group_size,
# window_size, NFUNC) — every kwarg is hashable to a stable key. Two
# training steps that hand in the same `cu_seqlens_global` values get
# the same `func` tensor; nothing else needs to match.
#
# So the cache is keyed by the FULL kwarg content, not by anything
# scope-bound. Two consequences:
#   1. Within one HSTUBlock.forward, all 8 layers share the same 2 (=
#      cp_size) builds — same as before.
#   2. Across training steps, the cache is reused whenever the dataset
#      replays the same `cu_seqlens` values. The HSTU benchmark generates
#      `num_generated_batches` unique batches and cycles them, so after
#      the first few hundred iters every call is a cache hit and the
#      build cost drops to 0 ms / step.
#
# History of this cache:
#  - EOS round-3 no caching:           1150 ms/step (32 builds)
#  - cw-dfw round-4 forward-only TLS:   734 ms/step (forward thread's
#    TLS was invisible to the autograd worker thread; backward rebuilt)
#  - cw-dfw round-5 module-level dict,
#    keyed by (step,) only:             165 ms/step (still 2 builds /
#    step from per-step scope_enter() reset)
#  - this rev (round-7), keyed by full
#    kwarg content, no per-step reset:  builds amortise to 0 once the
#    dataset cycles
#
# Threading note: process-global (NOT thread-local) so the autograd
# worker thread that runs backward sees what the main thread populated
# in forward. HSTU training is single-step single-thread, so no
# concurrent writers race on this dict.
#
# Memory bound: capped at `_CP_FUNC_CACHE_MAX` entries to handle truly
# infinite-cardinality datasets without unbounded growth. When the cap
# trips we evict an arbitrary entry (FIFO order from `next(iter(...))`)
# — for the HSTU benchmark this never trips. For production with
# random batches per step, set `_CP_FUNC_CACHE_MAX` lower or call
# `cp_func_cache_scope_enter()` periodically (it is a "clear" hook).
# ----------------------------------------------------------------------------
_cp_func_cache: dict[tuple, torch.Tensor] = {}
_CP_FUNC_CACHE_MAX = 1024


def cp_func_cache_scope_enter() -> None:
    """Optional hook to clear the cache at a workload boundary.

    Most callers should NOT use this — the cache is keyed by content,
    so leaving it alone across training steps is correct AND the
    intended fast path. Provided as an escape hatch for tests or for
    workloads that want to force a re-build (e.g. memory-pressure
    intervention between phases).
    """
    global _cp_func_cache
    _cp_func_cache = {}


def cp_func_cache_scope_exit() -> None:
    """No-op kept for API symmetry."""
    return


def _hash_int_tensor_or_none(t: Optional[torch.Tensor]) -> Optional[tuple]:
    """Project an int32/int64 tensor down to a hashable tuple of ints.

    Forces a GPU→CPU sync via `.tolist()` — accepted because we already
    pay one such sync per forward (the kernel's `cu_seqlens_q` argument
    is consumed CPU-side anyway), and folding it into the cache key
    saves a *full* `localize_func_for_cp_step` build (which itself
    starts with `tolist()` and runs ~tens of ms of numpy work).
    """
    if t is None:
        return None
    return tuple(t.tolist())


def _cached_localize_func_for_cp_step(
    *,
    step: int,
    cu_seqlens_global: torch.Tensor,
    cp_size: int,
    cp_rank: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
    NFUNC: int = 3,
    device: Optional[torch.device] = None,
    cu_seqlens_global_tuple: Optional[tuple] = None,
    num_contexts_tuple: Optional[tuple] = None,
    num_targets_tuple: Optional[tuple] = None,
) -> torch.Tensor:
    """Cached wrapper for `localize_func_for_cp_step`.

    Cache key is the full kwarg content; cu_seqlens / num_contexts /
    num_targets are projected to int tuples via `.tolist()`. This costs
    one GPU→CPU sync per call (per integer tensor) — saves the
    multi-tens-of-ms numpy build on every cache hit, but the sync
    itself is wasteful when called repeatedly per layer.

    Caller fast path: if you already have the tuple form (e.g. you
    `.tolist()`'d once at the autograd Function entry, paying the sync
    once per training step instead of once per layer × ring step), pass
    `cu_seqlens_global_tuple` / `num_contexts_tuple` / `num_targets_tuple`
    to skip the internal hashing sync entirely. The tuples must reflect
    the SAME values as the corresponding tensor arguments — caller's
    contract.

    Cache survives across training steps. The dataset cycles after
    `num_generated_batches`, so steady-state hit rate is ~100%.
    """
    from ._mask_func import localize_func_for_cp_step

    cu_key = (
        cu_seqlens_global_tuple
        if cu_seqlens_global_tuple is not None
        else _hash_int_tensor_or_none(cu_seqlens_global)
    )
    nc_key = (
        num_contexts_tuple
        if num_contexts_tuple is not None or num_contexts is None
        else _hash_int_tensor_or_none(num_contexts)
    )
    nt_key = (
        num_targets_tuple
        if num_targets_tuple is not None or num_targets is None
        else _hash_int_tensor_or_none(num_targets)
    )
    key = (
        step,
        cp_size,
        cp_rank,
        cu_key,
        nc_key,
        nt_key,
        target_group_size,
        tuple(window_size),
        NFUNC,
    )
    cache = _cp_func_cache
    hit = cache.get(key)
    if hit is not None:
        return hit
    # Bound the cache (FIFO eviction) — the HSTU benchmark's
    # `num_generated_batches=100` × `cp_size` = 200 keys never trips
    # `_CP_FUNC_CACHE_MAX=1024`, so this is just defensive.
    if len(cache) >= _CP_FUNC_CACHE_MAX:
        cache.pop(next(iter(cache)))
    built = localize_func_for_cp_step(
        step=step,
        cu_seqlens_global=cu_seqlens_global,
        cp_size=cp_size,
        cp_rank=cp_rank,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
        window_size=window_size,
        NFUNC=NFUNC,
        device=device,
    )
    cache[key] = built
    return built


def _get_cp_stream(
    device: torch.device, user_stream: Optional["torch.cuda.Stream"]
) -> "torch.cuda.Stream":
    """Return the comm stream this rank should use for ring P2P.

    Caller may inject `user_stream` via `hstu_attn_varlen_cp_func(cp_stream=...)`
    to share a stream across modules; otherwise we cache one per device on
    the module. Each device gets its own stream so multi-stream code on one
    rank with > 1 visible CUDA device (e.g. tests) does not cross-pollute.
    """
    if user_stream is not None:
        return user_stream
    key = device.index if device.index is not None else torch.cuda.current_device()
    cached = _default_cp_streams.get(key)
    if cached is None:
        cached = torch.cuda.Stream(device=key)
        _default_cp_streams[key] = cached
    return cached


# ----------------------------------------------------------------------------
# DualChunkSwap dispatch helper (T3.2).
#
# Maps a global packed batch onto the local shard owned by `(cp_rank, cp_size)`.
# Pure permutation — no `torch.distributed` calls.
#
# Per sequence of length L (must be divisible by 2*cp_size), each chunk is
# size c = L / (2*cp_size). Rank r owns chunks {r, 2*cp_size-1-r}; local
# layout per sample is [chunk_r, chunk_(2cp-1-r)].
# ----------------------------------------------------------------------------
def get_batch_on_this_cp_rank_for_hstu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_global: torch.Tensor,
    *,
    cp_size: int,
    cp_rank: int,
) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]
]:
    """Gather rank `cp_rank`'s DualChunkSwap chunks from a global batch.

    Args:
        q, k, v: global packed tensors, shape (total_tokens, num_heads, head_dim).
        cu_seqlens_global: int32 (B+1,) global cu_seqlens.
        cp_size: number of CP ranks.
        cp_rank: this rank's id, in `[0, cp_size)`.

    Returns:
        q_local, k_local, v_local: per-rank shards in DualChunkSwap order.
        cu_seqlens_local: int32 (B+1,), each sample of length 2 * (per-sample chunk size).
        local_to_global: int64 (sum_b 2*c_b,), maps local row → global row index
          (used by `gather_global_from_cp_rank` and by the multi-GPU output
          scatter logic).
        chunk_sizes: list of per-sample chunk sizes c_b.

    Raises:
        GuardError: if `cp_size < 1`, `cp_rank` out of range, or any per-sample
          seqlen not divisible by `2 * cp_size`.
    """
    if cp_size < 1:
        raise GuardError(f"cp_size must be ≥ 1; got {cp_size}")
    if not 0 <= cp_rank < cp_size:
        raise GuardError(f"cp_rank must be in [0, {cp_size}); got {cp_rank}")
    if cp_size == 1:
        # Degenerate: every rank holds the entire global batch. Return as-is.
        idx = torch.arange(q.shape[0], device=q.device, dtype=torch.long)
        seqlens = (cu_seqlens_global[1:] - cu_seqlens_global[:-1]).tolist()
        return q, k, v, cu_seqlens_global, idx, list(seqlens)

    chunks_per_seq = 2 * cp_size
    own = (cp_rank, chunks_per_seq - 1 - cp_rank)
    device = q.device

    seqlens_global = (cu_seqlens_global[1:] - cu_seqlens_global[:-1]).tolist()
    cu_global_list = cu_seqlens_global.tolist()

    rows: list[torch.Tensor] = []
    local_lens: list[int] = []
    chunk_sizes: list[int] = []
    for b, L in enumerate(seqlens_global):
        if L % chunks_per_seq != 0:
            raise GuardError(
                f"sample {b} seqlen {L} is not divisible by 2*cp_size={chunks_per_seq}"
            )
        c_b = L // chunks_per_seq
        chunk_sizes.append(c_b)
        base = cu_global_list[b]
        for chunk_id in own:
            rows.append(
                torch.arange(
                    base + chunk_id * c_b, base + (chunk_id + 1) * c_b, device=device
                )
            )
        local_lens.append(2 * c_b)

    local_to_global = torch.cat(rows)
    q_local = q[local_to_global].contiguous()
    k_local = k[local_to_global].contiguous()
    v_local = v[local_to_global].contiguous()
    cu_local = (
        torch.tensor([0] + local_lens, dtype=torch.int32, device=device).cumsum(0).int()
    )
    return q_local, k_local, v_local, cu_local, local_to_global, chunk_sizes


def gather_global_from_cp_rank(
    local: torch.Tensor,
    local_to_global: torch.Tensor,
    *,
    global_total_tokens: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of `get_batch_on_this_cp_rank_for_hstu` for testing.

    Scatters a single rank's local tensor back into a globally-shaped buffer.
    In production, this gather runs across CP ranks (e.g. an all-gather);
    here it is local-only and used by single-process tests.
    """
    if out is None:
        shape = (global_total_tokens, *local.shape[1:])
        out = torch.zeros(shape, dtype=local.dtype, device=local.device)
    out[local_to_global] += local
    return out


# ----------------------------------------------------------------------------
# T6.3 — JaggedData-level DualChunkSwap dispatch.
#
# The single-tensor helper above (`get_batch_on_this_cp_rank_for_hstu`) takes
# already-projected Q/K/V. T6.3 ships the dispatcher one step earlier so the
# input embedding `X` (single jagged tensor) gets sharded BEFORE the UVQK
# linear projection. Each rank then runs the projection and all downstream
# token-wise layers (LayerNorm, MLP, output projection) on its local shard
# only — that is the memory-wall enabler.
#
# Caller contract: pass a JaggedData whose `values` is the global
# (T_global, hidden_dim) embedding output. Returns a JaggedData with
# `values=(T_local, hidden_dim)` in DualChunkSwap order, plus the
# `local_to_global` index that the loss-side gather will need to undo the
# permutation. Heterogeneous-mask fields (num_candidates, contextual_seqlen)
# are NOT permuted; v0 already rejects them at the wrapper, so the helper
# refuses to dispatch when they are non-trivial rather than silently producing
# a wrong shard.
# ----------------------------------------------------------------------------
def apply_dualchunkswap_to_jagged(
    jd,  # type: ignore[no-untyped-def]
    *,
    cp_size: int,
    cp_rank: int,
):
    """Return (jd_local, local_to_global) — the DualChunkSwap shard of `jd`.

    `jd` must be a `modules.jagged_data.JaggedData` instance. We do a
    real `isinstance` check rather than duck-typing because the
    constructor we use to build `jd_local` is the JaggedData class
    itself, so accepting anything else would just defer the same
    failure into the constructor with a less useful error message.

    The `from modules.jagged_data import` is done lazily (inside the
    function body) so the wrapper module can be imported in
    environments that do not have the training-side `modules.*` tree
    on `sys.path` (e.g. the smoke tests under
    `examples/hstu/test/cp/test_cp_api_smoke.py` which only need the
    `hstu_attn_varlen_cp_func` Q/K/V path). Production callers
    that already use JaggedData necessarily have the import path set
    up; this helper trades a clear ImportError at first call for the
    decoupling.

    `cp_size == 1` short-circuits to (jd, identity). For cp_size > 1 each
    sample's seqlen must be divisible by `2 * cp_size` (DualChunkSwap
    requirement; same as `get_batch_on_this_cp_rank_for_hstu`).

    Returns the local jagged data and an int64 index `local_to_global` such
    that `local.values == jd.values[local_to_global]`. Pass that index to the
    loss-side gather (`gather_jagged_from_cp_rank`).

    Refuses to dispatch when JaggedData metadata is incompatible with
    DualChunkSwap permutation:
      - padding_length > 0 (would be permuted into the live region;
        SP+CP composition is out of v0 scope, see SPEC §2)

    `has_interleaved_action=True` IS supported. The interleaved layout
    is `[item₀, action₀, item₁, action₁, ...]` per sample — each row is
    an independent Q/K token in attention's view, causal mask applies
    per row position. DualChunkSwap row-level slicing is mathematically
    equivalent regardless of whether rows came from interleaved or
    item-only layout (verified vs single-GPU baseline at fp64).

    Het-mask metadata (`max_num_candidates`, `num_candidates`,
    `num_candidates_offsets`, `contextual_max_seqlen`, `contextual_seqlen`,
    `contextual_seqlen_offsets`) ARE preserved through the dispatch in
    the het-mask track (`docs/cp/het_mask_design.md`). They are
    per-sample fields (not per-token) so DualChunkSwap permutation does
    not touch them; downstream `FusedHSTUAttention` consumes them via
    the CP wrapper's arbitrary-mask path.
    """
    # Lazy local import so this file can be loaded without the training
    # tree on PYTHONPATH (test_cp_api_smoke.py imports the wrapper module
    # without the modules.* tree).
    from modules.jagged_data import JaggedData  # noqa: WPS433  (intentional)

    if not isinstance(jd, JaggedData):
        raise GuardError(
            f"apply_dualchunkswap_to_jagged expects a modules.jagged_data."
            f"JaggedData instance; got {type(jd).__name__}"
        )

    if cp_size < 1:
        raise GuardError(f"cp_size must be ≥ 1; got {cp_size}")
    if not 0 <= cp_rank < cp_size:
        raise GuardError(f"cp_rank must be in [0, {cp_size}); got {cp_rank}")

    if jd.padding_length > 0:
        raise GuardError(
            "apply_dualchunkswap_to_jagged: padding_length > 0 not supported "
            "in v0 — SP+CP composition is out of v0 scope (see SPEC §2). "
            "Either disable SP for the CP run, or unpad before calling."
        )

    if cp_size == 1:
        idx = torch.arange(
            jd.values.shape[0], device=jd.values.device, dtype=torch.long
        )
        return jd, idx

    chunks_per_seq = 2 * cp_size
    own = (cp_rank, chunks_per_seq - 1 - cp_rank)
    device = jd.values.device

    seqlens_global = jd.seqlen.tolist()
    cu_global_list = jd.seqlen_offsets.tolist()

    rows: list[torch.Tensor] = []
    local_lens: list[int] = []
    for b, L in enumerate(seqlens_global):
        if L % chunks_per_seq != 0:
            raise GuardError(
                f"sample {b} seqlen {L} is not divisible by 2*cp_size={chunks_per_seq}"
            )
        c_b = L // chunks_per_seq
        base = cu_global_list[b]
        for chunk_id in own:
            rows.append(
                torch.arange(
                    base + chunk_id * c_b, base + (chunk_id + 1) * c_b, device=device
                )
            )
        local_lens.append(2 * c_b)

    local_to_global = torch.cat(rows)
    values_local = jd.values[local_to_global].contiguous()
    seqlen_local = torch.tensor(local_lens, dtype=jd.seqlen.dtype, device=device)
    seqlen_offsets_local = (
        torch.tensor([0] + local_lens, dtype=jd.seqlen_offsets.dtype, device=device)
        .cumsum(0)
        .to(jd.seqlen_offsets.dtype)
    )
    max_seqlen_local = max(local_lens) if local_lens else 0

    jd_local = JaggedData(
        values=values_local,
        seqlen=seqlen_local,
        seqlen_offsets=seqlen_offsets_local,
        max_seqlen=max_seqlen_local,
        # Heterogeneous-mask fields are preserved as-is — they are
        # per-sample (not per-token) so DualChunkSwap permutation does
        # not change them. The downstream CP wrapper consumes them via
        # the arbitrary-mask path (`docs/cp/het_mask_design.md`).
        max_num_candidates=jd.max_num_candidates,
        num_candidates=jd.num_candidates,
        num_candidates_offsets=jd.num_candidates_offsets,
        contextual_max_seqlen=jd.contextual_max_seqlen,
        contextual_seqlen=jd.contextual_seqlen,
        contextual_seqlen_offsets=jd.contextual_seqlen_offsets,
        has_interleaved_action=jd.has_interleaved_action,
        scaling_seqlen=jd.scaling_seqlen,
        padding_length=0,
        total_candidates_seq_len=jd.total_candidates_seq_len,
    )
    return jd_local, local_to_global


def gather_jagged_from_cp_rank(
    local_values: torch.Tensor,
    local_to_global: torch.Tensor,
    *,
    cp_group: Optional["dist.ProcessGroup"] = None,
    global_total_tokens: int,
) -> torch.Tensor:
    """Inverse of `apply_dualchunkswap_to_jagged` on a single rank's `values`.

    For `cp_group=None` (single-process / cp_size==1), this is a local
    scatter — equivalent to the testing-only `gather_global_from_cp_rank`.
    For multi-rank `cp_group`, the gather is an all-reduce SUM across the
    group: each rank scatters its shard into a globally-shaped buffer (zeros
    elsewhere) and the all-reduce assembles the full output. Each global
    position is owned by exactly one rank, so SUM is identity.

    Returned tensor lives on the same device + dtype as `local_values`.
    """
    out = torch.zeros(
        (global_total_tokens, *local_values.shape[1:]),
        dtype=local_values.dtype,
        device=local_values.device,
    )
    out[local_to_global] = local_values
    if cp_group is not None and dist.get_world_size(cp_group) > 1:
        # Promote to fp32 for the reduction (consistent with SPEC §2's
        # "reduction in fp32" rule used inside the wrapper). We reduce
        # over a dtype-converted clone so the dtype-dependent semantics
        # of dist.all_reduce on bf16 do not bite.
        buf32 = out.float()
        dist.all_reduce(buf32, op=dist.ReduceOp.SUM, group=cp_group)
        out = buf32.to(local_values.dtype)
    return out


# ----------------------------------------------------------------------------
# Hard guards (T3.1). Returns silently if all ok; raises `GuardError` otherwise.
# ----------------------------------------------------------------------------
def _enforce_v0_contract(
    *,
    q: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
    rab: Optional[torch.Tensor],
    has_drab: bool,
    kv_cache: Optional[torch.Tensor],
    page_offsets: Optional[torch.Tensor],
    page_ids: Optional[torch.Tensor],
    last_page_lens: Optional[torch.Tensor],
    func: Optional[torch.Tensor],
    quant_mode: Optional[int],
    cp_size: int,
) -> None:
    # 1-4. Heterogeneous mask + sliding-causal + arbitrary `func`.
    #
    # Step 4a: cp_size>1 het-mask **forward** is now supported via the
    # arbitrary-mask path (`_multi_gpu_forward_arbitrary` builds a per-step
    # `func` tensor from the 4-tuple spec). cp_size==1 still forwards
    # natively. Backward under cp_size>1 het-mask raises at .backward()
    # time (Step 4b).
    #
    # The explicit `func` tensor input is still rejected for cp_size>1
    # because per-step localisation requires the analytical predicate, not
    # an opaque tensor. Callers that need to feed a custom `func` directly
    # would need their own slicing logic — out of scope for v0.5.
    #
    # Mirror the FBGEMM kernel invariant
    # (`cuda_hstu_attention.py:614-621`): num_contexts / num_targets must
    # come with `window_size == (-1, 0)`. This is enforced at any
    # cp_size — at cp==1 by the kernel itself, at cp>1 by us, since the
    # arbitrary-mask path could otherwise silently accept the combo while
    # the cp==1 path rejects it. Keeps behaviour identical across
    # topologies.
    ws = tuple(window_size)
    if num_contexts is not None and ws != _SUPPORTED_WINDOW_SIZE:
        raise GuardError(
            f"num_contexts requires window_size=(-1, 0); got {ws}. "
            f"Sliding window combined with contextual prefix is rejected "
            f"by the FBGEMM kernel ABI; the CP arbitrary-mask path "
            f"matches this invariant. ({_SPEC_REF})"
        )
    if num_targets is not None and ws != _SUPPORTED_WINDOW_SIZE:
        raise GuardError(
            f"num_targets requires window_size=(-1, 0); got {ws}. "
            f"Sliding window combined with target groups is rejected by "
            f"the FBGEMM kernel ABI; the CP arbitrary-mask path matches "
            f"this invariant. ({_SPEC_REF})"
        )
    if cp_size > 1:
        if func is not None:
            raise GuardError(
                f"explicit `func` tensor not plumbed through CP — Step 4a "
                f"only supports the 4-tuple mask spec (num_contexts, "
                f"num_targets, target_group_size, window_size). At "
                f"cp_size==1 the wrapper forwards `func` to the kernel "
                f"directly. ({_SPEC_REF})"
            )
    # 5. rab / has_drab — out of v0 scope at any cp_size.
    if rab is not None:
        raise GuardError(f"rab is not supported in v0 ({_SPEC_REF})")
    if has_drab:
        raise GuardError(f"has_drab=True is not supported in v0 ({_SPEC_REF})")
    # 6-9. KV cache + paging — out of v0 scope at any cp_size.
    if kv_cache is not None:
        raise GuardError(f"kv_cache is not supported in v0 ({_SPEC_REF})")
    if page_offsets is not None:
        raise GuardError(f"page_offsets is not supported in v0 ({_SPEC_REF})")
    if page_ids is not None:
        raise GuardError(f"page_ids is not supported in v0 ({_SPEC_REF})")
    if last_page_lens is not None:
        raise GuardError(f"last_page_lens is not supported in v0 ({_SPEC_REF})")
    # 11. quant_mode (only `-1` (== off) is allowed; both `None` and any other
    #     int are rejected so users can't accidentally bypass quantisation
    #     guards by leaving the kwarg unset on a build that defaults to None).
    if quant_mode is None or quant_mode != -1:
        raise GuardError(
            f"quant_mode={quant_mode!r} not supported in v0; only -1 ({_SPEC_REF})"
        )
    # 12. seqused_q/k (the kernel takes them but v0 wrapper doesn't pass through)
    if seqused_q is not None:
        raise GuardError(f"seqused_q is not supported in v0 ({_SPEC_REF})")
    if seqused_k is not None:
        raise GuardError(f"seqused_k is not supported in v0 ({_SPEC_REF})")
    # 13. head_dim
    if q.dim() != 3:
        raise GuardError(
            f"q must be 3-D (total_tokens, num_heads, head_dim); got {q.dim()}-D"
        )
    head_dim = q.shape[-1]
    if head_dim not in _SUPPORTED_HEAD_DIMS:
        raise GuardError(
            f"head_dim={head_dim} not in supported set {_SUPPORTED_HEAD_DIMS} ({_SPEC_REF})"
        )
    # Self-attention contract: cu_seqlens_q must equal cu_seqlens_k for any
    # cp_size (HSTU is self-attention; the in-tree kernel ignores cu_seqlens_k
    # for true varlen self-attn but the wrapper enforces the contract so a
    # mismatched call is caught early).
    if not torch.equal(cu_seqlens_q, cu_seqlens_k):
        raise GuardError(
            "cu_seqlens_q must equal cu_seqlens_k (HSTU is self-attention only in v0)"
        )
    # Local DualChunkSwap layout requirement: each per-sample local length is
    # `2 * c_b` (chunk_r and chunk_(2cp-1-r) concatenated), so it must be
    # even. The stronger global divisibility `L_global % (2 * cp_size) == 0`
    # is the CALLER's responsibility — `get_batch_on_this_cp_rank_for_hstu`
    # enforces it during dispatch. Re-checking the global rule here would
    # incorrectly flag the (per-rank) local cu_seqlens, since `local_len =
    # global_len / cp_size` is divisible by 2 but generally NOT by 2*cp_size.
    if cp_size > 1:
        seqlens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).tolist()
        for b, L in enumerate(seqlens):
            if L % 2 != 0:
                raise GuardError(
                    f"sample {b}: local seqlen {L} not even (each rank's "
                    f"DualChunkSwap layout is 2*c_b per sample). Verify the "
                    f"caller used `get_batch_on_this_cp_rank_for_hstu` for dispatch."
                )


# ----------------------------------------------------------------------------
# Per-tile slice helpers (varlen-aware). These mirror the validated PoC at
# `examples/hstu/cp/poc_simrank_sim.py`. Pure Python/torch — no fused CUDA
# (per SPEC §2 v0 contract; CUDA fusion is a Slice 5 follow-up if profiling
# shows it matters).
# ----------------------------------------------------------------------------
def _chunk_sizes_from_cu(cu_local: torch.Tensor) -> list[int]:
    """Each sample's chunk size c_b given the local layout (2 chunks per sample,
    total 2*c_b)."""
    seqlens = (cu_local[1:] - cu_local[:-1]).tolist()
    return [s // 2 for s in seqlens]


def _zero_second_half_per_sample(
    t: torch.Tensor, cu_local: torch.Tensor, chunk_sizes: list[int]
) -> torch.Tensor:
    """Zero the second-half (chunk_(2cp-1-src)) slot of each sample's local layout."""
    out = t.clone()
    cu = cu_local.tolist()
    for b, c_b in enumerate(chunk_sizes):
        start = cu[b] + c_b
        end = cu[b + 1]
        out[start:end] = 0
    return out


def _select_second_half_per_sample(
    t: torch.Tensor, cu_local: torch.Tensor, chunk_sizes: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Take the second-half slot per sample. Returns (concat tensor, cu_seqlens_half)."""
    cu = cu_local.tolist()
    parts: list[torch.Tensor] = []
    half_lens: list[int] = []
    for b, c_b in enumerate(chunk_sizes):
        start = cu[b] + c_b
        end = cu[b + 1]
        parts.append(t[start:end])
        half_lens.append(c_b)
    out = torch.cat(parts, dim=0).contiguous()
    cu_half = (
        torch.tensor([0] + half_lens, dtype=torch.int32, device=t.device)
        .cumsum(0)
        .int()
    )
    return out, cu_half


def _scatter_second_half_per_sample(
    out_local: torch.Tensor,
    partial_half: torch.Tensor,
    cu_local: torch.Tensor,
    chunk_sizes: list[int],
) -> None:
    """In-place add `partial_half` (B*c_b rows concatenated) into `out_local`'s
    per-sample second-half slots."""
    cu = cu_local.tolist()
    cum_half = 0
    for b, c_b in enumerate(chunk_sizes):
        start = cu[b] + c_b
        end = cu[b + 1]
        out_local[start:end] += partial_half[cum_half : cum_half + c_b]
        cum_half += c_b


# ----------------------------------------------------------------------------
# Per-tile kernel calls. All three flavours pass the GLOBAL `scaling_seqlen`
# so partial outputs across ring steps share the same normaliser (plain-sum
# remains correct).
# ----------------------------------------------------------------------------
def _diag_call(
    q_loc, k_loc, v_loc, cu_loc, local_max, scaling_seqlen, alpha
) -> torch.Tensor:
    return hstu_attn_varlen_func(
        q=q_loc,
        k=k_loc,
        v=v_loc,
        cu_seqlens_q=cu_loc,
        cu_seqlens_k=cu_loc,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=local_max,
        max_seqlen_k=local_max,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        alpha=alpha,
        quant_mode=-1,
    )


def _lower_call(
    q_loc, k_pad, v_pad, cu_loc, local_max, scaling_seqlen, alpha
) -> torch.Tensor:
    return hstu_attn_varlen_func(
        q=q_loc,
        k=k_pad,
        v=v_pad,
        cu_seqlens_q=cu_loc,
        cu_seqlens_k=cu_loc,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=local_max,
        max_seqlen_k=local_max,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, -1),
        alpha=alpha,
        quant_mode=-1,
    )


def _upper_call(
    q_half,
    k_full,
    v_full,
    cu_q_half,
    cu_full,
    half_max,
    local_max,
    scaling_seqlen,
    alpha,
) -> torch.Tensor:
    return hstu_attn_varlen_func(
        q=q_half,
        k=k_full,
        v=v_full,
        cu_seqlens_q=cu_q_half,
        cu_seqlens_k=cu_full,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=half_max,
        max_seqlen_k=local_max,
        scaling_seqlen=scaling_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, -1),
        alpha=alpha,
        quant_mode=-1,
    )


# ----------------------------------------------------------------------------
# Ring P2P helper.
#
# Slice 5 adds two-stream overlap: when `comm_stream` is provided, NCCL P2P
# is launched on that stream so the kernel scheduler is free to overlap with
# attention compute on the default stream. Single-stream call sites still
# work — pass `comm_stream=None` (or omit) and the helper degenerates to the
# Slice 3 behaviour.
# ----------------------------------------------------------------------------
def _ring_send_recv_kv(
    cur_k: torch.Tensor,
    cur_v: torch.Tensor,
    recv_k: torch.Tensor,
    recv_v: torch.Tensor,
    *,
    cp_group: dist.ProcessGroup,
    cp_global_ranks: Sequence[int],
    cp_rank: int,
    cp_size: int,
    direction: str = "forward",
    comm_stream: Optional[torch.cuda.Stream] = None,
) -> list[dist.Work]:
    """Issue P2P send + recv for one ring step.

    `direction="forward"`: send to `(rank+1)`, recv from `(rank-1)`.
    `direction="backward"`: send to `(rank-1)`, recv from `(rank+1)`. Used by
    T4.2 (multi-GPU backward) to send dKV partials home along the reverse
    ring. Note that for backward, the tensors typically named `cur_k/cur_v`
    actually carry dK/dV gradients — the helper is direction-agnostic.

    `comm_stream`: optional secondary CUDA stream on which to launch the NCCL
    collective. If provided, the caller is responsible for the cross-stream
    sync (`comm_stream.wait_stream(default)` BEFORE the call so the comm
    stream sees the producer writes; `default.wait_stream(comm_stream)` AFTER
    the work has waited so the consumer stream sees the recv'd bytes). When
    None, the collective runs on whatever stream is current at call time —
    Slice 3 behaviour.

    Uses `batch_isend_irecv` to avoid the deadlock pattern of naive isend/irecv
    pairs. Returns the list of `Work` handles; caller must call `.wait()`
    before consuming `recv_k`/`recv_v`.
    """
    if direction == "forward":
        dst = cp_global_ranks[(cp_rank + 1) % cp_size]
        src = cp_global_ranks[(cp_rank - 1) % cp_size]
    elif direction == "backward":
        dst = cp_global_ranks[(cp_rank - 1) % cp_size]
        src = cp_global_ranks[(cp_rank + 1) % cp_size]
    else:
        raise ValueError(
            f"direction must be 'forward' or 'backward'; got {direction!r}"
        )

    def _issue_two_pairs() -> list[dist.Work]:
        # NCCL `batch_isend_irecv` reliably handles a single send/recv pair
        # per batch. Bundling K and V into a 4-op batch hangs in some
        # NCCL/torch combinations (verified on the production image's NCCL
        # on A100 PCIe). Issue K and V as two separate 2-op batches; caller
        # gets one combined Work list to wait on.
        rk = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, cur_k, dst, group=cp_group),
                dist.P2POp(dist.irecv, recv_k, src, group=cp_group),
            ]
        )
        rv = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, cur_v, dst, group=cp_group),
                dist.P2POp(dist.irecv, recv_v, src, group=cp_group),
            ]
        )
        return list(rk) + list(rv)

    if comm_stream is None:
        return _issue_two_pairs()
    with torch.cuda.stream(comm_stream):
        return _issue_two_pairs()


# ----------------------------------------------------------------------------
# T3.3 + T5.1: multi-GPU forward. Two-stream comm/compute overlap.
#
# Step `i` issues NCCL P2P for step `i+1`'s KV on `cp_stream` while attention
# compute for step `i` runs on the default stream. Cross-stream sync via
# `wait_stream` on both sides:
#   - cp_stream.wait_stream(default) BEFORE P2P, so P2P sees the latest writes
#     to `cur_k/cur_v` (the initial clone, or the previous iteration's swap).
#   - default.wait_stream(cp_stream) AFTER `r.wait()` and BEFORE the next-step
#     compute consumes the swapped-in tensors.
# ----------------------------------------------------------------------------
def _multi_gpu_forward(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    *,
    max_seqlen_q_global: int,
    scaling_seqlen: int,
    alpha: float,
    cp_group: dist.ProcessGroup,
    cp_global_ranks: Sequence[int],
    cp_rank: int,
    cp_size: int,
    cp_stream: Optional["torch.cuda.Stream"] = None,
) -> torch.Tensor:
    """Run the (rank, step) classification grid as a real multi-GPU ring.

    `q_local, k_local, v_local, cu_seqlens_local` are this rank's DualChunkSwap
    shard (already produced by `get_batch_on_this_cp_rank_for_hstu` upstream).
    `max_seqlen_q_global` is the unsharded global max; we compute `local_max`
    internally. `scaling_seqlen` is the global `1/N` divisor (must NOT change
    across ring steps — that's why every per-tile call passes the same value).

    `cp_stream`: secondary CUDA stream for NCCL P2P. If None, we lazily create
    or reuse a per-device cached stream (see `_get_cp_stream`).

    Reduction is in fp32 (per SPEC §2). The returned tensor is cast back to
    `q_local.dtype` on exit.
    """
    local_max = (
        max_seqlen_q_global // cp_size
    )  # 2 chunks per sample → local len = global / cp_size
    half_max = local_max // 2  # one chunk per sample
    chunk_sizes = _chunk_sizes_from_cu(cu_seqlens_local)

    default_stream = torch.cuda.current_stream()
    comm_stream = _get_cp_stream(q_local.device, cp_stream)

    # Ping-pong KV buffers. Critical: clone the initial K/V so subsequent
    # buffer swaps never mutate the caller's input tensors. Without the clone,
    # after step 0's swap `recv_k` becomes the original `k_local`, and step 1's
    # P2P would write peer KV into the user's input — silent data corruption.
    cur_k = k_local.clone()
    cur_v = v_local.clone()
    recv_k = torch.empty_like(k_local)
    recv_v = torch.empty_like(v_local)

    # Output accumulator in fp32 for numerical stability across cp_size adds.
    out_local = torch.zeros_like(q_local, dtype=torch.float32)

    for step in range(cp_size):
        # 1. Issue next-step KV exchange on `comm_stream` (skip on last step).
        # The comm stream waits for the producer of cur_k/cur_v, which is the
        # default stream (initial clone in step 0; swap-stamped writes in
        # steps > 0 originate from comm_stream itself, but we resync for
        # safety since the swap is just a Python pointer rebind).
        reqs: list[dist.Work] = []
        if step < cp_size - 1:
            comm_stream.wait_stream(default_stream)
            reqs = _ring_send_recv_kv(
                cur_k,
                cur_v,
                recv_k,
                recv_v,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_rank=cp_rank,
                cp_size=cp_size,
                comm_stream=comm_stream,
            )

        # 2. Compute on the current KV (still owned), on default stream.
        if step == 0:
            partial = _diag_call(
                q_local,
                cur_k,
                cur_v,
                cu_seqlens_local,
                local_max,
                scaling_seqlen,
                alpha,
            )
            out_local += partial.float()
        elif step <= cp_rank:
            # Lower-tri: zero peer's second-half (chunk_(2cp-1-src)) so
            # K_len == Q_len; SiLU(α Q · 0) · 0 contributes 0.
            k_pad = _zero_second_half_per_sample(cur_k, cu_seqlens_local, chunk_sizes)
            v_pad = _zero_second_half_per_sample(cur_v, cu_seqlens_local, chunk_sizes)
            partial = _lower_call(
                q_local,
                k_pad,
                v_pad,
                cu_seqlens_local,
                local_max,
                scaling_seqlen,
                alpha,
            )
            out_local += partial.float()
        else:
            # Upper-tri: Q's second-half (chunk_(2cp-1-rank)) × peer's full K.
            q_half, cu_q_half = _select_second_half_per_sample(
                q_local, cu_seqlens_local, chunk_sizes
            )
            partial_half = _upper_call(
                q_half,
                cur_k,
                cur_v,
                cu_q_half,
                cu_seqlens_local,
                half_max,
                local_max,
                scaling_seqlen,
                alpha,
            )
            _scatter_second_half_per_sample(
                out_local, partial_half.float(), cu_seqlens_local, chunk_sizes
            )

        # 3. Wait for next-step KV to arrive before overwriting `cur_*`.
        # `Work.wait()` does CPU-blocking; default_stream.wait_stream(comm_stream)
        # is the GPU-side ordering — default sees the recv'd buffer contents
        # before the next iteration's compute consumes them.
        if step < cp_size - 1:
            for r in reqs:
                r.wait()
            default_stream.wait_stream(comm_stream)
            # 4. Swap buffers for the next iteration.
            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    return out_local.to(q_local.dtype)


# ----------------------------------------------------------------------------
# Het-mask Step 4a — multi-GPU forward via arbitrary `func` mask.
#
# Replaces the 3-region (diag/lower-tri/upper-tri) tile classifier with a
# single kernel call per ring step. Each step builds a per-step `func`
# tensor via `localize_func_for_cp_step` that encodes the global mask
# (causal + targets + group + contextual + sliding) projected onto the
# (local_q × peer_k) layout. The kernel runs full local Q × full peer K
# with `window_size=(-1, -1)` (structured mask disabled) and `func`
# carrying every constraint.
#
# Net effect: the wrapper handles arbitrary HSTU mask families under CP
# without zero-padding tricks or half-Q kernel calls. Plain-causal goes
# through this path too (with the func encoding [0, q+1) per row), and
# is exercised by the existing CP regression suite via Step 4 dispatch.
# ----------------------------------------------------------------------------
def _multi_gpu_forward_arbitrary(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    cu_seqlens_global: torch.Tensor,
    *,
    max_seqlen_q_global: int,
    scaling_seqlen: int,
    alpha: float,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
    cp_group: dist.ProcessGroup,
    cp_global_ranks: Sequence[int],
    cu_seqlens_global_tuple: Optional[tuple] = None,
    num_contexts_tuple: Optional[tuple] = None,
    num_targets_tuple: Optional[tuple] = None,
    cp_rank: int,
    cp_size: int,
    cp_stream: Optional["torch.cuda.Stream"] = None,
) -> torch.Tensor:
    """Multi-GPU forward via FBGEMM arbitrary-mask `func` per ring step."""
    local_max = max_seqlen_q_global // cp_size

    default_stream = torch.cuda.current_stream()
    comm_stream = _get_cp_stream(q_local.device, cp_stream)

    # Ping-pong KV buffers. recv_k/recv_v come from the module-level
    # pool — same physical buffer is reused across all 8 layers'
    # forwards. Stream-correctness is preserved by the
    # `comm_stream.wait_stream(default_stream)` issued before each
    # NCCL P2P (waits for the previous layer's last kernel read on the
    # buffer) and by `default_stream.wait_stream(comm_stream)` after
    # the wait (read can see NCCL's writes).
    #
    # `cur_k`/`cur_v` are kept as fresh `clone()` even for cp_size==2.
    # We tried `cur_k = k_local` (no clone) in round-12 to save a
    # ~56 MB memcpy per layer; NCCL's batch_isend_irecv rejected it
    # with `ValueError: Tensors for P2P must be non-overlapping and
    # dense`, presumably because k_local is the autograd Function's
    # input and shares storage / version metadata that NCCL's
    # validator doesn't accept as a P2P tensor. The clone is mandatory.
    cur_k = k_local.clone()
    cur_v = v_local.clone()
    recv_k = _get_recv_buffer(k_local, slot="fwd_recv_k")
    recv_v = _get_recv_buffer(v_local, slot="fwd_recv_v")

    # Allocate the fp32 accumulator BEFORE the loop (precision contract +
    # an implicit allocator-side stream barrier that prevents the iter-400
    # crash — see commit log for round-15-debug, where moving this alloc
    # into the loop with a lazy-init pattern triggered a CUDA illegal
    # memory access at iter ~400 unless `CUDA_LAUNCH_BLOCKING=1` was set).
    # We use `empty_like` + first-iter `copy_` instead of `zeros_like`
    # + `+=` to skip the ~112 MB memset (saves ~50 µs / accumulator ×
    # 8 layers ≈ 0.4 ms / training step on the forward path; same
    # pattern in backward saves ~1.2 ms / training step across the 3
    # gradient accumulators).
    out_local = torch.empty_like(q_local, dtype=torch.float32)

    for step in range(cp_size):
        # 1. Issue next-step KV exchange.
        reqs: list[dist.Work] = []
        if step < cp_size - 1:
            comm_stream.wait_stream(default_stream)
            reqs = _ring_send_recv_kv(
                cur_k,
                cur_v,
                recv_k,
                recv_v,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_rank=cp_rank,
                cp_size=cp_size,
                comm_stream=comm_stream,
            )

        # 2. Build per-step `func` tensor and run the kernel. The
        # `_cached_localize_func_for_cp_step` reuses across all layers
        # in the same `HSTUBlock.forward` scope (set up via
        # `cp_func_cache_scope_enter/exit`). Major perf win:
        # 8 layers × cp_size builds → cp_size builds per training step.
        # Pre-computed tuple keys (when provided) skip ~30 redundant
        # CUDA syncs per training step (one per layer × ring step).
        func = _cached_localize_func_for_cp_step(
            cu_seqlens_global=cu_seqlens_global,
            cp_size=cp_size,
            cp_rank=cp_rank,
            step=step,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            window_size=window_size,
            NFUNC=3,
            device=q_local.device,
            cu_seqlens_global_tuple=cu_seqlens_global_tuple,
            num_contexts_tuple=num_contexts_tuple,
            num_targets_tuple=num_targets_tuple,
        )
        partial = hstu_attn_varlen_func(
            q=q_local,
            k=cur_k,
            v=cur_v,
            cu_seqlens_q=cu_seqlens_local,
            cu_seqlens_k=cu_seqlens_local,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=local_max,
            max_seqlen_k=local_max,
            scaling_seqlen=scaling_seqlen,
            num_contexts=None,  # `func` carries the full mask;
            num_targets=None,  # structured layer disabled to avoid
            target_group_size=1,  # double-application (kernel intersects
            window_size=(-1, -1),  # structured + arbitrary).
            alpha=alpha,
            func=func,
            quant_mode=-1,
        )
        # Step 0 SETS the (uninitialised) accumulator via copy_; step 1+
        # accumulates with +=. We keep the explicit `.float()` cast on
        # both branches so the allocator-side stream events emitted by
        # the fp32 temp allocation are identical to round-7 — that's
        # the implicit barrier round-15-debug showed is necessary to
        # prevent the iter-400 stream race.
        if step == 0:
            out_local.copy_(partial.float())
        else:
            out_local += partial.float()

        # 3. Wait for P2P + swap.
        if step < cp_size - 1:
            for r in reqs:
                r.wait()
            default_stream.wait_stream(comm_stream)
            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    return out_local.to(q_local.dtype)


# ----------------------------------------------------------------------------
# Direct kernel-bwd helper — bypasses the `enable_grad → autograd.grad`
# replay-forward that the legacy `_per_tile_partial_grads` uses. The
# replay was wasteful because:
#   1. The kernel's own bwd op (`hstu_varlen_bwd_*`) ALREADY recomputes
#      forward internally — its `ctx.saved_tensors` only stores the
#      inputs, not the forward output, so it must recompute.
#   2. Re-running `hstu_attn_varlen_func` (an autograd.Function) inside
#      `enable_grad` builds a fresh graph + 3 fresh tensors via
#      `q.detach().clone().requires_grad_(True)` per ring step. For
#      cp_size=2 / 8 layers / typical (L=2048, B=8, H=8, D=128) that's
#      ~48 MB × 16 ring steps ≈ 768 MB of transient allocs per step —
#      and an extra forward kernel call we throw away the result of.
#
# Calling the kernel bwd op directly skips both. Net saving on
# cw-dfw round-5: ~80 ms / step (165 → ~85 ms) — the last gap to
# the 80% DP4-parity target.
# ----------------------------------------------------------------------------
def _call_hstu_bwd_kernel(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    dout: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    scaling_seqlen: int,
    alpha: float,
    window_size: tuple[int, int],
    func: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Direct call to `torch.ops.fbgemm.hstu_varlen_bwd_*`.

    Mirrors the kernel half of `HstuAttnVarlenFunc.backward` (in the
    installed `hstu` package) for the v0 CP feature subset:
      - no FP8 quantisation (quant_mode == -1),
      - no `rab` / `has_drab`,
      - no `seqused_q` / `seqused_k`,
      - no kv_cache.

    Dispatch by GPU SM (Ampere-80 / Hopper-90 / Blackwell-100) — same
    branch as the upstream wrapper.
    """
    major_version = torch.cuda.get_device_capability(q.device)[0]
    window_left, window_right = window_size

    if major_version == 8:
        dq, dk, dv, _dRab = torch.ops.fbgemm.hstu_varlen_bwd_80(
            dout,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen_q,
            max_seqlen_k,
            scaling_seqlen,
            None,  # positions 12-14: unused placeholders kept None
            None,  #   to match the upstream fwd/bwd_80 op signature
            None,  #   exactly (see HstuAttnVarlenFunc.backward).
            None,  # num_contexts (CP wrapper carries het-mask via `func`)
            None,  # num_targets
            1,  # target_group_size (already absorbed into `func`)
            window_left,
            window_right,
            alpha,
            None,  # rab
            False,  # has_drab
            func,
            False,  # deterministic
        )
    elif major_version == 9:
        dq, dk, dv, _dRab = torch.ops.fbgemm.hstu_varlen_bwd_90(
            dout,
            None,  # dout_t (FP8 only)
            q,
            None,  # qt
            k,
            None,  # kt
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            None,  # seqused_q
            None,  # seqused_k
            max_seqlen_q,
            max_seqlen_k,
            scaling_seqlen,
            None,
            None,
            None,
            None,  # num_contexts
            None,  # num_targets
            1,  # target_group_size
            window_left,
            window_right,
            alpha,
            -1,  # quant_mode
            None,  # rab_padded
            False,  # has_drab
            func,
            None,  # q_descale
            None,  # qt_descale
            None,  # k_descale
            None,  # kt_descale
            None,  # v_descale
            None,  # do_descale
            None,  # dot_descale
            None,  # cu_seqlens_qt_descale
            None,  # cu_seqlens_kt_descale
            None,  # cu_seqlens_q_block_descale
            None,  # cu_seqlens_kv_block_descale
            0 if dout.dtype == torch.bfloat16 else 1,  # output_dtype
            False,  # deterministic
        )
    elif major_version == 10:
        from fbgemm_gpu.experimental.hstu.hstu_blackwell import hstu_ops_gpu as _sm100

        dq, dk, dv, _dRab = _sm100.hstu_varlen_bwd_100(
            dout,
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            None,
            None,
            None,
            None,  # num_contexts
            None,  # num_targets
            1,  # target_group_size
            window_left,
            window_right,
            alpha,
            None,  # rab_padded
            False,  # has_drab
            func,
            False,  # deterministic
        )
    else:
        raise RuntimeError(
            f"Unsupported GPU compute capability {major_version}.x; "
            f"hstu kernel ships sm80, sm90, sm100"
        )
    return dq, dk, dv


# ----------------------------------------------------------------------------
# T4.2: multi-GPU backward. Reverse-direction ring; dQ stays local; dK/dV
# partials ride the reverse ring back to their owning rank with copy-on-first /
# add-after semantics.
# ----------------------------------------------------------------------------
def _per_tile_partial_grads(
    q_input: torch.Tensor,
    k_input: torch.Tensor,
    v_input: torch.Tensor,
    cu_q: torch.Tensor,
    cu_k: torch.Tensor,
    *,
    max_q: int,
    max_k: int,
    scaling_seqlen: int,
    alpha: float,
    window_size: tuple[int, int],
    dout_partial: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run a per-tile forward with autograd on the LOCAL (detached + cloned)
    inputs, then call torch.autograd.grad to extract partial dQ, dK, dV.

    The wrapping of `hstu_attn_varlen_func` already has its own
    `autograd.Function`, so this is just a thin re-execution that propagates
    `dout_partial` back through it.

    `torch.enable_grad()` is required because we are called from inside
    `_HSTUVarlenCPFunc.backward`, where torch by default disables grad mode.
    Without enabling, the `requires_grad_(True)` flag would be ignored and
    `torch.autograd.grad` would error out on a graph-less output.
    """
    with torch.enable_grad():
        q_in = q_input.detach().clone().requires_grad_(True)
        k_in = k_input.detach().clone().requires_grad_(True)
        v_in = v_input.detach().clone().requires_grad_(True)
        out = hstu_attn_varlen_func(
            q=q_in,
            k=k_in,
            v=v_in,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_q,
            max_seqlen_k=max_k,
            scaling_seqlen=scaling_seqlen,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=window_size,
            alpha=alpha,
            quant_mode=-1,
        )
        dq, dk, dv = torch.autograd.grad(out, (q_in, k_in, v_in), dout_partial)
    return dq.detach(), dk.detach(), dv.detach()


def _multi_gpu_backward(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    dout_local: torch.Tensor,
    *,
    max_seqlen_q_global: int,
    scaling_seqlen: int,
    alpha: float,
    cp_group: dist.ProcessGroup,
    cp_global_ranks: Sequence[int],
    cp_rank: int,
    cp_size: int,
    cp_stream: Optional["torch.cuda.Stream"] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse-direction-ring backward for HSTU CP forward.

    Algorithm:
      1. Initialise dq_local = 0, dk_local = 0, dv_local = 0.
      2. For each forward step `i` in 0..cp_size-1, redo the per-tile forward
         with autograd-enabled inputs and use `torch.autograd.grad` to extract
         (dq_partial, dk_partial, dv_partial).
         dq_partial accumulates locally; dk_partial / dv_partial are gradients
         for the KV that was held at this step (peer rank src=(rank-i)%cp_size).
      3. dKV partials ride the **reverse** ring back to their owners:
         - At backward iteration 0, the rank holds its OWN dKV (step==0
           diagonal tile). Add to local dk/dv directly.
         - For iteration j>=1, the rank sends the dKV computed at forward
           step j (which belongs to peer src=(rank-j)%cp) to that peer
           via reverse ring (dst = (rank - j) % cp_size, equivalently
           dst = src). Receives from rank `(rank + j) % cp_size` the dKV
           that they computed for OUR K/V at their forward step j.
         - The received dKV adds to local dk_local, dv_local.

    Returned dtypes match the forward inputs (cast back from the fp32
    accumulators).
    """
    local_max = max_seqlen_q_global // cp_size
    half_max = local_max // 2
    chunk_sizes = _chunk_sizes_from_cu(cu_seqlens_local)

    default_stream = torch.cuda.current_stream()
    comm_stream = _get_cp_stream(q_local.device, cp_stream)

    # We need the SAME KV stream the forward saw at each step. Re-run the
    # forward ring locally (read-only) to reconstruct kv_at_step[i]. Cheap
    # because comm dominates and we already paid that cost in forward.
    # Clone to avoid the same aliasing hazard as in `_multi_gpu_forward`
    # (buffer swap would otherwise mutate the saved-for-backward k_local/v_local).
    cur_k = k_local.clone()
    cur_v = v_local.clone()
    recv_k = torch.empty_like(k_local)
    recv_v = torch.empty_like(v_local)

    # fp32 accumulators (per SPEC §2 "Reduction in fp32").
    dq_acc = torch.zeros_like(q_local, dtype=torch.float32)
    dk_acc = torch.zeros_like(k_local, dtype=torch.float32)
    dv_acc = torch.zeros_like(v_local, dtype=torch.float32)

    # We collect (step, dk_partial, dv_partial) so that after the forward-pass
    # backward computation, we send each dKV back to its rightful owner via
    # the reverse ring. dq is purely local — accumulated inline.
    # dk/dv at forward step 0 belong to rank itself; add directly.
    dkv_to_send: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    for step in range(cp_size):
        # Issue next-step KV exchange (forward direction) so cur_k/v matches
        # what was used in forward. Same two-stream pattern as
        # `_multi_gpu_forward` — comm overlaps with `_per_tile_partial_grads`.
        reqs: list[dist.Work] = []
        if step < cp_size - 1:
            comm_stream.wait_stream(default_stream)
            reqs = _ring_send_recv_kv(
                cur_k,
                cur_v,
                recv_k,
                recv_v,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_rank=cp_rank,
                cp_size=cp_size,
                direction="forward",
                comm_stream=comm_stream,
            )

        # Compute per-tile partial grads.
        if step == 0:
            # Diagonal: full Q × full local KV, causal.
            dq_p, dk_p, dv_p = _per_tile_partial_grads(
                q_local,
                cur_k,
                cur_v,
                cu_seqlens_local,
                cu_seqlens_local,
                max_q=local_max,
                max_k=local_max,
                scaling_seqlen=scaling_seqlen,
                alpha=alpha,
                window_size=(-1, 0),
                dout_partial=dout_local,
            )
            dq_acc += dq_p.float()
            # Diagonal dKV is ours; add to local accumulator immediately.
            dk_acc += dk_p.float()
            dv_acc += dv_p.float()
        elif step <= cp_rank:
            # Lower-tri: full Q × peer K first-half (zero-padded), no causal.
            k_pad = _zero_second_half_per_sample(cur_k, cu_seqlens_local, chunk_sizes)
            v_pad = _zero_second_half_per_sample(cur_v, cu_seqlens_local, chunk_sizes)
            dq_p, dk_p, dv_p = _per_tile_partial_grads(
                q_local,
                k_pad,
                v_pad,
                cu_seqlens_local,
                cu_seqlens_local,
                max_q=local_max,
                max_k=local_max,
                scaling_seqlen=scaling_seqlen,
                alpha=alpha,
                window_size=(-1, -1),
                dout_partial=dout_local,
            )
            dq_acc += dq_p.float()
            # The padded second-half slots received zero contribution in
            # forward, so dK/dV at those positions is exactly 0. Zero them
            # again for safety before sending back.
            dk_p_for_peer = _zero_second_half_per_sample(
                dk_p, cu_seqlens_local, chunk_sizes
            )
            dv_p_for_peer = _zero_second_half_per_sample(
                dv_p, cu_seqlens_local, chunk_sizes
            )
            dkv_to_send.append((step, dk_p_for_peer, dv_p_for_peer))
        else:
            # Upper-tri: Q's second-half × full peer K, no mask.
            q_half, cu_q_half = _select_second_half_per_sample(
                q_local, cu_seqlens_local, chunk_sizes
            )
            dout_half, _ = _select_second_half_per_sample(
                dout_local, cu_seqlens_local, chunk_sizes
            )
            dq_half_p, dk_p, dv_p = _per_tile_partial_grads(
                q_half,
                cur_k,
                cur_v,
                cu_q_half,
                cu_seqlens_local,
                max_q=half_max,
                max_k=local_max,
                scaling_seqlen=scaling_seqlen,
                alpha=alpha,
                window_size=(-1, -1),
                dout_partial=dout_half,
            )
            # Scatter dq_half_p into rank's second-half slots.
            _scatter_second_half_per_sample(
                dq_acc, dq_half_p.float(), cu_seqlens_local, chunk_sizes
            )
            # dk_p, dv_p are full local-shape (matching cur_k/v). Send back.
            dkv_to_send.append((step, dk_p, dv_p))

        # Wait for next-step KV (forward ring) to arrive before swap.
        if step < cp_size - 1:
            for r in reqs:
                r.wait()
            default_stream.wait_stream(comm_stream)
            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    # 3. Reverse-ring exchange of dKV partials. For each (step, dk_p, dv_p)
    # in `dkv_to_send`, dk_p / dv_p belong to peer src = (rank - step) % cp.
    # We send to dst=src and receive from peer that computed grads for our
    # K/V at THEIR step (which is our same step index).
    #
    # Why simultaneous: at step `i`, every rank computed grads-for-its-peer
    # at the same `i`. So each rank's `i`-step partial → dst=(rank-i)%cp;
    # each rank's `i`-step incoming → from src=(rank+i)%cp.
    recv_dk = torch.empty_like(k_local)
    recv_dv = torch.empty_like(v_local)
    for step, dk_p, dv_p in dkv_to_send:
        send_dst = cp_global_ranks[(cp_rank - step) % cp_size]
        recv_src = cp_global_ranks[(cp_rank + step) % cp_size]
        # Two separate 2-op batches: see _ring_send_recv_kv comment for why a
        # 4-op batch hangs on the production NCCL.
        reqs_k = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, dk_p.contiguous(), send_dst, group=cp_group),
                dist.P2POp(dist.irecv, recv_dk, recv_src, group=cp_group),
            ]
        )
        for r in reqs_k:
            r.wait()
        reqs_v = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, dv_p.contiguous(), send_dst, group=cp_group),
                dist.P2POp(dist.irecv, recv_dv, recv_src, group=cp_group),
            ]
        )
        for r in reqs_v:
            r.wait()
        dk_acc += recv_dk.float()
        dv_acc += recv_dv.float()

    return (
        dq_acc.to(q_local.dtype),
        dk_acc.to(k_local.dtype),
        dv_acc.to(v_local.dtype),
    )


# ----------------------------------------------------------------------------
# Het-mask Step 4b — multi-GPU backward via arbitrary `func` mask.
#
# Mirrors `_multi_gpu_forward_arbitrary`: single kernel call per ring step,
# `func` carries the full mask. Reverse-ring dKV exchange mirrors
# `_multi_gpu_backward`.
# ----------------------------------------------------------------------------
def _multi_gpu_backward_arbitrary(
    q_local: torch.Tensor,
    k_local: torch.Tensor,
    v_local: torch.Tensor,
    cu_seqlens_local: torch.Tensor,
    cu_seqlens_global: torch.Tensor,
    dout_local: torch.Tensor,
    *,
    max_seqlen_q_global: int,
    scaling_seqlen: int,
    alpha: float,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
    cp_group: dist.ProcessGroup,
    cp_global_ranks: Sequence[int],
    cp_rank: int,
    cp_size: int,
    cp_stream: Optional["torch.cuda.Stream"] = None,
    cu_seqlens_global_tuple: Optional[tuple] = None,
    num_contexts_tuple: Optional[tuple] = None,
    num_targets_tuple: Optional[tuple] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reverse-ring backward for the arbitrary-mask CP forward."""
    local_max = max_seqlen_q_global // cp_size

    default_stream = torch.cuda.current_stream()
    comm_stream = _get_cp_stream(q_local.device, cp_stream)

    # Re-run the forward ring locally to reconstruct kv_at_step[i]. Same
    # ping-pong + clone pattern as the fwd path. Distinct slot names
    # from the forward path so fwd's autograd-saved clones and bwd's
    # recv buffers cannot collide (PyTorch autograd may run bwd on a
    # worker thread, but each pool slot is a single physical buffer —
    # sharing across fwd/bwd would invite a race).
    #
    # Same NCCL-validator constraint as the fwd path: `cur_k = k_local`
    # without clone is rejected at P2P time with `ValueError: Tensors
    # for P2P must be non-overlapping and dense`. The clone is
    # mandatory.
    cur_k = k_local.clone()
    cur_v = v_local.clone()
    recv_k = _get_recv_buffer(k_local, slot="bwd_recv_k")
    recv_v = _get_recv_buffer(v_local, slot="bwd_recv_v")

    # Pre-allocate the fp32 accumulators with `empty_like` (no memset);
    # step 0 of the main loop populates them via `copy_(...float())`,
    # subsequent contributions use `+=` as before. Saves three ~112 MB
    # memsets per layer × 8 layers ≈ 1.2 ms / training step. Allocator
    # is hit BEFORE the loop — preserves the implicit alloc-side stream
    # event that round-15-debug showed is necessary to avoid the
    # iter-400 stream-race CUDA illegal-memory-access.
    dq_acc = torch.empty_like(q_local, dtype=torch.float32)
    dk_acc = torch.empty_like(k_local, dtype=torch.float32)
    dv_acc = torch.empty_like(v_local, dtype=torch.float32)

    # Collect (step, dk_partial, dv_partial) for the reverse-ring exchange.
    # step==0 dKV is owned by self → added directly. step≥1 dKV belongs to
    # peer src=(rank-step)%cp.
    dkv_to_send: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    for step in range(cp_size):
        reqs: list[dist.Work] = []
        if step < cp_size - 1:
            comm_stream.wait_stream(default_stream)
            reqs = _ring_send_recv_kv(
                cur_k,
                cur_v,
                recv_k,
                recv_v,
                cp_group=cp_group,
                cp_global_ranks=cp_global_ranks,
                cp_rank=cp_rank,
                cp_size=cp_size,
                direction="forward",
                comm_stream=comm_stream,
            )

        # Per-step `func` tensor (cached: forward already built it for
        # this scope, so backward gets a hit and avoids re-building).
        # Pre-computed tuple keys (when provided) skip the redundant
        # CUDA syncs that hashing the tensor would do per layer × step.
        func = _cached_localize_func_for_cp_step(
            cu_seqlens_global=cu_seqlens_global,
            cp_size=cp_size,
            cp_rank=cp_rank,
            step=step,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            window_size=window_size,
            NFUNC=3,
            device=q_local.device,
            cu_seqlens_global_tuple=cu_seqlens_global_tuple,
            num_contexts_tuple=num_contexts_tuple,
            num_targets_tuple=num_targets_tuple,
        )
        # Direct kernel-bwd call. Skips the replay-forward + autograd.grad
        # pattern the legacy `_per_tile_partial_grads` uses (see
        # `_call_hstu_bwd_kernel` docstring for why that pattern is
        # wasteful — saves ~80 ms / step on cw-dfw).
        dq_p, dk_p, dv_p = _call_hstu_bwd_kernel(
            q=q_local,
            k=cur_k,
            v=cur_v,
            cu_seqlens_q=cu_seqlens_local,
            cu_seqlens_k=cu_seqlens_local,
            dout=dout_local,
            max_seqlen_q=local_max,
            max_seqlen_k=local_max,
            scaling_seqlen=scaling_seqlen,
            alpha=alpha,
            window_size=(-1, -1),
            func=func,
        )
        # Step 0 SETS the (uninitialised) accumulators via copy_; step 1+
        # uses += into the already-populated dq_acc. Keep the explicit
        # `.float()` cast so the allocator-side fp32 temp pattern is
        # identical to round-7 (the implicit barrier round-15-debug
        # confirmed is needed to prevent the iter-400 stream race).
        if step == 0:
            dq_acc.copy_(dq_p.float())
            # peer == self; dK/dV are ours.
            dk_acc.copy_(dk_p.float())
            dv_acc.copy_(dv_p.float())
        else:
            dq_acc += dq_p.float()
            dkv_to_send.append((step, dk_p, dv_p))

        if step < cp_size - 1:
            for r in reqs:
                r.wait()
            default_stream.wait_stream(comm_stream)
            cur_k, recv_k = recv_k, cur_k
            cur_v, recv_v = recv_v, cur_v

    # Reverse-ring dKV exchange (same logic as `_multi_gpu_backward`):
    # send dKV computed at step i to peer (cp_rank - i) % cp_size; receive
    # from peer (cp_rank + i) % cp_size who computed dKV for OUR KV at
    # their step i. Distinct pool slot from the bwd KV-reload path so
    # the two reverse rings don't collide on the same buffer mid-step.
    recv_dk = _get_recv_buffer(k_local, slot="bwd_recv_dk")
    recv_dv = _get_recv_buffer(v_local, slot="bwd_recv_dv")
    for step, dk_p, dv_p in dkv_to_send:
        send_dst = cp_global_ranks[(cp_rank - step) % cp_size]
        recv_src = cp_global_ranks[(cp_rank + step) % cp_size]
        # Issue K-pair and V-pair as TWO separate 2-op batches but
        # WITHOUT waiting on K before issuing V — lets NCCL queue both
        # transfers and overlap them at the device level. We still
        # avoid the 4-op single batch which the comment in
        # `_multi_gpu_backward` notes hangs on the production NCCL.
        # Wait once at the end before consuming the recv buffers.
        reqs_k = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, dk_p.contiguous(), send_dst, group=cp_group),
                dist.P2POp(dist.irecv, recv_dk, recv_src, group=cp_group),
            ]
        )
        reqs_v = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, dv_p.contiguous(), send_dst, group=cp_group),
                dist.P2POp(dist.irecv, recv_dv, recv_src, group=cp_group),
            ]
        )
        for r in reqs_k:
            r.wait()
        for r in reqs_v:
            r.wait()
        dk_acc += recv_dk.float()
        dv_acc += recv_dv.float()

    return (
        dq_acc.to(q_local.dtype),
        dk_acc.to(k_local.dtype),
        dv_acc.to(v_local.dtype),
    )


# ----------------------------------------------------------------------------
# Multi-GPU autograd Function. Forward (T3.3) and backward (T4.2) implemented.
# ----------------------------------------------------------------------------
def _is_het_mask(
    *,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
) -> bool:
    """DEPRECATED: kept only for tests that read the predicate directly.

    Pre-`func` design: this returned True iff the mask required the
    arbitrary-mask path. Pure-causal went through the legacy
    `_multi_gpu_forward` (4 ops/ring step with zero-pad + select
    workarounds). Now ALL CP paths route through the
    `func`-based `_multi_gpu_forward_arbitrary` (1 kernel/ring step) —
    the FBGEMM HSTU kernel's arbitrary-mask interface is more
    flexible than flash-attn's `is_causal`, so the legacy
    tile-classification trick was technical debt from before the
    het-mask track existed.
    """
    if num_contexts is not None:
        return True
    if num_targets is not None:
        return True
    if target_group_size != 1:
        return True
    if tuple(window_size) != _SUPPORTED_WINDOW_SIZE:
        return True
    return False


class _HSTUVarlenCPFunc(torch.autograd.Function):
    """Multi-GPU forward+backward driver."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scaling_seqlen,
        alpha,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        cp_group,
        cp_global_ranks,
        cp_stream,
        cp_comm_type,
    ):
        if cp_comm_type != "p2p":
            raise GuardError(
                f"cp_comm_type={cp_comm_type!r} not supported in v0; only 'p2p'"
            )
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)

        # All CP paths route through the `func`-based
        # `_multi_gpu_forward_arbitrary` (1 kernel call per ring step).
        # The legacy `_multi_gpu_forward` (4 ops per ring step with zero-pad
        # + select workarounds) was a pre-`func` design from when the kernel
        # didn't yet have arbitrary-mask support. Now that `func` is
        # available, plain-causal CP is just a special case where each Q
        # row's allowed K interval is `[0, q+1)` — encoded by
        # `localize_func_for_cp_step` and run as one kernel per step.
        #
        # Requires the FBGEMM kernel to be built with
        # `HSTU_ARBITRARY_NFUNC >= 3`. Production builds without it will
        # raise `RuntimeError: This hstu attention build does not support
        # arbitrary mask` — bake the rebuilt kernel into the deployment
        # container, OR push HSTU_ARBITRARY_NFUNC=3 upstream.
        cu_seqlens_global = cu_seqlens_q * cp_size
        # Pre-compute the tuple-form of the integer tensors that drive
        # the func-cache key. ONE GPU→CPU sync per training step here
        # (vs ~16 if the cache hashes the tensor on every layer × ring
        # step call inside `_multi_gpu_forward_arbitrary`).
        # `num_contexts` / `num_targets` are None for our `--no_action
        # --no_contextual` config, so they have zero sync cost — the
        # `_hash_int_tensor_or_none` call short-circuits on None.
        cu_seqlens_global_tuple = tuple(cu_seqlens_global.tolist())
        num_contexts_tuple = (
            tuple(num_contexts.tolist()) if num_contexts is not None else None
        )
        num_targets_tuple = (
            tuple(num_targets.tolist()) if num_targets is not None else None
        )
        out = _multi_gpu_forward_arbitrary(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_global,
            max_seqlen_q_global=max_seqlen_q,
            scaling_seqlen=scaling_seqlen,
            alpha=alpha,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            window_size=tuple(window_size),
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
            cu_seqlens_global_tuple=cu_seqlens_global_tuple,
            num_contexts_tuple=num_contexts_tuple,
            num_targets_tuple=num_targets_tuple,
            cp_rank=cp_rank,
            cp_size=cp_size,
            cp_stream=cp_stream,
        )

        # Save for backward (T4.2).
        ctx.save_for_backward(q, k, v, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.scaling_seqlen = scaling_seqlen
        ctx.alpha = alpha
        ctx.cp_group = cp_group
        # Snapshot cp_global_ranks as a tuple so caller mutations between
        # forward and backward cannot route reverse-ring partials to the
        # wrong absolute ranks.
        ctx.cp_global_ranks = tuple(cp_global_ranks)
        ctx.cp_rank = cp_rank
        ctx.cp_size = cp_size
        # Reuse the same comm stream across forward/backward to avoid
        # re-creating per device.
        ctx.cp_stream = cp_stream
        # Backward also uses the `func` path
        # (`_multi_gpu_backward_arbitrary`) for symmetry with forward.
        ctx.num_contexts = num_contexts
        ctx.num_targets = num_targets
        ctx.target_group_size = target_group_size
        ctx.window_size = tuple(window_size)
        # Stash tuple form so backward can pass them straight through to
        # the func cache (one tolist sync at fwd entry feeds all ~32
        # cache lookups across fwd + bwd of one training step).
        ctx.cu_seqlens_global_tuple = cu_seqlens_global_tuple
        ctx.num_contexts_tuple = num_contexts_tuple
        ctx.num_targets_tuple = num_targets_tuple
        return out

    @staticmethod
    def backward(ctx, dout):  # type: ignore[override]
        q, k, v, cu_seqlens_q, _cu_seqlens_k = ctx.saved_tensors
        # All paths route through `_multi_gpu_backward_arbitrary` for
        # symmetry with forward (single kernel per ring step via `func`).
        cu_seqlens_global = cu_seqlens_q * ctx.cp_size
        grads = _multi_gpu_backward_arbitrary(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_global,
            dout,
            max_seqlen_q_global=ctx.max_seqlen_q,
            scaling_seqlen=ctx.scaling_seqlen,
            alpha=ctx.alpha,
            num_contexts=ctx.num_contexts,
            num_targets=ctx.num_targets,
            target_group_size=ctx.target_group_size,
            window_size=ctx.window_size,
            cp_group=ctx.cp_group,
            cp_global_ranks=ctx.cp_global_ranks,
            cp_rank=ctx.cp_rank,
            cp_size=ctx.cp_size,
            cp_stream=ctx.cp_stream,
            cu_seqlens_global_tuple=ctx.cu_seqlens_global_tuple,
            num_contexts_tuple=ctx.num_contexts_tuple,
            num_targets_tuple=ctx.num_targets_tuple,
        )
        return (
            *grads,
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # scaling_seqlen
            None,  # alpha
            None,  # num_contexts
            None,  # num_targets
            None,  # target_group_size
            None,  # window_size
            None,  # cp_group
            None,  # cp_global_ranks
            None,  # cp_stream
            None,  # cp_comm_type
        )


# ----------------------------------------------------------------------------
# Public entry point.
#
# Signature mirrors the installed `hstu_attn_varlen_func` exactly (per
# `examples/hstu/test/cp/conftest.py::CANONICAL_HSTU_PARAMS`) plus four CP
# arguments. Body order:
#   1. Determine cp_size from cp_group.
#   2. Run the 13-item hard-guard battery uniformly (cp=1 included).
#   3. cp_size == 1 ⇒ direct delegation to `hstu_attn_varlen_func`.
#   4. cp_size > 1 ⇒ dispatch via `_HSTUVarlenCPFunc.apply` (T3.3 forward
#      + T4.2 backward).
# ----------------------------------------------------------------------------
def hstu_attn_varlen_cp_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqused_q: Optional[torch.Tensor],
    seqused_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    scaling_seqlen: int,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int = 1,
    window_size: tuple[int, int] = (-1, -1),
    alpha: float = 1.0,
    rab: Optional[torch.Tensor] = None,
    has_drab: bool = False,
    kv_cache: Optional[torch.Tensor] = None,
    page_offsets: Optional[torch.Tensor] = None,
    page_ids: Optional[torch.Tensor] = None,
    last_page_lens: Optional[torch.Tensor] = None,
    func: Optional[torch.Tensor] = None,
    quant_mode: Optional[int] = -1,
    *,
    cp_group: Optional["torch.distributed.ProcessGroup"] = None,
    cp_global_ranks: Optional[list[int]] = None,
    cp_stream: Optional[torch.cuda.Stream] = None,
    cp_comm_type: str = "p2p",
) -> torch.Tensor:
    """HSTU varlen attention with optional context parallelism.

    See SPEC §1-§2 for v0 scope. When `cp_group is None` or the group has size 1,
    the call short-circuits to the production single-GPU `hstu_attn_varlen_func`.
    Otherwise the CP path runs (plan T3.3 forward / T4.2 backward).
    """
    # 1. Determine cp_size up front.
    if cp_group is None:
        cp_size = 1
    else:
        cp_size = dist.get_world_size(cp_group)

    # 2. Hard guards. Applied UNIFORMLY at both cp=1 and cp>1 paths so the
    #    contract is the same regardless of CP topology. The cost is a few
    #    Python conditionals (well under any kernel-side overhead).
    _enforce_v0_contract(
        q=q,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        seqused_q=seqused_q,
        seqused_k=seqused_k,
        num_contexts=num_contexts,
        num_targets=num_targets,
        target_group_size=target_group_size,
        window_size=window_size,
        rab=rab,
        has_drab=has_drab,
        kv_cache=kv_cache,
        page_offsets=page_offsets,
        page_ids=page_ids,
        last_page_lens=last_page_lens,
        func=func,
        quant_mode=quant_mode,  # leave None as None so guard fires
        cp_size=cp_size,
    )
    if max_seqlen_q != max_seqlen_k:
        raise GuardError(
            f"v0 supports self-attention only; got max_seqlen_q={max_seqlen_q} "
            f"!= max_seqlen_k={max_seqlen_k}"
        )

    # 3. cp_size == 1 short-circuit. After guards have rejected non-v0 modes,
    #    the call is just the bare installed kernel. cp_size==1 forwards the
    #    user-supplied heterogeneous-mask params (`num_contexts`,
    #    `num_targets`, `target_group_size`, `window_size`, `func`) to the
    #    kernel directly — at cp_size==1 there is no DualChunkSwap reordering,
    #    so the kernel handles them natively. cp_size > 1 paths still reject
    #    these (see `_enforce_v0_contract`); the
    #    `docs/cp/het_mask_design.md` track will lift the cp_size > 1
    #    rejection by translating the mask spec into a per-step `func` tensor.
    if cp_size == 1:
        return hstu_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            scaling_seqlen=scaling_seqlen,
            num_contexts=num_contexts,
            num_targets=num_targets,
            target_group_size=target_group_size,
            window_size=window_size,
            alpha=alpha,
            func=func,
            quant_mode=-1,
        )

    # 4. Multi-GPU CP path. cp_global_ranks defaults to the absolute world-rank
    #    IDs of `cp_group` (correct for both the default world group and any
    #    sub-group). NCCL P2P needs absolute ranks, so we resolve them now.
    if cp_global_ranks is None:
        cp_global_ranks = dist.get_process_group_ranks(cp_group)
    if (
        not isinstance(cp_global_ranks, (list, tuple))
        or len(cp_global_ranks) != cp_size
    ):
        raise GuardError(
            f"cp_global_ranks must be a list of length cp_size={cp_size}; "
            f"got {cp_global_ranks!r}"
        )
    if cp_comm_type != "p2p":
        raise GuardError(
            f"cp_comm_type={cp_comm_type!r} not supported in v0; only 'p2p' (SPEC §2)"
        )

    return _HSTUVarlenCPFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        scaling_seqlen,
        alpha,
        num_contexts,
        num_targets,
        target_group_size,
        window_size,
        cp_group,
        cp_global_ranks,
        cp_stream,
        cp_comm_type,
    )
