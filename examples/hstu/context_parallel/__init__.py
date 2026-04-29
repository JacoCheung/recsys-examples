# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Context Parallelism wrapper for HSTU attention.

The kernel itself ships from the FBGEMM-style `hstu` package; this module
adds a CP (context-parallel) wrapper around `hstu.hstu_attn_varlen_func`
plus the DualChunkSwap dispatch helper. See `docs/cp/SPEC.md` for the v0
contract.

Public API (re-exported from `hstu_attn_cp`):
- `hstu_attn_varlen_cp_func` — drop-in replacement for
  `hstu_attn_varlen_func` plus four CP arguments (cp_group, cp_global_ranks,
  cp_stream, cp_comm_type).
- `get_batch_on_this_cp_rank_for_hstu` — pure-permutation DualChunkSwap
  dispatch helper for already-projected Q/K/V tensors (used by single-rank
  PoC + tests).
- `gather_global_from_cp_rank` — testing-only inverse of the Q/K/V dispatch.
- `apply_dualchunkswap_to_jagged` — JaggedData-level DualChunkSwap
  dispatcher. Use this in the trainer to shard the embedding output
  *before* the UVQK projection so each rank only runs the projection +
  attention + output projection on its own shard (Slice 6 / T6.3).
- `gather_jagged_from_cp_rank` — inverse of `apply_dualchunkswap_to_jagged`,
  optionally backed by a CP-group all-reduce when assembling the global
  output for loss computation.
- `GuardError` — typed alias of ValueError raised by guards.
"""

from .hstu_attn_cp import (
    GuardError,
    apply_dualchunkswap_to_jagged,
    gather_global_from_cp_rank,
    gather_jagged_from_cp_rank,
    get_batch_on_this_cp_rank_for_hstu,
    hstu_attn_varlen_cp_func,
)

__all__ = [
    "hstu_attn_varlen_cp_func",
    "get_batch_on_this_cp_rank_for_hstu",
    "gather_global_from_cp_rank",
    "apply_dualchunkswap_to_jagged",
    "gather_jagged_from_cp_rank",
    "GuardError",
]
