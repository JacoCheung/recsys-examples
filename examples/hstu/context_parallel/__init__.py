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
  dispatch helper (used by the dataloader before the CP forward).
- `gather_global_from_cp_rank` — testing-only inverse of the dispatch.
- `GuardError` — typed alias of ValueError raised by guards.
"""

from .hstu_attn_cp import (
    GuardError,
    gather_global_from_cp_rank,
    get_batch_on_this_cp_rank_for_hstu,
    hstu_attn_varlen_cp_func,
)

__all__ = [
    "hstu_attn_varlen_cp_func",
    "get_batch_on_this_cp_rank_for_hstu",
    "gather_global_from_cp_rank",
    "GuardError",
]
