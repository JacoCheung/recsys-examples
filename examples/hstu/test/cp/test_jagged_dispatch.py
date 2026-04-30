# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
T6.3 unit tests — JaggedData-level DualChunkSwap dispatcher.

Tests `context_parallel.apply_dualchunkswap_to_jagged` and the
inverse `context_parallel.gather_jagged_from_cp_rank`. Single-GPU,
no torchrun. Auto-skipped under torchrun for the same CUDA-context
reason as `test_module_routing.py`.

Coverage:
  - cp_size=1 short-circuit returns identity.
  - Round-trip: dispatch then gather (single-process, cp_group=None)
    returns the original `values`.
  - Per-rank shard: union of all ranks' index sets equals the global
    index set (no missing tokens, no double-counting).
  - Per-rank shard: each rank's local layout is `[chunk_r, chunk_(2cp-1-r)]`
    per sample (matches the wrapper's expectation).
  - Guards reject heterogeneous-mask metadata, contextual prefix,
    interleaved action, padding > 0.
  - Guards reject divisibility violation (seqlen % 2*cp_size != 0).
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
        "test_jagged_dispatch.py is single-GPU-only; skipping under torchrun",
        allow_module_level=True,
    )

# Append `examples/hstu/` and `examples/` so both `context_parallel` and
# the existing HSTU `modules.*` package resolve. Append (not insert) so
# installed packages still win — same rationale as the cp/conftest.py
# append.
_HSTU_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = Path(__file__).resolve().parents[3]
for p in (_HSTU_ROOT, _EXAMPLES_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))


@pytest.fixture(scope="module")
def cuda_device() -> Iterator[torch.device]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    yield torch.device("cuda:0")


def _build_jd(
    seqlens: list[int], hidden_dim: int, *, device: torch.device, seed: int = 0
):
    """Build a JaggedData with deterministic values that encode their global
    row index in column 0 (so the round-trip can verify reordering)."""
    from modules.jagged_data import JaggedData

    g = torch.Generator(device=device).manual_seed(seed)
    cu = [0]
    for L in seqlens:
        cu.append(cu[-1] + L)
    total = cu[-1]
    # Encode `i` (global row index) into the values so we can detect
    # round-trip permutation errors that no other invariant catches.
    base = torch.arange(total, dtype=torch.float32, device=device).unsqueeze(1)
    payload = torch.randn(
        total, hidden_dim - 1, generator=g, dtype=torch.float32, device=device
    )
    values = torch.cat([base, payload], dim=1)
    seqlen = torch.tensor(seqlens, dtype=torch.int32, device=device)
    seqlen_offsets = torch.tensor(cu, dtype=torch.int32, device=device)
    return JaggedData(
        values=values,
        seqlen=seqlen,
        seqlen_offsets=seqlen_offsets,
        max_seqlen=max(seqlens),
    )


def test_cp_size_1_returns_identity(cuda_device: torch.device) -> None:
    from context_parallel import apply_dualchunkswap_to_jagged

    jd = _build_jd([16, 32, 16], hidden_dim=8, device=cuda_device)
    jd_loc, idx = apply_dualchunkswap_to_jagged(jd, cp_size=1, cp_rank=0)
    assert jd_loc is jd
    assert torch.equal(idx, torch.arange(jd.values.shape[0], device=cuda_device))


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_round_trip_dispatch_then_gather(
    cuda_device: torch.device, cp_size: int
) -> None:
    """Dispatch every rank's shard, gather them all back; result must equal
    the original `values`."""
    from context_parallel import (
        apply_dualchunkswap_to_jagged,
        gather_jagged_from_cp_rank,
    )

    seqlens = [2 * cp_size * mult for mult in (1, 2, 3, 5)]
    jd = _build_jd(seqlens, hidden_dim=4, device=cuda_device)

    accum = torch.zeros_like(jd.values)
    for rank in range(cp_size):
        jd_loc, l2g = apply_dualchunkswap_to_jagged(jd, cp_size=cp_size, cp_rank=rank)
        # Sanity: local shard size = sum of seqlens / cp_size.
        assert jd_loc.values.shape[0] == sum(seqlens) // cp_size
        # gather_jagged_from_cp_rank with cp_group=None scatters into a zero
        # buffer — sum across ranks reconstructs the global tensor.
        accum = accum + gather_jagged_from_cp_rank(
            jd_loc.values,
            l2g,
            cp_group=None,
            global_total_tokens=jd.values.shape[0],
        )
    assert torch.equal(accum, jd.values), f"round-trip mismatch at cp_size={cp_size}"


@pytest.mark.parametrize("cp_size", [2, 4, 8])
def test_per_rank_index_partitions_global(
    cuda_device: torch.device, cp_size: int
) -> None:
    """Union of all ranks' local_to_global indices must equal the full
    global index set, with no overlap (each token owned by exactly one rank)."""
    from context_parallel import apply_dualchunkswap_to_jagged

    seqlens = [2 * cp_size, 4 * cp_size]
    jd = _build_jd(seqlens, hidden_dim=4, device=cuda_device)
    total = jd.values.shape[0]

    seen = torch.zeros(total, dtype=torch.int32, device=cuda_device)
    for rank in range(cp_size):
        _, l2g = apply_dualchunkswap_to_jagged(jd, cp_size=cp_size, cp_rank=rank)
        seen[l2g] += 1
    assert torch.equal(
        seen, torch.ones(total, dtype=torch.int32, device=cuda_device)
    ), "per-token ownership not exactly-once"


@pytest.mark.parametrize("cp_size", [2, 4])
def test_local_layout_is_chunk_r_then_chunk_2cp1_r(
    cuda_device: torch.device, cp_size: int
) -> None:
    """v0 wrapper assumes per-sample local layout `[chunk_r, chunk_(2cp-1-r)]`.
    Verify by inspecting the recovered global indices for sample 0 on rank 0."""
    from context_parallel import apply_dualchunkswap_to_jagged

    L = 4 * cp_size  # 2 chunks of size c=2 per rank
    seqlens = [L]
    jd = _build_jd(seqlens, hidden_dim=2, device=cuda_device)
    chunks_per_seq = 2 * cp_size
    c = L // chunks_per_seq  # chunk size

    for rank in range(cp_size):
        jd_loc, l2g = apply_dualchunkswap_to_jagged(jd, cp_size=cp_size, cp_rank=rank)
        # Sample 0's local rows are l2g[0:2c]. They should be:
        # chunk_r at positions [r*c, (r+1)*c) and
        # chunk_(2cp-1-r) at positions [(2cp-1-r)*c, (2cp-r)*c).
        first_half = l2g[:c].cpu().tolist()
        second_half = l2g[c : 2 * c].cpu().tolist()
        assert first_half == list(
            range(rank * c, (rank + 1) * c)
        ), f"rank {rank} first half wrong: {first_half}"
        assert second_half == list(
            range((chunks_per_seq - 1 - rank) * c, (chunks_per_seq - rank) * c)
        ), f"rank {rank} second half wrong: {second_half}"


def test_dispatch_preserves_num_candidates(cuda_device: torch.device) -> None:
    """Het-mask metadata (num_candidates / num_candidates_offsets /
    max_num_candidates) is per-sample (not per-token); the dispatcher
    must pass it through unchanged so downstream `FusedHSTUAttention`
    can forward it to the CP wrapper's arbitrary-mask path."""
    from context_parallel import apply_dualchunkswap_to_jagged

    jd = _build_jd([8], hidden_dim=2, device=cuda_device)
    nc = torch.tensor([4], dtype=torch.int32, device=cuda_device)
    nc_off = torch.tensor([0, 4], dtype=torch.int32, device=cuda_device)
    jd.max_num_candidates = 4
    jd.num_candidates = nc
    jd.num_candidates_offsets = nc_off
    jd_loc, _l2g = apply_dualchunkswap_to_jagged(jd, cp_size=2, cp_rank=0)
    assert jd_loc.max_num_candidates == 4
    assert torch.equal(jd_loc.num_candidates, nc)
    assert torch.equal(jd_loc.num_candidates_offsets, nc_off)


def test_dispatch_preserves_contextual_seqlen(cuda_device: torch.device) -> None:
    """Het-mask contextual prefix metadata is per-sample; dispatcher must
    pass it through unchanged."""
    from context_parallel import apply_dualchunkswap_to_jagged

    jd = _build_jd([8], hidden_dim=2, device=cuda_device)
    cs = torch.tensor([4], dtype=torch.int32, device=cuda_device)
    cs_off = torch.tensor([0, 4], dtype=torch.int32, device=cuda_device)
    jd.contextual_max_seqlen = 4
    jd.contextual_seqlen = cs
    jd.contextual_seqlen_offsets = cs_off
    jd_loc, _l2g = apply_dualchunkswap_to_jagged(jd, cp_size=2, cp_rank=0)
    assert jd_loc.contextual_max_seqlen == 4
    assert torch.equal(jd_loc.contextual_seqlen, cs)
    assert torch.equal(jd_loc.contextual_seqlen_offsets, cs_off)


def test_guard_padding_length(cuda_device: torch.device) -> None:
    from context_parallel import GuardError, apply_dualchunkswap_to_jagged

    jd = _build_jd([8], hidden_dim=2, device=cuda_device)
    jd.padding_length = 4
    with pytest.raises(GuardError, match="padding_length"):
        apply_dualchunkswap_to_jagged(jd, cp_size=2, cp_rank=0)


def test_dispatch_preserves_interleaved_action_flag(
    cuda_device: torch.device,
) -> None:
    """Lifted reject: `has_interleaved_action=True` is now SUPPORTED.

    DualChunkSwap row-level chunking is mathematically equivalent under
    interleaved layout — each row is an independent Q/K token, causal
    mask uses global row position. Verified vs single-GPU baseline at
    fp64 (`tasks/verify_cp_grad_standalone.py`). The dispatcher must
    therefore accept the flag and propagate it to `jd_local`.
    """
    from context_parallel import apply_dualchunkswap_to_jagged

    jd = _build_jd([8], hidden_dim=2, device=cuda_device)
    jd.has_interleaved_action = True
    jd_local, _ = apply_dualchunkswap_to_jagged(jd, cp_size=2, cp_rank=0)
    assert jd_local.has_interleaved_action is True
    # local layout: 2 chunks of size c_b = 8/(2*2) = 2 → local seqlen = 4.
    assert jd_local.values.shape[0] == 4


def test_guard_divisibility(cuda_device: torch.device) -> None:
    """seqlen not divisible by 2*cp_size must be rejected."""
    from context_parallel import GuardError, apply_dualchunkswap_to_jagged

    jd = _build_jd([7], hidden_dim=2, device=cuda_device)  # 7 % 4 != 0
    with pytest.raises(GuardError, match="not divisible"):
        apply_dualchunkswap_to_jagged(jd, cp_size=2, cp_rank=0)


def test_guard_invalid_cp_size_or_rank(cuda_device: torch.device) -> None:
    from context_parallel import GuardError, apply_dualchunkswap_to_jagged

    jd = _build_jd([8], hidden_dim=2, device=cuda_device)
    with pytest.raises(GuardError, match="cp_size"):
        apply_dualchunkswap_to_jagged(jd, cp_size=0, cp_rank=0)
    with pytest.raises(GuardError, match="cp_rank"):
        apply_dualchunkswap_to_jagged(jd, cp_size=2, cp_rank=2)
    with pytest.raises(GuardError, match="cp_rank"):
        apply_dualchunkswap_to_jagged(jd, cp_size=2, cp_rank=-1)


def test_guard_non_jaggeddata(cuda_device: torch.device) -> None:
    from context_parallel import GuardError, apply_dualchunkswap_to_jagged

    with pytest.raises(GuardError, match="JaggedError|JaggedData"):
        apply_dualchunkswap_to_jagged({"not": "jaggeddata"}, cp_size=2, cp_rank=0)


# ----------------------------------------------------------------------------
# Codex round 2 IMPORTANT #1 — bf16 fp32-promote all-reduce branch.
#
# The gather helper has a multi-rank branch that promotes to fp32 before
# `dist.all_reduce(SUM)` and casts back to the local dtype. Single-process
# tests above never exercise this path. Mock the dist primitives so we can
# assert the call shape (op, dtype, group, in-place semantics) without
# spinning up real NCCL.
# ----------------------------------------------------------------------------
def test_gather_multi_rank_uses_fp32_all_reduce_sum(cuda_device: torch.device) -> None:
    from unittest.mock import patch

    from context_parallel import gather_jagged_from_cp_rank

    # Local shard: 8 bf16 tokens with 4-dim embedding. Choose dtype=bf16
    # explicitly (vs fp32 used elsewhere) so the up/down cast is exercised.
    local = torch.randn(
        8,
        4,
        dtype=torch.bfloat16,
        device=cuda_device,
        generator=torch.Generator(device=cuda_device).manual_seed(7),
    )
    l2g = torch.tensor([0, 2, 4, 6, 8, 10, 12, 14], device=cuda_device)
    fake_grp = object()

    captured: dict = {}

    def _fake_all_reduce(tensor, *, op, group):
        # Capture the call shape; do not modify the tensor (single-rank
        # all-reduce is a no-op so this matches the in-place SUM that real
        # NCCL would do for cp_size=1 within the group).
        captured["dtype"] = tensor.dtype
        captured["op"] = op
        captured["group"] = group
        captured["shape"] = tuple(tensor.shape)

    with patch("torch.distributed.get_world_size", return_value=2), patch(
        "torch.distributed.all_reduce", side_effect=_fake_all_reduce
    ):
        out = gather_jagged_from_cp_rank(
            local, l2g, cp_group=fake_grp, global_total_tokens=16
        )

    assert captured["dtype"] == torch.float32, "all-reduce must promote to fp32"
    assert captured["op"] is torch.distributed.ReduceOp.SUM
    assert captured["group"] is fake_grp
    assert captured["shape"] == (16, 4)
    # Output is the original local shard scattered into the global buffer
    # (since the mocked all_reduce is a no-op): non-l2g rows are 0,
    # l2g rows match `local`.
    assert out.dtype == torch.bfloat16, "down-cast must restore local dtype"
    expected = torch.zeros(16, 4, dtype=torch.bfloat16, device=cuda_device)
    expected[l2g] = local
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_gather_single_rank_skips_all_reduce(cuda_device: torch.device) -> None:
    """`cp_group` with `world_size==1` must NOT invoke all_reduce — the
    scatter is a pure local op."""
    from unittest.mock import patch

    from context_parallel import gather_jagged_from_cp_rank

    local = torch.ones(4, 2, dtype=torch.bfloat16, device=cuda_device)
    l2g = torch.tensor([0, 1, 2, 3], device=cuda_device)
    fake_grp = object()

    def _boom(*a, **kw):
        raise AssertionError("single-rank gather must not all-reduce")

    with patch("torch.distributed.get_world_size", return_value=1), patch(
        "torch.distributed.all_reduce", side_effect=_boom
    ):
        out = gather_jagged_from_cp_rank(
            local, l2g, cp_group=fake_grp, global_total_tokens=4
        )
    assert torch.equal(out, local)


# ----------------------------------------------------------------------------
# Codex round 2 IMPORTANT #2 — autograd flow through dispatch.
#
# The earlier round-trip tests use detached fp32 tensors; an autograd-breaking
# mutation in the gather path (e.g. accidental .detach()) would pass them
# silently. This test runs dispatch + gather under requires_grad and checks
# the gradient lands back at the right global rows.
# ----------------------------------------------------------------------------
def test_dispatch_then_gather_preserves_autograd(
    cuda_device: torch.device,
) -> None:
    from context_parallel import (
        apply_dualchunkswap_to_jagged,
        gather_jagged_from_cp_rank,
    )

    cp_size = 4
    seqlens = [4 * cp_size, 8 * cp_size]
    jd = _build_jd(seqlens, hidden_dim=3, device=cuda_device)
    # Attach a leaf with requires_grad — replacing the dataclass values
    # field directly so the JaggedData pipes the leaf through.
    leaf = jd.values.detach().clone().requires_grad_(True)
    jd.values = leaf

    # Dispatch on rank 0 only; sum the gathered output and call backward.
    jd_local, l2g = apply_dualchunkswap_to_jagged(jd, cp_size=cp_size, cp_rank=0)
    out_global = gather_jagged_from_cp_rank(
        jd_local.values,
        l2g,
        cp_group=None,
        global_total_tokens=jd.values.shape[0],
    )
    out_global.sum().backward()
    assert leaf.grad is not None, "no gradient propagated through dispatch"
    # Each row in `local_to_global` got summed into `out_global` exactly
    # once (by definition of the partition); other rows do not contribute.
    expected_grad = torch.zeros_like(leaf)
    expected_grad[l2g] = 1.0
    torch.testing.assert_close(leaf.grad, expected_grad, rtol=0, atol=0)
