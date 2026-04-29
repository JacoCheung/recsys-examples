# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
T6.x het-mask multi-GPU correctness test (torchrun pytest).

Step 4a (forward only): the CP wrapper builds a per-step `func` tensor
via `localize_func_for_cp_step` to express heterogeneous masks
(num_contexts, num_targets, target_group_size, sliding window) under
DualChunkSwap. This test compares the CP wrapper output to the
single-GPU baseline (`hstu_attn_varlen_func` with the same 4-tuple
spec) on a small matrix of het-mask configurations.

Run:
    bash examples/hstu/cp/run_cp_tests.sh

(test gated to cp=2 / cp=4 cells via -k filter; no backward yet.)
"""

from __future__ import annotations

import os
from typing import Iterator, Optional

import pytest
import torch
import torch.distributed as dist
from conftest import random_varlen_batch
from context_parallel import (
    get_batch_on_this_cp_rank_for_hstu,
    hstu_attn_varlen_cp_func,
)
from hstu import hstu_attn_varlen_func


def _alpha_for(head_dim: int) -> float:
    return 1.0 / (head_dim**0.5)


def _kernel_has_arbitrary() -> bool:
    """Probe whether the installed FBGEMM hstu kernel supports arbitrary mask.

    Het-mask under CP requires `HSTU_ARBITRARY_NFUNC > 0`; some pre-built
    binaries ship with arbitrary disabled.
    """
    if not torch.cuda.is_available():
        return False
    try:
        device = torch.device("cuda:0")
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
def cp_world() -> Iterator[dict]:
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    if not _HAS_ARBITRARY:
        pytest.skip(
            "installed hstu kernel was built without arbitrary-mask "
            "support (HSTU_ARBITRARY_NFUNC=0); rebuild per "
            "docs/cp/het_mask_design.md §4.A"
        )
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("multi-GPU test requires WORLD_SIZE >= 2 (run under torchrun)")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    cp_group = dist.new_group(list(range(world_size)), backend="nccl")
    cp_global_ranks = list(range(world_size))
    yield dict(
        cp_group=cp_group,
        cp_global_ranks=cp_global_ranks,
        rank=rank,
        world_size=world_size,
        device=torch.device(f"cuda:{rank}"),
    )
    dist.barrier()


# Each entry: id, seqlens, num_heads, head_dim, num_contexts (per-sample),
# num_targets (per-sample), target_group_size, window_size.
# Seqlens divisible by 2 * cp_size (cp ∈ {2, 4} per the matrix below).
HET_MASK_MATRIX_CP2 = [
    dict(
        id="cp2_targets_g1",
        seqlens=[64, 64],
        num_heads=2,
        head_dim=32,
        nc=None,
        nt=[8, 8],
        tgs=1,
        ws=(-1, 0),
    ),
    dict(
        id="cp2_targets_g2",
        seqlens=[64, 64],
        num_heads=2,
        head_dim=32,
        nc=None,
        nt=[8, 8],
        tgs=2,
        ws=(-1, 0),
    ),
    dict(
        id="cp2_contextual",
        seqlens=[64, 64],
        num_heads=2,
        head_dim=32,
        nc=[4, 4],
        nt=None,
        tgs=1,
        ws=(-1, 0),
    ),
    dict(
        id="cp2_full_combo",
        seqlens=[64, 64],
        num_heads=2,
        head_dim=32,
        nc=[4, 4],
        nt=[8, 8],
        tgs=2,
        ws=(-1, 0),
    ),
    # Sliding-causal-only (no het-mask params) routes through the same
    # arbitrary-mask path because `_is_het_mask` returns True for any
    # non-causal window_size. Verifies the v0.5 sliding track no longer
    # needs separate plumbing — it shares the het-mask `func` machinery.
    dict(
        id="cp2_sliding_w8",
        seqlens=[64, 64],
        num_heads=2,
        head_dim=32,
        nc=None,
        nt=None,
        tgs=1,
        ws=(8, 0),
    ),
    dict(
        id="cp2_sliding_w16",
        seqlens=[64, 64],
        num_heads=2,
        head_dim=32,
        nc=None,
        nt=None,
        tgs=1,
        ws=(16, 0),
    ),
]
HET_MASK_MATRIX_CP4 = [
    dict(
        id="cp4_targets_g1",
        seqlens=[128, 128],
        num_heads=2,
        head_dim=32,
        nc=None,
        nt=[16, 16],
        tgs=1,
        ws=(-1, 0),
    ),
    dict(
        id="cp4_full_combo",
        seqlens=[128, 128],
        num_heads=2,
        head_dim=32,
        nc=[8, 8],
        nt=[16, 16],
        tgs=2,
        ws=(-1, 0),
    ),
    dict(
        id="cp4_sliding_w16",
        seqlens=[128, 128],
        num_heads=2,
        head_dim=32,
        nc=None,
        nt=None,
        tgs=1,
        ws=(16, 0),
    ),
]


def _to_int32_or_none(
    xs: Optional[list[int]], device: torch.device
) -> Optional[torch.Tensor]:
    if xs is None:
        return None
    return torch.tensor(xs, dtype=torch.int32, device=device)


def _baseline_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    *,
    max_seqlen: int,
    alpha: float,
    num_contexts: Optional[torch.Tensor],
    num_targets: Optional[torch.Tensor],
    target_group_size: int,
    window_size: tuple[int, int],
) -> torch.Tensor:
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
        alpha=alpha,
    )


def _run_one_correctness(entry: dict, cp_world: dict) -> None:
    cp_group = cp_world["cp_group"]
    cp_size = cp_world["world_size"]
    cp_rank = cp_world["rank"]
    device = cp_world["device"]

    seqlens = entry["seqlens"]
    head_dim = entry["head_dim"]
    num_heads = entry["num_heads"]
    alpha = _alpha_for(head_dim)
    max_seqlen = max(seqlens)

    # Step 1: deterministic global batch on every rank.
    q_g, k_g, v_g, cu_g = random_varlen_batch(
        seqlens, num_heads=num_heads, head_dim=head_dim, device=device, seed=0
    )
    nc = _to_int32_or_none(entry["nc"], device)
    nt = _to_int32_or_none(entry["nt"], device)
    tgs = entry["tgs"]
    ws = entry["ws"]

    # Step 2: single-GPU baseline (every rank computes; deterministic seed).
    out_base = _baseline_fwd(
        q_g,
        k_g,
        v_g,
        cu_g,
        max_seqlen=max_seqlen,
        alpha=alpha,
        num_contexts=nc,
        num_targets=nt,
        target_group_size=tgs,
        window_size=ws,
    )

    # Step 3: dispatch this rank's DualChunkSwap shard.
    q_loc, k_loc, v_loc, cu_loc, l2g, _ = get_batch_on_this_cp_rank_for_hstu(
        q_g, k_g, v_g, cu_g, cp_size=cp_size, cp_rank=cp_rank
    )

    # Step 4: CP wrapper.
    out_loc = hstu_attn_varlen_cp_func(
        q=q_loc,
        k=k_loc,
        v=v_loc,
        cu_seqlens_q=cu_loc,
        cu_seqlens_k=cu_loc,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        scaling_seqlen=max_seqlen,
        num_contexts=nc,
        num_targets=nt,
        target_group_size=tgs,
        window_size=ws,
        alpha=alpha,
        cp_group=cp_group,
        cp_global_ranks=cp_world["cp_global_ranks"],
    )

    # Step 5: scatter back to global, all-reduce SUM.
    contrib = torch.zeros_like(q_g, dtype=torch.float32)
    contrib[l2g] = out_loc.float()
    dist.all_reduce(contrib, op=dist.ReduceOp.SUM, group=cp_group)
    out_global = contrib.to(q_g.dtype)

    # Step 6: tolerance compare.
    diff = (out_global.float() - out_base.float()).abs()
    max_abs = diff.max().item()
    base_max = out_base.float().abs().max().item()
    assert torch.isfinite(out_global).all().item(), f"non-finite max={max_abs}"
    torch.testing.assert_close(
        out_global.float(),
        out_base.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=lambda m: (
            f"{entry['id']}: cp_size={cp_size} max_abs={max_abs:.3e} "
            f"base_max={base_max:.3e}\n{m}"
        ),
    )


@pytest.mark.parametrize("entry", HET_MASK_MATRIX_CP2, ids=lambda e: e["id"])
def test_cp2(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 2:
        pytest.skip(f"requires WORLD_SIZE=2; got {cp_world['world_size']}")
    _run_one_correctness(entry, cp_world)


@pytest.mark.parametrize("entry", HET_MASK_MATRIX_CP4, ids=lambda e: e["id"])
def test_cp4(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 4:
        pytest.skip(f"requires WORLD_SIZE=4; got {cp_world['world_size']}")
    _run_one_correctness(entry, cp_world)


# ============================================================================
# Backward (Step 4b): ctx-saved het-mask spec + reverse-ring dKV exchange.
# ============================================================================
def _run_one_correctness_bwd(entry: dict, cp_world: dict) -> None:
    cp_group = cp_world["cp_group"]
    cp_size = cp_world["world_size"]
    cp_rank = cp_world["rank"]
    device = cp_world["device"]

    seqlens = entry["seqlens"]
    head_dim = entry["head_dim"]
    num_heads = entry["num_heads"]
    alpha = _alpha_for(head_dim)
    max_seqlen = max(seqlens)

    q_g, k_g, v_g, cu_g = random_varlen_batch(
        seqlens, num_heads=num_heads, head_dim=head_dim, device=device, seed=0
    )
    nc = _to_int32_or_none(entry["nc"], device)
    nt = _to_int32_or_none(entry["nt"], device)
    tgs = entry["tgs"]
    ws = entry["ws"]

    # Deterministic dout on every rank (same seed).
    g = torch.Generator(device=device).manual_seed(101)
    dout_g = torch.randn(q_g.shape, generator=g, dtype=q_g.dtype, device=device)

    # Single-GPU baseline fwd+bwd.
    q_b = q_g.detach().clone().requires_grad_(True)
    k_b = k_g.detach().clone().requires_grad_(True)
    v_b = v_g.detach().clone().requires_grad_(True)
    out_base = _baseline_fwd(
        q_b,
        k_b,
        v_b,
        cu_g,
        max_seqlen=max_seqlen,
        alpha=alpha,
        num_contexts=nc,
        num_targets=nt,
        target_group_size=tgs,
        window_size=ws,
    )
    dq_base, dk_base, dv_base = torch.autograd.grad(out_base, (q_b, k_b, v_b), dout_g)

    # CP wrapper fwd+bwd on this rank's shard.
    q_loc, k_loc, v_loc, cu_loc, l2g, _ = get_batch_on_this_cp_rank_for_hstu(
        q_g, k_g, v_g, cu_g, cp_size=cp_size, cp_rank=cp_rank
    )
    q_loc = q_loc.detach().clone().requires_grad_(True)
    k_loc = k_loc.detach().clone().requires_grad_(True)
    v_loc = v_loc.detach().clone().requires_grad_(True)
    dout_loc = dout_g[l2g]

    out_loc = hstu_attn_varlen_cp_func(
        q=q_loc,
        k=k_loc,
        v=v_loc,
        cu_seqlens_q=cu_loc,
        cu_seqlens_k=cu_loc,
        seqused_q=None,
        seqused_k=None,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        scaling_seqlen=max_seqlen,
        num_contexts=nc,
        num_targets=nt,
        target_group_size=tgs,
        window_size=ws,
        alpha=alpha,
        cp_group=cp_group,
        cp_global_ranks=cp_world["cp_global_ranks"],
    )
    out_loc.backward(dout_loc)
    dq_loc = q_loc.grad.detach()
    dk_loc = k_loc.grad.detach()
    dv_loc = v_loc.grad.detach()

    def _scatter_allreduce(local: torch.Tensor) -> torch.Tensor:
        contrib = torch.zeros_like(q_g, dtype=torch.float32)
        contrib[l2g] = local.float()
        dist.all_reduce(contrib, op=dist.ReduceOp.SUM, group=cp_group)
        return contrib.to(q_g.dtype)

    dq_g = _scatter_allreduce(dq_loc)
    dk_g = _scatter_allreduce(dk_loc)
    dv_g = _scatter_allreduce(dv_loc)
    out_global = _scatter_allreduce(out_loc.detach())

    for name, t in [
        ("out", out_global),
        ("dq", dq_g),
        ("dk", dk_g),
        ("dv", dv_g),
    ]:
        assert torch.isfinite(t).all().item(), f"{name}: non-finite"
    torch.testing.assert_close(
        out_global.float(),
        out_base.float(),
        rtol=2e-2,
        atol=2e-2,
        msg=f"{entry['id']} fwd",
    )
    torch.testing.assert_close(
        dq_g.float(),
        dq_base.float(),
        rtol=5e-2,
        atol=5e-2,
        msg=f"{entry['id']} dq",
    )
    torch.testing.assert_close(
        dk_g.float(),
        dk_base.float(),
        rtol=5e-2,
        atol=5e-2,
        msg=f"{entry['id']} dk",
    )
    torch.testing.assert_close(
        dv_g.float(),
        dv_base.float(),
        rtol=5e-2,
        atol=5e-2,
        msg=f"{entry['id']} dv",
    )


@pytest.mark.parametrize("entry", HET_MASK_MATRIX_CP2, ids=lambda e: e["id"])
def test_cp2_bwd(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 2:
        pytest.skip(f"requires WORLD_SIZE=2; got {cp_world['world_size']}")
    _run_one_correctness_bwd(entry, cp_world)


@pytest.mark.parametrize("entry", HET_MASK_MATRIX_CP4, ids=lambda e: e["id"])
def test_cp4_bwd(entry: dict, cp_world: dict) -> None:
    if cp_world["world_size"] != 4:
        pytest.skip(f"requires WORLD_SIZE=4; got {cp_world['world_size']}")
    _run_one_correctness_bwd(entry, cp_world)
