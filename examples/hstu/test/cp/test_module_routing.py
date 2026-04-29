# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
T6.2 unit test — `FusedHSTUAttention` correctly routes to the CP wrapper
when configured with a multi-rank CP group, and to the legacy single-GPU
kernel when no CP group is provided.

Single-GPU; no torchrun. Uses a fake process group + a monkey-patched
`hstu_attn_varlen_cp_func` to assert the dispatch path without spinning
up real NCCL.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch

import pytest
import torch

# This file is single-GPU-only and uses unittest.mock to patch
# `torch.distributed.get_world_size`. Running it under torchrun (where
# every rank is a separate process all reading cuda:0 in our fixture)
# corrupts CUDA contexts and breaks unrelated tests collected in the
# same shard. Skip the whole module when WORLD_SIZE > 1.
if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    pytest.skip(
        "test_module_routing.py is single-GPU-only; skipping under torchrun",
        allow_module_level=True,
    )

# `examples/hstu` is appended by conftest.py for `context_parallel`. This
# test additionally needs `examples/` so the HSTU modules' own
# `from commons.utils ...` imports resolve (the existing convention in
# the repo). Append (not insert) so installed packages still win.
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


class _FakeCPGroup:
    """Mimics the bits of `torch.distributed.ProcessGroup` we read from."""

    def __init__(self, size: int):
        self._size = size

    def __repr__(self) -> str:
        return f"<FakeCPGroup size={self._size}>"


@contextmanager
def _fake_cp_group(size: int) -> Iterator[_FakeCPGroup]:
    grp = _FakeCPGroup(size)
    with patch("torch.distributed.get_world_size", return_value=size):
        yield grp


def _build_jagged_inputs(
    *, batch: int, seqlen: int, num_heads: int, head_dim: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(0)
    total = batch * seqlen
    d = num_heads * head_dim
    tq = torch.randn(total, d, generator=g, dtype=torch.bfloat16, device=device)
    tk = torch.randn(total, d, generator=g, dtype=torch.bfloat16, device=device)
    tv = torch.randn(total, d, generator=g, dtype=torch.bfloat16, device=device)
    cu = torch.arange(0, total + 1, seqlen, dtype=torch.int32, device=device)
    return tq, tk, tv, cu


def test_no_cp_group_routes_to_legacy_kernel(cuda_device: torch.device) -> None:
    """No `cp_group` ⇒ pre-CP path (`hstu_attn_varlen_func`)."""
    # Import lazily so test discovery on a CPU box doesn't fail at collection.
    from modules.hstu_attention import FusedHSTUAttention

    attn = FusedHSTUAttention(
        num_heads=2, attention_dim=32, linear_dim=32, is_causal=True
    )
    tq, tk, tv, cu = _build_jagged_inputs(
        batch=4, seqlen=64, num_heads=2, head_dim=32, device=cuda_device
    )
    expected = attn(tq, tk, tv, cu, max_seqlen=64, scaling_seqlen=64)

    # Spy: ensure the CP wrapper is NOT called.
    with patch("context_parallel.hstu_attn_varlen_cp_func") as cp_spy, patch.object(
        torch.distributed, "get_world_size", return_value=1
    ):
        got = attn(tq, tk, tv, cu, max_seqlen=64, scaling_seqlen=64)
    assert torch.equal(got, expected)
    cp_spy.assert_not_called()


def test_cp_group_size_1_routes_to_legacy(cuda_device: torch.device) -> None:
    """`cp_group` whose `world_size==1` ⇒ legacy path (no CP overhead)."""
    from modules.hstu_attention import FusedHSTUAttention

    with _fake_cp_group(1) as grp:
        attn = FusedHSTUAttention(
            num_heads=2,
            attention_dim=32,
            linear_dim=32,
            is_causal=True,
            cp_group=grp,
            cp_global_ranks=[0],
        )
        tq, tk, tv, cu = _build_jagged_inputs(
            batch=4, seqlen=64, num_heads=2, head_dim=32, device=cuda_device
        )
        with patch("context_parallel.hstu_attn_varlen_cp_func") as cp_spy:
            attn(tq, tk, tv, cu, max_seqlen=64, scaling_seqlen=64)
    cp_spy.assert_not_called()


def test_cp_group_size_4_routes_to_cp_wrapper(cuda_device: torch.device) -> None:
    """Multi-rank `cp_group` ⇒ dispatches to `hstu_attn_varlen_cp_func`."""
    import context_parallel
    from modules.hstu_attention import FusedHSTUAttention

    with _fake_cp_group(4) as grp:
        attn = FusedHSTUAttention(
            num_heads=2,
            attention_dim=32,
            linear_dim=32,
            is_causal=True,
            cp_group=grp,
            cp_global_ranks=[0, 1, 2, 3],
        )
        tq, tk, tv, cu = _build_jagged_inputs(
            batch=4, seqlen=64, num_heads=2, head_dim=32, device=cuda_device
        )
        sentinel = torch.zeros(
            tq.numel() // (2 * 32) * 1,
            2 * 32,
            dtype=tq.dtype,
            device=tq.device,
        )
        # The CP wrapper would itself call `hstu_attn_varlen_func` per tile;
        # short-circuit with a sentinel so we don't actually try to run NCCL.
        with patch.object(
            context_parallel,
            "hstu_attn_varlen_cp_func",
            return_value=sentinel,
        ) as cp_spy:
            out = attn(tq, tk, tv, cu, max_seqlen=64, scaling_seqlen=64)
    cp_spy.assert_called_once()
    # And we received the patched sentinel reshaped to (T, num_heads*linear_dim).
    assert out.shape[-1] == attn.num_heads * attn.linear_dim


def test_cp_rejects_heterogeneous_mask(cuda_device: torch.device) -> None:
    """v0 contract: cp>1 + heterogeneous mask params ⇒ ValueError."""
    from modules.hstu_attention import FusedHSTUAttention

    with _fake_cp_group(2) as grp:
        attn = FusedHSTUAttention(
            num_heads=2,
            attention_dim=32,
            linear_dim=32,
            is_causal=True,
            cp_group=grp,
            cp_global_ranks=[0, 1],
        )
        tq, tk, tv, cu = _build_jagged_inputs(
            batch=4, seqlen=64, num_heads=2, head_dim=32, device=cuda_device
        )
        # num_contextuals = int -> normalised to a tensor inside forward;
        # tracker after that point checks `is None`. So we test the
        # already-tensor case (production path) directly.
        bad_num_contextuals = torch.tensor(
            [4] * 4, dtype=torch.int32, device=cuda_device
        )
        with pytest.raises(ValueError, match="heterogeneous mask"):
            attn(
                tq,
                tk,
                tv,
                cu,
                max_seqlen=64,
                scaling_seqlen=64,
                num_contextuals=bad_num_contextuals,
            )


def test_cp_requires_global_ranks(cuda_device: torch.device) -> None:
    """Constructing FusedHSTUAttention with cp_group but no cp_global_ranks
    should fail at __init__, not at forward."""
    from modules.hstu_attention import FusedHSTUAttention

    with _fake_cp_group(2) as grp:
        with pytest.raises(ValueError, match="cp_global_ranks is required"):
            FusedHSTUAttention(
                num_heads=2,
                attention_dim=32,
                linear_dim=32,
                is_causal=True,
                cp_group=grp,
                cp_global_ranks=None,
            )
