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
