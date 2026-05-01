# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Parity guard for `_call_hstu_bwd_kernel`.

`_multi_gpu_backward_arbitrary` used to extract per-tile grads via
    q.detach().clone().requires_grad_(True)
    out = hstu_attn_varlen_func(q=..., k=..., v=..., func=func, ...)
    dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)

That path was replaced by a direct call to
`torch.ops.fbgemm.hstu_varlen_bwd_*` via `_call_hstu_bwd_kernel`. Both
paths invoke the same underlying kernel bwd op with the same args, so
the produced gradients must be bit-identical (`torch.equal`, not just
`assert_close`). This test pins that invariant.

Single-process, single-GPU. Skipped under torchrun for the same reason
as the other single-process CP tests.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import torch

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    pytest.skip(
        "test_direct_bwd_parity is single-process; skipping under torchrun",
        allow_module_level=True,
    )

_HSTU_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = Path(__file__).resolve().parents[3]
for p in (_HSTU_ROOT, _EXAMPLES_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))


@pytest.fixture(scope="module")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return torch.device("cuda:0")


def _build_inputs(
    *,
    seqlens: list[int],
    num_heads: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
):
    g = torch.Generator(device=device).manual_seed(seed)
    total = sum(seqlens)
    q = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    cu = (
        torch.tensor([0] + list(seqlens), dtype=torch.int32, device=device)
        .cumsum(0)
        .int()
    )
    dout = torch.randn(
        total, num_heads, head_dim, generator=g, dtype=dtype, device=device
    )
    return q, k, v, cu, dout


def _autograd_grad_path(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu: torch.Tensor,
    max_seqlen: int,
    alpha: float,
    scaling_seqlen: int,
    window_size: tuple[int, int],
    func: torch.Tensor | None,
    dout: torch.Tensor,
):
    """Mirror the OLD enable_grad → autograd.grad path for a single tile."""
    from hstu import hstu_attn_varlen_func

    with torch.enable_grad():
        q_in = q.detach().clone().requires_grad_(True)
        k_in = k.detach().clone().requires_grad_(True)
        v_in = v.detach().clone().requires_grad_(True)
        out = hstu_attn_varlen_func(
            q=q_in,
            k=k_in,
            v=v_in,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            scaling_seqlen=scaling_seqlen,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=window_size,
            alpha=alpha,
            func=func,
            quant_mode=-1,
        )
        dq, dk, dv = torch.autograd.grad(out, (q_in, k_in, v_in), dout)
    return dq.detach(), dk.detach(), dv.detach()


@pytest.mark.parametrize(
    "seqlens, num_heads, head_dim",
    [
        ([16, 32], 2, 32),
        ([64, 64, 64, 64], 4, 64),
        ([128, 256], 4, 128),
    ],
)
def test_direct_bwd_matches_autograd_path_no_func(
    cuda_device: torch.device,
    seqlens: list[int],
    num_heads: int,
    head_dim: int,
):
    """No-`func` path (plain causal): direct bwd op = autograd.grad."""
    from context_parallel.hstu_attn_cp import _call_hstu_bwd_kernel

    q, k, v, cu, dout = _build_inputs(
        seqlens=seqlens,
        num_heads=num_heads,
        head_dim=head_dim,
        device=cuda_device,
    )
    max_seqlen = max(seqlens)
    alpha = 1.0 / (head_dim**0.5)

    dq_ref, dk_ref, dv_ref = _autograd_grad_path(
        q=q,
        k=k,
        v=v,
        cu=cu,
        max_seqlen=max_seqlen,
        alpha=alpha,
        scaling_seqlen=max_seqlen,
        window_size=(-1, 0),
        func=None,
        dout=dout,
    )
    dq, dk, dv = _call_hstu_bwd_kernel(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        dout=dout,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        scaling_seqlen=max_seqlen,
        alpha=alpha,
        window_size=(-1, 0),
        func=None,
    )

    # Both paths invoke the same kernel bwd op with the same inputs —
    # results must be bit-identical, not merely close.
    assert torch.equal(dq, dq_ref), "dq diverges from autograd.grad path"
    assert torch.equal(dk, dk_ref), "dk diverges from autograd.grad path"
    assert torch.equal(dv, dv_ref), "dv diverges from autograd.grad path"


def test_direct_bwd_matches_autograd_path_with_func(cuda_device: torch.device):
    """With a `func` mask tensor (CP path): direct bwd op = autograd.grad.

    This is the actual code path `_multi_gpu_backward_arbitrary` uses.
    The arbitrary-mask kernel build (HSTU_ARBITRARY_NFUNC>=3) is
    required; the test skips if the build doesn't have it.
    """
    from context_parallel._mask_func import build_global_mask_func
    from context_parallel.hstu_attn_cp import _call_hstu_bwd_kernel

    seqlens = [32, 32]  # cp_size=2 chunkable: each sample has chunks of 8
    num_heads = 4
    head_dim = 64
    q, k, v, cu, dout = _build_inputs(
        seqlens=seqlens,
        num_heads=num_heads,
        head_dim=head_dim,
        device=cuda_device,
    )
    max_seqlen = max(seqlens)
    alpha = 1.0 / (head_dim**0.5)

    # Build a `func` tensor that encodes plain causal globally — should
    # produce the same grads as window_size=(-1, 0) without `func`.
    func = build_global_mask_func(
        cu_seqlens_q=cu,
        max_seqlen_q=max_seqlen,
        num_contexts=None,
        num_targets=None,
        target_group_size=1,
        window_size=(-1, 0),
        NFUNC=3,
        device=cuda_device,
    )

    try:
        dq_ref, dk_ref, dv_ref = _autograd_grad_path(
            q=q,
            k=k,
            v=v,
            cu=cu,
            max_seqlen=max_seqlen,
            alpha=alpha,
            scaling_seqlen=max_seqlen,
            window_size=(-1, -1),
            func=func,
            dout=dout,
        )
    except RuntimeError as e:
        if "arbitrary mask" in str(e).lower():
            pytest.skip(
                "kernel build lacks HSTU_ARBITRARY_NFUNC support — "
                "rebuild with HSTU_ARBITRARY_NFUNC=3 to run this test"
            )
        raise

    dq, dk, dv = _call_hstu_bwd_kernel(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        dout=dout,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        scaling_seqlen=max_seqlen,
        alpha=alpha,
        window_size=(-1, -1),
        func=func,
    )

    assert torch.equal(dq, dq_ref), "dq diverges from autograd.grad path"
    assert torch.equal(dk, dk_ref), "dk diverges from autograd.grad path"
    assert torch.equal(dv, dv_ref), "dv diverges from autograd.grad path"
