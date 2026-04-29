# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
T6.4 unit test — `HSTUBlock` wires DualChunkSwap dispatch + gather around
the layer stack when `context_parallel_size > 1`.

Single-GPU; no torchrun.  Uses a fake `cp_group` and patches
`parallel_state.get_context_parallel_*` so we can verify the
`apply_dualchunkswap_to_jagged` / `gather_jagged_from_cp_rank` calls
without spinning up a real CP world.

Auto-skipped under torchrun (same rationale as `test_module_routing.py`:
real `dist` state collides with the patched one).
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

if int(os.environ.get("WORLD_SIZE", "1")) > 1:
    pytest.skip(
        "test_hstu_block_cp.py is single-GPU-only; skipping under torchrun",
        allow_module_level=True,
    )

# `examples/hstu/` for `context_parallel`; `examples/` for `commons`/`modules`.
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
    def __init__(self, size: int):
        self._size = size

    def __repr__(self) -> str:
        return f"<FakeCPGroup size={self._size}>"

    # `dist.get_world_size(group)` calls `group.size()`; the gather and
    # forward both query world_size on this group, so expose `.size()`
    # to avoid having to patch `dist.get_world_size` everywhere.
    def size(self) -> int:
        return self._size


@contextmanager
def _fake_cp(size: int, ranks: tuple) -> Iterator[_FakeCPGroup]:
    grp = _FakeCPGroup(size)
    with patch(
        "megatron.core.parallel_state.get_context_parallel_group",
        return_value=grp,
    ), patch(
        "megatron.core.parallel_state.get_context_parallel_global_ranks",
        return_value=list(ranks),
    ), patch(
        "torch.distributed.get_world_size", return_value=size
    ), patch(
        "torch.distributed.get_rank", return_value=0
    ):
        yield grp


def _make_jd(seqlens: list[int], hidden_dim: int, device: torch.device):
    """Build a deterministic JaggedData with row-index encoded in column 0."""
    from modules.jagged_data import JaggedData

    cu = [0]
    for L in seqlens:
        cu.append(cu[-1] + L)
    total = cu[-1]
    base = torch.arange(total, dtype=torch.float32, device=device).unsqueeze(1)
    pad = torch.zeros(total, hidden_dim - 1, dtype=torch.float32, device=device)
    values = torch.cat([base, pad], dim=1)
    return JaggedData(
        values=values,
        seqlen=torch.tensor(seqlens, dtype=torch.int32, device=device),
        seqlen_offsets=torch.tensor(cu, dtype=torch.int32, device=device),
        max_seqlen=max(seqlens),
    )


def test_block_cp_dispatch_and_gather_around_layer_stack(
    cuda_device: torch.device,
) -> None:
    """When cp_size>1, HSTUBlock.forward must:
      1. Apply DualChunkSwap dispatch on the post-preprocessor JaggedData
         (the layer stack sees the local shard).
      2. Run all attention layers on the local shard.
      3. Gather the post-attention values back to global shape before the
         postprocessor (so postprocessor metadata splits work).
      4. Return a global JaggedData (matching the cp_size==1 contract).

    Stub the preprocessor (returns a known JaggedData) and the layer
    stack (identity layers via spy) and the postprocessor (returns
    whatever it gets) so we can verify the wiring without real layer
    weights.
    """
    from context_parallel import apply_dualchunkswap_to_jagged
    from modules.hstu_block import HSTUBlock

    cp_size = 2
    seqlens = [16, 16]  # divisible by 2*cp_size=4
    hidden_dim = 4
    global_jd = _make_jd(seqlens, hidden_dim=hidden_dim, device=cuda_device)
    total_global_tokens = global_jd.values.shape[0]

    # Build a stub HSTUBlock.  We don't actually need a real config — we
    # just need a class that exposes _preprocessor / _attention_layers /
    # _postprocessor and the cp_size capture logic.  Test the forward
    # method as if it were called on a real instance, with stubbed
    # attributes.
    block = HSTUBlock.__new__(HSTUBlock)
    torch.nn.Module.__init__(block)
    block.config = type("C", (), {"num_layers": 1})()
    block._cp_size = cp_size
    block._cp_global_ranks = (0, 1)
    fake_grp = _FakeCPGroup(cp_size)
    block._cp_group = fake_grp

    # Identity preprocessor: returns the global JaggedData regardless of
    # input.
    block._preprocessor = lambda *args, **kwargs: global_jd
    # Spy layer that records its input.values for later inspection.
    captured: dict = {}

    def _spy_layer(jd):
        captured["layer_in_values"] = jd.values
        captured["layer_in_seqlen"] = jd.seqlen
        captured["layer_in_max_seqlen"] = jd.max_seqlen
        return jd  # identity (no actual attention)

    block._attention_layers = [_spy_layer]
    block._postprocessor = lambda jd: jd  # passthrough

    # Patch the dispatch helper to also record what it was called with.
    real_dispatch = apply_dualchunkswap_to_jagged

    def _spy_dispatch(jd, *, cp_size, cp_rank):
        captured["dispatch_cp_size"] = cp_size
        captured["dispatch_cp_rank"] = cp_rank
        return real_dispatch(jd, cp_size=cp_size, cp_rank=cp_rank)

    with patch("torch.distributed.get_rank", return_value=0), patch(
        "context_parallel.apply_dualchunkswap_to_jagged",
        side_effect=_spy_dispatch,
    ), patch(
        # gather_jagged_from_cp_rank calls dist.all_reduce; with a fake
        # cp_group there's no real backend.  Stub it as in-place identity
        # (single-rank simulation: every position is owned by exactly
        # one rank, so SUM over [self_contrib, 0, 0, …] = self_contrib).
        "torch.distributed.all_reduce",
        side_effect=lambda tensor, op=None, group=None: None,
    ):
        out_jd, meta = block.forward(embeddings={}, batch=None)

    # 1. dispatch was called with cp_size=2, cp_rank=0.
    assert captured["dispatch_cp_size"] == cp_size
    assert captured["dispatch_cp_rank"] == 0

    # 2. layer saw a LOCAL shard (half the tokens).
    assert captured["layer_in_values"].shape[0] == total_global_tokens // cp_size

    # 3. layer's input seqlen reflects the local shard (each sample has
    # local_L = global_L / cp_size).
    expected_local_seqlens = [L // cp_size for L in seqlens]
    assert captured["layer_in_seqlen"].cpu().tolist() == expected_local_seqlens

    # 4. The output JD's `values` is back to GLOBAL shape (after gather).
    assert out_jd.values.shape[0] == total_global_tokens

    # 5. Returned metadata in the second-tuple is GLOBAL seqlen (not
    # local), captured pre-dispatch.
    seqlen_meta, _, _ = meta
    assert seqlen_meta.cpu().tolist() == seqlens

    # 6. Critical: the layer must see GLOBAL max_seqlen, not local. The
    # CP wrapper expects max_seqlen_q to be the global value (it
    # internally divides by cp_size to get the kernel-local max);
    # passing local max here would cause the wrapper to divide twice
    # and emit kernel-tile sizes off-by-cp_size. Codex round-1 BLOCKER
    # regression-guard.
    assert captured["layer_in_max_seqlen"] == max(seqlens)


def test_block_cp_size_1_skips_dispatch(cuda_device: torch.device) -> None:
    """cp_size=1 must NOT engage `apply_dualchunkswap_to_jagged` —
    the layer stack runs on the global JaggedData unchanged."""
    from modules.hstu_block import HSTUBlock

    seqlens = [12, 12]
    hidden_dim = 4
    global_jd = _make_jd(seqlens, hidden_dim=hidden_dim, device=cuda_device)
    total_global_tokens = global_jd.values.shape[0]

    block = HSTUBlock.__new__(HSTUBlock)
    torch.nn.Module.__init__(block)
    block.config = type("C", (), {"num_layers": 1})()
    block._cp_size = 1
    block._cp_global_ranks = None
    block._cp_group = None
    block._preprocessor = lambda *a, **k: global_jd
    captured: dict = {}

    def _spy_layer(jd):
        captured["values_shape"] = jd.values.shape
        return jd

    block._attention_layers = [_spy_layer]
    block._postprocessor = lambda jd: jd

    def _boom(*a, **kw):
        raise AssertionError("dispatch must NOT fire at cp_size=1")

    with patch("context_parallel.apply_dualchunkswap_to_jagged", side_effect=_boom):
        out_jd, _ = block.forward(embeddings={}, batch=None)

    assert captured["values_shape"][0] == total_global_tokens
    assert out_jd.values.shape[0] == total_global_tokens


def _make_cp_config_stub(
    *,
    cp_size: int,
    hstu_layer_type,
    sequence_parallel: bool,
):
    """Build a minimal duck-typed config for `_validate_cp_config`.

    `HSTUBlock._validate_cp_config` reads only the three CP-relevant
    fields, so we don't need to construct a full `HSTUConfig` (which
    would require a Megatron `parallel_state` init for TP/PP queries).
    A `SimpleNamespace` is sufficient and keeps the test honest: it
    exercises the *exact* code path that `HSTUBlock.__init__` runs.
    """
    from types import SimpleNamespace

    return SimpleNamespace(
        context_parallel_size=cp_size,
        hstu_layer_type=hstu_layer_type,
        sequence_parallel=sequence_parallel,
    )


def test_block_cp_rejects_fused_layer(cuda_device: torch.device) -> None:
    """cp_size>1 requires HSTULayerType.NATIVE; FUSED has no CP wrapper.

    Behavioural reject (Codex round-3 IMPORTANT). Calls the same
    classmethod that `HSTUBlock.__init__` runs at __init__ time, so
    a refactor that silently removes the guard breaks this test.
    """
    from configs.hstu_config import HSTULayerType
    from modules.hstu_block import HSTUBlock

    config = _make_cp_config_stub(
        cp_size=2, hstu_layer_type=HSTULayerType.FUSED, sequence_parallel=False
    )
    with pytest.raises(ValueError, match="HSTULayerType.NATIVE"):
        HSTUBlock._validate_cp_config(config)

    # Sanity: cp_size=1 with FUSED must NOT raise (the layer-type
    # restriction only applies under CP).
    safe = _make_cp_config_stub(
        cp_size=1, hstu_layer_type=HSTULayerType.FUSED, sequence_parallel=False
    )
    HSTUBlock._validate_cp_config(safe)  # must not raise

    # Sanity: cp_size>1 with NATIVE + sequence_parallel=False is OK.
    ok_native = _make_cp_config_stub(
        cp_size=2, hstu_layer_type=HSTULayerType.NATIVE, sequence_parallel=False
    )
    HSTUBlock._validate_cp_config(ok_native)  # must not raise


def test_block_cp_rejects_sequence_parallel(cuda_device: torch.device) -> None:
    """cp_size>1 must reject `sequence_parallel=True` (Codex round-2 IMPORTANT;
    upgraded to behavioural reject in round-3).

    The SP preprocessor scatters `jd.values` along row-dim by tp_size
    (`hstu_processor.py:scatter_to_sequence_parallel_region`); the CP
    dispatch then indexes that scattered tensor with global
    `seqlen_offsets` — shape/index mismatch. `_validate_cp_config`
    must fail loudly when both are configured. Calling it directly
    means a regressor that drops only the SP check (but keeps the
    FUSED check) still fails the test.
    """
    from configs.hstu_config import HSTULayerType
    from modules.hstu_block import HSTUBlock

    config = _make_cp_config_stub(
        cp_size=2, hstu_layer_type=HSTULayerType.NATIVE, sequence_parallel=True
    )
    with pytest.raises(ValueError, match="sequence_parallel"):
        HSTUBlock._validate_cp_config(config)

    # Sanity: cp_size=1 with sequence_parallel=True must NOT raise
    # (SP is fine on its own; only the CP+SP stack is rejected).
    safe = _make_cp_config_stub(
        cp_size=1, hstu_layer_type=HSTULayerType.NATIVE, sequence_parallel=True
    )
    HSTUBlock._validate_cp_config(safe)  # must not raise
