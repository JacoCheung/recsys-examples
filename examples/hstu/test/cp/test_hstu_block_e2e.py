# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
T6.5 E2E smoke — full `HSTUBlock` forward+backward under cp_size>1.

This is the ship-gate test for Slice 6. T6.4 (`test_hstu_block_cp.py`)
covers the dispatch/gather wiring with stubbed components on a single
GPU; this file builds a real `HSTUBlock` (real preprocessor + real
NATIVE layer stack + real postprocessor) under torchrun cp=2 and
verifies that:

  1. Construction succeeds when Megatron `parallel_state` is
     initialised with `context_parallel_size=2`.
  2. Forward returns finite, non-zero output.
  3. Every rank gathers the SAME global output (CP must be symmetric:
     after the gather inside `HSTUBlock.forward`, both ranks hold the
     full-shape output bit-equal at fp32, close at bf16).
  4. Backward succeeds — `.backward()` runs and gradients on
     trainable parameters are finite and non-zero.

This is the integration gate that catches preprocessor / layer-stack /
gather / postprocessor wiring bugs that the unit tests miss.

We don't compare against a separate cp=1 oracle in this same process
because Megatron `parallel_state` is global state — running cp=1
inside a process already initialised at cp=2 requires destroy + re-
init which is brittle. The cross-rank symmetry check (#3) plus the
unit-level cp=1 path in `test_hstu_block_cp.py::
test_block_cp_size_1_skips_dispatch` together cover both directions.

Run:
    bash examples/hstu/cp/run_cp_tests.sh    # part of regression
    # or directly:
    WORLD_SIZE=2 NCCL_P2P_DISABLE=1 torchrun --standalone \
        --nproc_per_node=2 -m pytest \
        examples/hstu/test/cp/test_hstu_block_e2e.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterator

import pytest
import torch
import torch.distributed as dist

# We need both `examples/hstu/` (for `configs`/`modules`) and
# `examples/` (for `commons`) on sys.path. Match the conftest pattern.
_HSTU_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_ROOT = Path(__file__).resolve().parents[3]
for p in (_HSTU_ROOT, _EXAMPLES_ROOT):
    if str(p) not in sys.path:
        sys.path.append(str(p))


@pytest.fixture(scope="module")
def cp_world() -> Iterator[dict]:
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < 2:
        pytest.skip("E2E test requires WORLD_SIZE >= 2 (run under torchrun)")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    # Initialise Megatron parallel-state with CP=world_size. HSTUBlock
    # construction reads `parallel_state.get_context_parallel_group()`
    # so this must run before `HSTUBlock(config)`.
    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            context_parallel_size=world_size,
        )
    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    yield dict(
        rank=rank,
        world_size=world_size,
        device=torch.device(f"cuda:{rank}"),
    )
    dist.barrier()


def _build_random_batch(
    *,
    batch_size: int,
    max_seqlen: int,
    seed: int,
    device: torch.device,
):
    """Build a deterministic HSTUBatch with item-only features.

    Seqlens are divisible by 2*cp_size to satisfy DualChunkSwap. Same
    seed across all ranks → identical batch on every rank (the global
    input).
    """
    from commons.datasets.hstu_batch import FeatureConfig, HSTUBatch

    torch.manual_seed(seed)

    feature_configs = [
        FeatureConfig(
            feature_names=["item"],
            max_item_ids=[1000],
            max_sequence_length=max_seqlen,
            is_jagged=True,
        )
    ]
    return HSTUBatch.random(
        batch_size=batch_size,
        feature_configs=feature_configs,
        item_feature_name="item",
        contextual_feature_names=[],
        action_feature_name=None,
        max_num_candidates=0,
        device=device,
    )


def _enforce_chunk_alignment(batch, cp_size: int):
    """Pad sample lengths to be divisible by 2*cp_size.

    DualChunkSwap requires `L_b % (2 * cp_size) == 0` per sample.
    The random batch generator gives arbitrary lengths in
    `[1, max_seqlen]`. Round each up to the next multiple of
    2*cp_size by truncating the KJT lengths.
    """
    chunks = 2 * cp_size
    new_lengths = (batch.features.lengths() // chunks) * chunks
    # Ensure no zero-length sample (DualChunkSwap rejects).
    new_lengths = torch.where(
        new_lengths > 0, new_lengths, torch.full_like(new_lengths, chunks)
    )
    # Truncate the KJT.
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    feats = batch.features
    keys = feats.keys()
    n_keys = len(keys)
    old_lengths = feats.lengths().view(n_keys, -1)
    new_lengths_2d = new_lengths.view(n_keys, -1)
    # Build new values by slicing each sample's prefix.
    old_values = feats.values()
    old_offsets = feats.offsets()
    new_values_list = []
    for k in range(n_keys):
        for s in range(old_lengths.shape[1]):
            base = int(old_offsets[k * old_lengths.shape[1] + s].item())
            take = int(new_lengths_2d[k, s].item())
            new_values_list.append(old_values[base : base + take])
    new_values = torch.cat(new_values_list)
    new_feats = KeyedJaggedTensor.from_lengths_sync(
        keys=keys, values=new_values, lengths=new_lengths
    )
    import dataclasses

    return dataclasses.replace(batch, features=new_feats)


def test_hstu_block_e2e_forward_backward(cp_world: dict) -> None:
    """Build a real `HSTUBlock` with `cp_size=world_size`, run
    fwd+bwd on a deterministic varlen batch, and assert the gather
    contract holds across ranks."""
    pytest.importorskip("megatron.core")

    from configs import get_hstu_config
    from configs.hstu_config import HSTULayerType, KernelBackend
    from modules.hstu_block import HSTUBlock
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    rank = cp_world["rank"]
    world_size = cp_world["world_size"]
    device = cp_world["device"]

    # Small but realistic shapes. seqlens divisible by 2*cp_size after
    # `_enforce_chunk_alignment`.
    batch_size = 4
    max_seqlen = 32  # plenty of headroom for chunk alignment
    hidden_dim_per_head = 32
    num_heads = 2
    hidden_size = hidden_dim_per_head * num_heads

    batch = _build_random_batch(
        batch_size=batch_size,
        max_seqlen=max_seqlen,
        seed=42,
        device=device,
    )
    batch = _enforce_chunk_alignment(batch, cp_size=world_size)

    # Build the HSTU config with cp wired in. NATIVE layer is the only
    # CP-supported layer type. The HSTU kernel only accepts fp16/bf16
    # (raises "HSTU only support fp16 and bf16 data type" on fp32),
    # so we run bf16 + wrap in `Float16Module` to align layernorm
    # weight dtype with the input dtype (matches the production
    # training pattern in `examples/hstu/test/test_hstu_layer.py`).
    hstu_config = get_hstu_config(
        hidden_size=hidden_size,
        kv_channels=hidden_dim_per_head,
        num_attention_heads=num_heads,
        num_layers=1,
        dtype=torch.bfloat16,
        is_causal=True,
        kernel_backend=KernelBackend.CUTLASS,
        target_group_size=1,
        hstu_layer_type=HSTULayerType.NATIVE,
        sequence_parallel=False,
    )
    # `get_hstu_config` reads cp_size from `parallel_state` directly;
    # confirm it landed at world_size.
    assert (
        hstu_config.context_parallel_size == world_size
    ), f"expected cp_size={world_size}, got {hstu_config.context_parallel_size}"

    from megatron.core.transformer.module import Float16Module

    block = HSTUBlock(hstu_config)
    block = Float16Module(hstu_config, block).to(device)

    # Build embeddings (feature → KJT). The preprocessor consumes a
    # dict[feature_name → JaggedTensor], not the KJT directly.
    feats = batch.features
    seqlen_sum = int(feats.lengths().sum().item())
    torch.manual_seed(43)
    embedding_values = torch.randn(
        seqlen_sum, hidden_size, dtype=torch.bfloat16, device=device, requires_grad=True
    )
    embeddings = KeyedJaggedTensor.from_lengths_sync(
        keys=feats.keys(),
        values=embedding_values,
        lengths=feats.lengths(),
    ).to_dict()

    # Forward.
    out_jd, meta = block(embeddings, batch)

    # 1. Output shape: global (full token count) on every rank.
    expected_total_tokens = int(feats.lengths().sum().item())
    # Account for postprocessor target slicing — but with
    # max_num_candidates=0 the postprocessor passthrough doesn't slice.
    assert (
        out_jd.values.shape[0] == expected_total_tokens
    ), f"rank {rank}: expected {expected_total_tokens} tokens, got {out_jd.values.shape[0]}"
    assert out_jd.values.shape[1] == hidden_size

    # 2. Output is finite + non-zero (it's a real layer with random
    # weights; sigmoid + linear produces non-trivial output).
    assert torch.isfinite(out_jd.values).all(), f"rank {rank}: NaN/Inf in output"
    assert (
        out_jd.values.abs().sum().item() > 0
    ), f"rank {rank}: output is all zero — gather likely broken"

    # 3. CROSS-RANK SYMMETRY: the gather inside `HSTUBlock.forward`
    # must produce the same global output on every rank. This is the
    # single strongest correctness check we have without a separate
    # cp=1 oracle process.
    # Cast to fp32 for tighter equality checks.
    out_fp32 = out_jd.values.detach().float()
    out_other = torch.zeros_like(out_fp32)
    if rank == 0:
        # rank 0 sends its output to every other rank for comparison.
        for r in range(1, world_size):
            dist.send(out_fp32, dst=r)
    else:
        dist.recv(out_other, src=0)
        diff = (out_fp32 - out_other).abs().max().item()
        assert diff < 1e-3, (
            f"rank {rank}: output differs from rank 0 by max {diff}; "
            "cp gather is asymmetric"
        )

    # 4. Backward: train a trivial MSE objective and verify gradients.
    target = torch.zeros_like(out_jd.values)
    loss = torch.nn.functional.mse_loss(out_jd.values.float(), target.float())
    loss.backward()

    # Embedding tensor's gradient must be finite + non-zero (block
    # actually back-props through the layer stack).
    assert embedding_values.grad is not None, "no grad on input embeddings"
    assert torch.isfinite(
        embedding_values.grad
    ).all(), f"rank {rank}: NaN/Inf in embedding grad"
    grad_abs = embedding_values.grad.abs().sum().item()
    assert grad_abs > 0, f"rank {rank}: embedding grad is zero — backward broken"

    # 5. At least one trainable param must have a non-trivial grad
    # (otherwise the layer didn't actually contribute).
    n_with_grad = 0
    for p in block.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            n_with_grad += 1
    assert n_with_grad > 0, f"rank {rank}: no block parameter received a gradient"
