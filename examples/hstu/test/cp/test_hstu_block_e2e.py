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
gather / postprocessor wiring bugs that the unit tests miss. It is
NOT a numerical-correctness oracle — that role belongs to the
kernel-level CP tests (`test_cp_forward.py`, `test_cp_backward.py`)
which compare gathered output and gradients against the single-GPU
single-rank baseline. Together those three files form the regression
matrix:

  - test_cp_forward.py / test_cp_backward.py: kernel-level numerical
    correctness vs single-GPU baseline. Catches mask, ring, scatter
    bugs at the math level. Required pass for v0 ship.
  - test_hstu_block_cp.py: HSTUBlock dispatch/gather wiring with
    stubbed layers (this file's predecessor — single-GPU only).
  - test_hstu_block_e2e.py (this file): integration smoke — proves
    the stubbed wiring works with REAL layers on REAL Megatron CP
    state and that the backward graph isn't replicated-local
    (each CP rank backprops its own Q chunk, so per-rank pre-reduce
    grads diverge above the bf16 noise floor).

Codex round-4 NOTE on coverage: this test's grad-divergence check
(items 6 below) catches the "CP completely disabled — every rank
runs full batch locally" failure mode (verified by prove-it: setting
`cp_active = False` in `HSTUBlock.forward` makes `rel_diff` collapse
to ~0% and the assertion fires). It does NOT prove the backward
reverse-ring is delivering peer-K contributions correctly — that
proof comes from `test_cp_backward.py`'s direct kernel-level
gradient comparison against the single-GPU baseline. The kernel
test is required to pass in the same regression matrix.

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
    # Teardown: destroy parallel state so a re-imported test (or
    # follow-up parametrize cell) starts clean. Codex round-2 Q3 —
    # the production helper `commons.utils.initialize.destroy_global_state`
    # (lines 65-79) destroys the TP and DP-with-CP groups before
    # `destroy_model_parallel()`; mirror that order here so any
    # in-process re-import sees a clean state.
    import gc

    torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    if parallel_state.model_parallel_is_initialized():
        if parallel_state.get_tensor_model_parallel_world_size() == 1:
            torch.distributed.destroy_process_group(
                group=parallel_state.get_tensor_model_parallel_group()
            )
            torch.distributed.destroy_process_group(
                group=parallel_state.get_data_parallel_group(with_context_parallel=True)
            )
        parallel_state.destroy_model_parallel()
    torch.cuda.empty_cache()
    gc.collect()


def _build_fixed_length_batch(
    *,
    batch_size: int,
    seqlen: int,
    seed: int,
    device: torch.device,
):
    """Build a deterministic HSTUBatch with all samples of the same length.

    DualChunkSwap requires `L_b % (2 * cp_size) == 0` per sample, so
    we sidestep variable-length alignment by using a single fixed
    length divisible by `2 * cp_size` for every sample. Same seed
    across all ranks → identical batch on every rank (the global
    input).

    Codex round-2 Q4 fixed: a previous version called
    `HSTUBatch.random(...)` (which sizes values to a per-sample
    random length sum, generally less than `n_samples * seqlen`),
    then sliced `feats.values()[:n_samples * seqlen]` — that slice
    silently truncated to the under-sized values tensor and produced
    a length/value mismatch in the KJT. Fix: build the KJT directly
    from `torch.randint(...)` sized to `n_samples * seqlen`, sized
    correctly for the fixed lengths.
    """
    from commons.datasets.hstu_batch import HSTUBatch
    from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

    torch.manual_seed(seed)
    n_samples = batch_size
    total = n_samples * seqlen
    item_values = torch.randint(0, 1000, (total,), dtype=torch.int64, device=device)
    fixed_lengths = torch.full((n_samples,), seqlen, dtype=torch.int32, device=device)
    feats = KeyedJaggedTensor.from_lengths_sync(
        keys=["item"], values=item_values, lengths=fixed_lengths
    )
    return HSTUBatch(
        features=feats,
        batch_size=n_samples,
        feature_to_max_seqlen={"item": seqlen},
        item_feature_name="item",
        action_feature_name=None,
        max_num_candidates=0,
        num_candidates=None,
    )


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

    # Fixed-length batch: every sample has length `seqlen`, divisible
    # by `2 * world_size`. cp=2 needs %4 == 0; cp=4 needs %8 == 0;
    # 24 satisfies both.
    batch_size = 4
    seqlen = 24
    assert (
        seqlen % (2 * world_size) == 0
    ), f"seqlen={seqlen} must be divisible by 2*cp_size={2 * world_size}"
    hidden_dim_per_head = 32
    num_heads = 2
    hidden_size = hidden_dim_per_head * num_heads

    batch = _build_fixed_length_batch(
        batch_size=batch_size,
        seqlen=seqlen,
        seed=42,
        device=device,
    )

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

    # 6. CP-distinguishing backward check (Codex round-3 Q2): the
    # right invariant is the INVERSE of round-1's broken assertion.
    # Real CP: each rank computes a PARTIAL gradient on its own Q
    # chunk; per-rank pre-reduce `param.grad` MUST DIFFER across
    # ranks because each rank backprops different Q rows.
    # CP-disabled fallback (every rank runs the full batch
    # locally): pre-reduce grads are IDENTICAL across ranks.
    #
    # So the test for "CP backward actually fired" is: per-rank
    # pre-reduce grad signature should differ from rank 0's
    # signature by a non-trivial relative amount.
    #
    # Round-1 asserted equality (wrong direction; passed by bf16
    # noise). Round-2 asserted post-reduce monotonicity (passes
    # trivially under CP-disabled — `post = world_size × pre` still
    # satisfies post >= pre, per Codex round-3 BLOCKER).
    grad_sig = sum(
        p.grad.detach().float().abs().sum().item()
        for p in block.parameters()
        if p.grad is not None
    )
    sig_t = torch.tensor([grad_sig], device=device, dtype=torch.float64)
    sig_other = torch.zeros_like(sig_t)
    if rank == 0:
        for r in range(1, world_size):
            dist.send(sig_t, dst=r)
    else:
        dist.recv(sig_other, src=0)
        # Real CP produces non-trivially different partials across
        # ranks. Threshold 0.5% relative diff catches the
        # "identical-grad CP-disabled" failure mode while staying
        # above bf16 numerical noise (~0.01% from the same input).
        rel_diff = abs(grad_sig - sig_other.item()) / max(sig_other.item(), 1e-9)
        assert rel_diff > 5e-3, (
            f"rank {rank}: grad signature {grad_sig:.6f} matches rank 0 "
            f"{sig_other.item():.6f} (rel diff {rel_diff:.6f}) — looks "
            "like CP-disabled fallback (every rank ran the full batch "
            "locally instead of CP-sharded backward)."
        )
