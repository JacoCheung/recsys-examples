# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Slice 5 / T5.2 — multi-GPU CP perf harness.

Measures forward-only step time of `hstu_attn_varlen_cp_func` at
cp_size ∈ {1, 2, 4, 8} on a fixed shape grid and reports per-token cost
relative to the single-GPU baseline. Used by T5.3 to verify the perf
gate: cp_size=4 step time ≤ 1.5× single-GPU per-token (plan §Phase 4).

Run modes
---------
    # Single-GPU baseline (call this script as a normal Python program):
    python examples/hstu/cp/bench/bench_cp.py --cp-size 1 \
        --output /tmp/bench_cp_size1.json

    # Multi-GPU CP (must be launched under torchrun with --nproc-per-node = cp_size):
    torchrun --standalone --nproc-per-node 4 \
        examples/hstu/cp/bench/bench_cp.py --cp-size 4 \
        --output /tmp/bench_cp_size4.json

The wrapper `examples/hstu/cp/run_bench_cp.sh` covers cp_size ∈ {1,2,4,8}
end-to-end and prints the perf gate verdict.

Output JSON shape
-----------------
{
  "commit": "...",
  "device": "NVIDIA A100 80GB PCIe",
  "cp_size": 4,
  "warmup": 50,
  "iters": 30,
  "world_size": 4,
  "rank": 0,
  "shapes": [
      {
        "label": "h4_d128_b8_s4096",
        "median_ms": 0.421,
        "p95_ms": 0.435,
        "tokens_per_s": 7.78e7,
        "global_tokens": 32768,
        "local_tokens": 8192
      },
      ...
  ]
}

Each shape's `median_ms` is the per-rank per-step wall-clock for the CP
forward call with this rank's local shard. `tokens_per_s` is computed
against `global_tokens` (so cp_size=N rows are directly comparable to
cp_size=1 rows on a per-token basis).
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

# Make `from context_parallel import ...` resolve when we run this script
# directly under torchrun (no pytest, no conftest.py to set this up).
# Append (don't insert at index 0) so installed packages still win — same
# rationale as the cp/test/conftest.py append.
sys.path.append(str(Path(__file__).resolve().parents[2]))  # examples/hstu/

from context_parallel import (  # noqa: E402
    get_batch_on_this_cp_rank_for_hstu,
    hstu_attn_varlen_cp_func,
)
from hstu import hstu_attn_varlen_func  # noqa: E402  — installed kernel

# Shape grid for the CP harness. Bigger than `baseline.py`'s grid because
# small shapes are dominated by Python-side dispatch and don't characterise
# CP comm/compute trade-off — the perf gate is meaningful at "real" shapes
# where attention compute >> P2P latency.
SHAPE_GRID: list[dict] = [
    dict(label="h4_d128_b8_s2048", batch=8, seqlen=2048, num_heads=4, head_dim=128),
    dict(label="h4_d128_b8_s4096", batch=8, seqlen=4096, num_heads=4, head_dim=128),
    dict(label="h4_d128_b8_s8192", batch=8, seqlen=8192, num_heads=4, head_dim=128),
    dict(label="h8_d128_b4_s8192", batch=4, seqlen=8192, num_heads=8, head_dim=128),
]


def _git_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _device_label() -> str:
    return torch.cuda.get_device_properties(0).name


def _build_equal_len_batch(
    batch: int,
    seqlen: int,
    num_heads: int,
    head_dim: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    device: torch.device,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(seed)
    total = batch * seqlen
    q = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    k = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    v = torch.randn(total, num_heads, head_dim, generator=g, dtype=dtype, device=device)
    cu = torch.arange(0, total + 1, seqlen, dtype=torch.int32, device=device)
    return q, k, v, cu


def _time_one_shape_single_gpu(
    shape: dict, *, warmup: int, iters: int, device: torch.device
) -> dict:
    """cp_size=1 path: time `hstu_attn_varlen_func` directly (the runtime kernel
    the CP wrapper short-circuits to in cp=1)."""
    q, k, v, cu = _build_equal_len_batch(
        shape["batch"],
        shape["seqlen"],
        shape["num_heads"],
        shape["head_dim"],
        device=device,
    )
    alpha = 1.0 / (shape["head_dim"] ** 0.5)
    max_s = shape["seqlen"]
    total_tokens = shape["batch"] * shape["seqlen"]

    def _step() -> torch.Tensor:
        return hstu_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            scaling_seqlen=max_s,
            num_contexts=None,
            num_targets=None,
            target_group_size=1,
            window_size=(-1, 0),
            alpha=alpha,
        )

    for _ in range(warmup):
        _step()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    samples_ms.sort()
    median_ms = statistics.median(samples_ms)
    p95_ms = samples_ms[min(int(0.95 * iters), iters - 1)]
    tokens_per_s = (
        total_tokens / (median_ms / 1000.0) if median_ms > 0 else float("inf")
    )
    return dict(
        label=shape["label"],
        median_ms=median_ms,
        p95_ms=p95_ms,
        tokens_per_s=tokens_per_s,
        global_tokens=total_tokens,
        local_tokens=total_tokens,
    )


def _time_one_shape_cp(
    shape: dict,
    *,
    warmup: int,
    iters: int,
    cp_group: dist.ProcessGroup,
    cp_global_ranks: Sequence[int],
    cp_size: int,
    cp_rank: int,
    device: torch.device,
    mask_mode: str = "causal",
) -> dict:
    """cp_size > 1 path: time `hstu_attn_varlen_cp_func` on the local shard.

    `mask_mode` selects the mask spec passed to the wrapper:
      - "causal" (default): plain causal, routes through `_multi_gpu_forward`.
      - "het_targets": num_targets set, target_group_size=2; routes
        through `_multi_gpu_forward_arbitrary` (per-step `func` builder).
      - "sliding": window_size=(w, 0); also routes through arbitrary.
    """
    q_global, k_global, v_global, cu_global = _build_equal_len_batch(
        shape["batch"],
        shape["seqlen"],
        shape["num_heads"],
        shape["head_dim"],
        device=device,
    )
    alpha = 1.0 / (shape["head_dim"] ** 0.5)
    max_s = shape["seqlen"]
    global_tokens = shape["batch"] * shape["seqlen"]

    q_loc, k_loc, v_loc, cu_loc, _, _ = get_batch_on_this_cp_rank_for_hstu(
        q_global, k_global, v_global, cu_global, cp_size=cp_size, cp_rank=cp_rank
    )
    local_tokens = q_loc.shape[0]

    if mask_mode == "causal":
        nc, nt, tgs, ws = None, None, 1, (-1, 0)
    elif mask_mode == "het_targets":
        # 1/8th of each sample is target region, group_size=2.
        nt_per_sample = max(2, max_s // 8)
        nt = torch.full(
            (shape["batch"],), nt_per_sample, dtype=torch.int32, device=device
        )
        nc, tgs, ws = None, 2, (-1, 0)
    elif mask_mode == "sliding":
        nc, nt, tgs = None, None, 1
        # Sliding window = 1/4 of seqlen (still routes through arbitrary).
        ws = (max(8, max_s // 4), 0)
    else:
        raise ValueError(f"unknown mask_mode={mask_mode!r}")

    def _step() -> torch.Tensor:
        return hstu_attn_varlen_cp_func(
            q=q_loc,
            k=k_loc,
            v=v_loc,
            cu_seqlens_q=cu_loc,
            cu_seqlens_k=cu_loc,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=max_s,
            max_seqlen_k=max_s,
            scaling_seqlen=max_s,
            num_contexts=nc,
            num_targets=nt,
            target_group_size=tgs,
            window_size=ws,
            alpha=alpha,
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
        )

    for _ in range(warmup):
        _step()
    dist.barrier()
    torch.cuda.synchronize()

    samples_ms: list[float] = []
    for _ in range(iters):
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _step()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        samples_ms.append((t1 - t0) * 1000.0)

    # Reduce per-rank medians to the worst (slowest) rank — that is the
    # critical path that bounds end-to-end step time.
    samples_ms.sort()
    median_ms = statistics.median(samples_ms)
    p95_ms = samples_ms[min(int(0.95 * iters), iters - 1)]
    median_t = torch.tensor([median_ms], device=device)
    p95_t = torch.tensor([p95_ms], device=device)
    dist.all_reduce(median_t, op=dist.ReduceOp.MAX, group=cp_group)
    dist.all_reduce(p95_t, op=dist.ReduceOp.MAX, group=cp_group)
    median_ms = float(median_t.item())
    p95_ms = float(p95_t.item())

    tokens_per_s = (
        global_tokens / (median_ms / 1000.0) if median_ms > 0 else float("inf")
    )
    return dict(
        label=shape["label"],
        median_ms=median_ms,
        p95_ms=p95_ms,
        tokens_per_s=tokens_per_s,
        global_tokens=global_tokens,
        local_tokens=local_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-GPU HSTU CP benchmark")
    parser.add_argument("--cp-size", type=int, required=True, choices=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--mask-mode",
        choices=["causal", "het_targets", "sliding"],
        default="causal",
        help="Mask spec for the CP path (cp_size > 1). `causal` routes "
        "through `_multi_gpu_forward` (3-region tile classifier); "
        "`het_targets` and `sliding` route through "
        "`_multi_gpu_forward_arbitrary` (per-step `func` builder). "
        "Used to characterise the arbitrary-mask path overhead vs the "
        "plain-causal path on the same shapes.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    payload = dict(
        commit=_git_commit_sha(),
        cp_size=args.cp_size,
        warmup=args.warmup,
        iters=args.iters,
        mask_mode=args.mask_mode,
    )

    if args.cp_size == 1:
        device = torch.device("cuda:0")
        payload["device"] = _device_label()
        payload["world_size"] = 1
        payload["rank"] = 0
        results: list[dict] = []
        for shape in SHAPE_GRID:
            print(f"== {shape['label']}")
            res = _time_one_shape_single_gpu(
                shape, warmup=args.warmup, iters=args.iters, device=device
            )
            print(
                f"   median={res['median_ms']:.3f}ms  p95={res['p95_ms']:.3f}ms  "
                f"throughput={res['tokens_per_s']:.3e} tokens/s"
            )
            results.append(res)
        payload["shapes"] = results
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.output}")
        return

    # cp_size > 1: must run under torchrun.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size != args.cp_size:
        raise RuntimeError(
            f"--cp-size={args.cp_size} requires torchrun --nproc-per-node={args.cp_size}; "
            f"WORLD_SIZE={world_size}"
        )
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    cp_group = dist.new_group(list(range(world_size)), backend="nccl")
    cp_global_ranks = list(range(world_size))
    payload["device"] = _device_label()
    payload["world_size"] = world_size
    payload["rank"] = rank

    results = []
    for shape in SHAPE_GRID:
        if rank == 0:
            print(f"== {shape['label']}")
        res = _time_one_shape_cp(
            shape,
            warmup=args.warmup,
            iters=args.iters,
            cp_group=cp_group,
            cp_global_ranks=cp_global_ranks,
            cp_size=args.cp_size,
            cp_rank=rank,
            device=device,
            mask_mode=args.mask_mode,
        )
        if rank == 0:
            print(
                f"   median={res['median_ms']:.3f}ms  p95={res['p95_ms']:.3f}ms  "
                f"throughput={res['tokens_per_s']:.3e} tokens/s "
                f"(global={res['global_tokens']} local={res['local_tokens']})"
            )
        results.append(res)
    payload["shapes"] = results

    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.output}")
    dist.barrier()


if __name__ == "__main__":
    main()
