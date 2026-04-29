# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SPEC §9.1 deferred analysis — DualChunkSwap padding cost measurement.

DualChunkSwap requires `global_seqlen % (2 * cp_size) == 0` per sample
(SPEC §3 / dispatch-helper guard). A batch with seqlens not all
divisible by `2*cp_size` therefore has to pad each sample up to the
next multiple. This script measures that padding overhead on
representative recsys-style seqlen distributions.

Outputs a per-cp_size table:
  cp_size  raw_tokens  padded_tokens  padding/total
  1        N           N              0%
  2        N           N + p2         p2 / (N + p2)
  4        N           N + p4         p4 / (N + p4)
  8        N           N + p8         p8 / (N + p8)

The SPEC §9 escalation threshold: if padding/total > 30%, the
heterogeneous-batch CP path needs a different chunk dispatcher
(Track B) before being wired into the training loop. Below 30%,
v0/v0.5 ship as-is and pad the few stragglers.

This is the analysis the v0 plan deferred ("padding-cost measurement
deferred to before Slice 6"). No GPU required; pure CPU.

Usage:
    python examples/hstu/cp/bench/padding_cost.py
    python examples/hstu/cp/bench/padding_cost.py --distribution kuairand
    python examples/hstu/cp/bench/padding_cost.py --custom 16,32,64,128,256
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

# Representative recsys seqlen distributions. These are coarse-grained
# stand-ins for production data; replace with empirical samples per
# dataset when available. Values are global per-sample seqlens (one
# entry per sample in a typical batch).
DISTRIBUTIONS: dict[str, list[int]] = {
    # Roughly approximates typical movie/short-video user histories
    # (KuaiRand-1k, MovieLens-32M trim): bimodal — many short users
    # and a tail of long ones.
    "synthetic_recsys": (
        [16] * 30
        + [32] * 30
        + [64] * 25
        + [128] * 20
        + [256] * 15
        + [512] * 10
        + [1024] * 6
        + [2048] * 3
        + [4096] * 1
    ),
    # Pure ranking-style: shorter histories, less long-tail.
    "synthetic_ranking": ([32] * 50 + [64] * 40 + [128] * 30 + [256] * 15 + [512] * 5),
    # KuaiRand-1k specific approximation. Real numbers should be
    # plugged in from `kuairand_1k` raw data once available.
    "kuairand_1k_approx": (
        [16] * 25
        + [32] * 35
        + [64] * 30
        + [128] * 25
        + [256] * 15
        + [512] * 8
        + [1024] * 4
    ),
}


def _round_up_to(x: int, multiple: int) -> int:
    """Smallest m such that m >= x and m % multiple == 0."""
    if multiple <= 1:
        return x
    rem = x % multiple
    return x if rem == 0 else x + (multiple - rem)


def _measure(seqlens: Sequence[int], cp_size: int) -> dict:
    """Compute padding cost for one (distribution, cp_size) pair."""
    raw = sum(seqlens)
    padded_per_sample = [_round_up_to(s, 2 * cp_size) for s in seqlens]
    padded = sum(padded_per_sample)
    pad_only = padded - raw
    pad_frac = pad_only / padded if padded > 0 else 0.0
    return {
        "cp_size": cp_size,
        "raw_tokens": raw,
        "padded_tokens": padded,
        "padding_only": pad_only,
        "padding_fraction": pad_frac,
    }


def _format_row(r: dict) -> str:
    return (
        f"  cp={r['cp_size']:<3}"
        f" raw={r['raw_tokens']:<8}"
        f" padded={r['padded_tokens']:<8}"
        f" pad={r['padding_only']:<6}"
        f" pad_frac={r['padding_fraction'] * 100:5.2f}%"
        f" {'OK' if r['padding_fraction'] <= 0.30 else 'ESCALATE'}"
    )


def _seqlen_summary(seqlens: Sequence[int]) -> str:
    return (
        f"  N={len(seqlens)} samples"
        f" min={min(seqlens)}"
        f" median={int(statistics.median(seqlens))}"
        f" max={max(seqlens)}"
        f" mean={statistics.mean(seqlens):.1f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DualChunkSwap padding cost analysis (SPEC §9.1)"
    )
    parser.add_argument(
        "--distribution",
        choices=list(DISTRIBUTIONS.keys()),
        default="synthetic_recsys",
    )
    parser.add_argument(
        "--custom",
        help="Comma-separated seqlens (overrides --distribution)",
        default=None,
    )
    parser.add_argument(
        "--cp-sizes", default="1,2,4,8", help="CSV of cp_size values to score"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Optional JSON output"
    )
    args = parser.parse_args()

    if args.custom:
        seqlens = [int(s) for s in args.custom.split(",") if s.strip()]
        dist_label = "custom"
    else:
        seqlens = DISTRIBUTIONS[args.distribution]
        dist_label = args.distribution

    cp_sizes = [int(c) for c in args.cp_sizes.split(",")]

    print(f"Distribution: {dist_label}")
    print(_seqlen_summary(seqlens))
    print()
    print("DualChunkSwap padding cost (per-batch sum):")

    results = []
    for cp in cp_sizes:
        r = _measure(seqlens, cp)
        print(_format_row(r))
        results.append(r)

    print()
    threshold = 0.30
    over = [r for r in results if r["padding_fraction"] > threshold]
    if over:
        print(
            f"WARN: {len(over)} cp_size(s) exceed the 30% padding threshold "
            f"(SPEC §9.1): cp_sizes = "
            + ", ".join(str(r["cp_size"]) for r in over)
            + ". Heterogeneous-batch CP path needs Track B dispatcher; do "
            "NOT wire current v0/v0.5 chunk-uniform CP into training loop "
            "for these cp_sizes against this distribution."
        )
        rc = 1
    else:
        print(
            "OK: every cp_size's padding fraction is within the 30% gate. "
            "v0/v0.5 chunk-uniform CP is safe to wire into training for "
            "this distribution."
        )
        rc = 0

    if args.output is not None:
        payload = {
            "distribution": dist_label,
            "n_samples": len(seqlens),
            "raw_tokens": sum(seqlens),
            "results": results,
            "threshold": threshold,
            "verdict": "ok" if rc == 0 else "escalate",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nWrote {args.output}")

    sys.exit(rc)


if __name__ == "__main__":
    main()
