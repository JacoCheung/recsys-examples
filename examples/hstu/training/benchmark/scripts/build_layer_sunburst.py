#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Build a sunburst chart of per-phase GPU time for fused HSTU layer from a
nsys sqlite export.

Iteration boundaries are taken from the outer ``hstu_layer_step <i>``
NVTX range emitted by hstu_layer_benchmark.py (older nsys-rep without
that outer marker falls back to implicit index alignment and no idle
slice).

For each iter:
  * Per-phase busy time = Σ GPU kernel duration for kernels whose CUDA
    runtime launch timestamp falls inside the phase's host NVTX range.
  * Iter GPU wall-clock = max(kernel.end) - min(kernel.start) of all
    kernels launched inside the outer ``hstu_layer_step <i>`` range.
  * idle = iter_wallclock - Σ phase_busy
    Represents fwd→bwd transition gap + intra-step stalls (anything the
    GPU was idle for during the step).

The "median step" shown on the sunburst is the iter whose *wall-clock*
is the median across captured iters (fwd_sum + bwd_sum + idle == that
iter's full GPU wall-clock, exactly).

Usage:
    # Given an nsys-rep, export to sqlite first, then:
    python build_layer_sunburst.py path/to/file.sqlite \
        --output path/to/sunburst.html --png path/to/sunburst.png

    # Typical end-to-end from within a container:
    nsys export --type sqlite --output X.sqlite --force-overwrite true X.nsys-rep
    python training/benchmark/scripts/build_layer_sunburst.py X.sqlite \
        --output X_sunburst.html --png X_sunburst.png
"""
import argparse
import sqlite3
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

FWD_PHASES = [
    "hstu ln+linear_bias+silu fwd",
    "hstu attn fwd",
    "hstu norm mul dropout fwd",
    "hstu linear_residual fwd",
]
BWD_PHASES = [
    "hstu linear_residual bwd",
    "norm_mul_dropout bwd",
    "hstu attn bwd",
    "ln_linear_silu bwd",
]


def fetch_named_events(cur, names: List[str]) -> Dict[str, List[Tuple[int, int]]]:
    """Return {name: [(start_ns, end_ns), ...]} from NVTX_EVENTS."""
    placeholders = ",".join("?" * len(names))
    q = f"""
        SELECT s.value, e.start, e.end
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE s.value IN ({placeholders})
        ORDER BY e.start
    """
    out: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    for name, start, end in cur.execute(q, names):
        out[name].append((start, end))
    return out


def phase_gpu_ms(
    phase_ranges: List[Tuple[int, int]],
    runtime_rows: List[Tuple[int, int]],  # (start_ns, correlationId)
    kernel_dur_by_corr: Dict[int, int],  # correlationId -> gpu_ns
) -> List[float]:
    """For each occurrence of a phase, sum GPU duration of kernels whose
    host launch falls inside that phase's host NVTX range. Returns ms."""
    # binary search via sorted runtime_rows would be ideal; N<<1000 so linear.
    out = []
    for r_start, r_end in phase_ranges:
        total_ns = 0
        for rt_start, corr in runtime_rows:
            if r_start <= rt_start < r_end:
                total_ns += kernel_dur_by_corr.get(corr, 0)
        out.append(total_ns / 1e6)  # ns → ms
    return out


def fetch_step_ranges(cur) -> List[Tuple[int, int, int]]:
    """Return [(iter_idx, start_ns, end_ns), ...] for outer step markers.

    Looks up NVTX events whose text matches `hstu_layer_step <N>`. Returns
    an empty list if the outer marker is absent (older nsys-rep).
    """
    q = """
        SELECT s.value, e.start, e.end
        FROM NVTX_EVENTS e
        JOIN StringIds s ON e.textId = s.id
        WHERE s.value LIKE 'hstu_layer_step %'
        ORDER BY e.start
    """
    out = []
    for name, start, end in cur.execute(q):
        try:
            idx = int(name.rsplit(" ", 1)[-1])
        except ValueError:
            continue
        out.append((idx, start, end))
    return out


def build(sqlite_path: str, output_html: str, output_png: str, label: str) -> None:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    # Kernels: keyed by correlationId → {start, end, duration_ns}.
    kernel_info: Dict[int, Tuple[int, int]] = {}  # corr -> (start_ns, end_ns)
    for corr, start, end in cur.execute(
        "SELECT correlationId, start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ):
        prev = kernel_info.get(corr)
        if prev is None:
            kernel_info[corr] = (start, end)
        else:
            # If a correlationId appears multiple times (rare), widen span.
            kernel_info[corr] = (min(prev[0], start), max(prev[1], end))
    kernel_dur = {corr: (e - s) for corr, (s, e) in kernel_info.items()}

    # Runtime launches: (start_ns, correlationId), sorted by start.
    runtime_rows = [
        (start, corr)
        for start, corr in cur.execute(
            "SELECT start, correlationId FROM CUPTI_ACTIVITY_KIND_RUNTIME ORDER BY start"
        )
    ]

    all_phases = FWD_PHASES + BWD_PHASES
    nvtx = fetch_named_events(cur, all_phases)

    missing = [p for p in all_phases if p not in nvtx]
    if missing:
        print(f"WARNING: these phases had no NVTX events: {missing}", file=sys.stderr)

    # Per-phase per-iter GPU ms (busy time, sum of kernels)
    per_phase_per_iter: Dict[str, List[float]] = {}
    for phase in all_phases:
        ranges = nvtx.get(phase, [])
        per_iter = phase_gpu_ms(ranges, runtime_rows, kernel_dur)
        per_phase_per_iter[phase] = per_iter
        med = statistics.median(per_iter) if per_iter else 0.0
        print(
            f"  {phase:<36s} n={len(per_iter):>3d}  "
            f"min={min(per_iter, default=0):6.3f}  "
            f"med={med:6.3f}  "
            f"max={max(per_iter, default=0):6.3f}  ms"
        )

    # ---- Identify iter boundaries + compute GPU wall-clock per iter ----
    # Preferred: outer `hstu_layer_step <i>` markers (emitted by
    # hstu_layer_benchmark.py --profile True). Fallback: implicit index
    # alignment (older logs — no idle slice available).
    step_ranges = fetch_step_ranges(cur)

    n_iters = len(nvtx[FWD_PHASES[0]])
    aligned = {
        p: per_phase_per_iter[p] + [0.0] * (n_iters - len(per_phase_per_iter[p]))
        for p in all_phases
    }

    if step_ranges:
        # Use the outer marker. For each iter range:
        #   * GPU wall-clock   = max(k.end) - min(k.start) across all kernels
        #     whose launch falls inside the step's host range
        #   * busy_union       = length of UNION of [k.start, k.end] intervals
        #                        (robust to stream overlap from async wgrad)
        #   * idle             = wall_clock - busy_union   (always ≥ 0)
        #   * per-phase busy   = Σ kernel duration for kernels in that phase
        #                        (may sum to > busy_union under stream overlap,
        #                        but rarely across fwd↔bwd for hstu-layer)
        # (iter_idx, wallclock_ms, busy_union_ms, idle_ms, phase_busy_dict)
        per_iter_info: List[Tuple[int, float, float, float, Dict[str, float]]] = []
        for iter_idx, step_start, step_end in step_ranges:
            kernel_intervals = []  # [(start, end), ...] for kernels in this iter
            for rt_start, corr in runtime_rows:
                if step_start <= rt_start < step_end and corr in kernel_info:
                    kernel_intervals.append(kernel_info[corr])
            if not kernel_intervals:
                continue
            kernel_intervals.sort()
            wallclock_ns = max(e for _, e in kernel_intervals) - min(
                s for s, _ in kernel_intervals
            )

            # Interval union length
            merged_s, merged_e = kernel_intervals[0]
            busy_union_ns = 0
            for s, e in kernel_intervals[1:]:
                if s > merged_e:
                    busy_union_ns += merged_e - merged_s
                    merged_s, merged_e = s, e
                else:
                    merged_e = max(merged_e, e)
            busy_union_ns += merged_e - merged_s

            wallclock_ms = wallclock_ns / 1e6
            busy_union_ms = busy_union_ns / 1e6
            idle_ms = max(0.0, wallclock_ms - busy_union_ms)

            # Per-phase raw busy (Σ of kernel durations; may slightly exceed
            # per-phase wall-clock if wgrad overlaps with dgrad on bwd).
            phase_busy = {}
            for phase in all_phases:
                dur = 0
                for pr_start, pr_end in nvtx.get(phase, []):
                    if step_start <= pr_start and pr_end <= step_end:
                        for rt_start, corr in runtime_rows:
                            if pr_start <= rt_start < pr_end:
                                dur += kernel_dur.get(corr, 0)
                phase_busy[phase] = dur / 1e6

            # Rescale phase busy so fwd + bwd + idle = wall_clock exactly in
            # the sunburst (preserves relative per-phase proportions).
            raw_busy_sum = sum(phase_busy.values())
            target_busy = wallclock_ms - idle_ms
            if raw_busy_sum > 0 and target_busy > 0:
                scale = target_busy / raw_busy_sum
                phase_busy = {p: v * scale for p, v in phase_busy.items()}
            per_iter_info.append(
                (iter_idx, wallclock_ms, busy_union_ms, idle_ms, phase_busy)
            )

        if not per_iter_info:
            print(
                "WARNING: no iters matched kernels; falling back to index alignment",
                file=sys.stderr,
            )
            step_ranges = []

    if step_ranges and per_iter_info:
        # Pick the median iter by WALL-CLOCK (includes idle).
        sorted_by_wc = sorted(per_iter_info, key=lambda x: x[1])
        n = len(sorted_by_wc)
        median_entry = sorted_by_wc[(n - 1) // 2]
        iter_idx = median_entry[0]
        wallclock_ms = median_entry[1]
        idle_ms = median_entry[3]
        phase_busy = median_entry[4]
        fwd_med = [phase_busy[p] for p in FWD_PHASES]
        bwd_med = [phase_busy[p] for p in BWD_PHASES]
        fwd_sum = sum(fwd_med)
        bwd_sum = sum(bwd_med)
        total = wallclock_ms
        print(
            f"\nMedian step (by GPU wall-clock): profile-iter index={iter_idx}, "
            f"{n} captured iters, wall-clock range {sorted_by_wc[0][1]:.2f} .. "
            f"{sorted_by_wc[-1][1]:.2f} ms"
        )
        print(
            f"  fwd busy :  {fwd_sum:.3f} ms  (Σ kernel dur, scaled to fit wall-clock)"
        )
        print(f"  bwd busy :  {bwd_sum:.3f} ms  (ditto)")
        print(
            f"  idle     :  {idle_ms:.3f} ms   (wall-clock − union-of-kernel-intervals)"
        )
        print(
            f"  e2e step :  {wallclock_ms:.3f} ms  (GPU wall-clock, max_end − min_start)"
        )
        has_idle = True
    else:
        # Fallback: implicit index alignment, no wall-clock / idle.
        per_iter_total = [
            sum(aligned[p][i] for p in all_phases) for i in range(n_iters)
        ]
        sorted_by_total = sorted(range(n_iters), key=lambda i: per_iter_total[i])
        median_idx = sorted_by_total[(n_iters - 1) // 2]

        fwd_med = [aligned[p][median_idx] for p in FWD_PHASES]
        bwd_med = [aligned[p][median_idx] for p in BWD_PHASES]
        fwd_sum = sum(fwd_med)
        bwd_sum = sum(bwd_med)
        total = fwd_sum + bwd_sum
        idle_ms = 0.0
        has_idle = False
        print(
            "\n(outer `hstu_layer_step` marker not found; using index alignment, "
            "no idle slice)"
        )
        print(f"Median iter {median_idx} of {n_iters}")
        print(f"  fwd busy:  {fwd_sum:.3f} ms")
        print(f"  bwd busy:  {bwd_sum:.3f} ms")
        print(f"  total   :  {total:.3f} ms")

    # ---- Plotly sunburst (optional) ----
    try:
        import plotly.graph_objects as go  # type: ignore

        labels = ["e2e step", "fwd", "bwd"]
        parents = ["", "e2e step", "e2e step"]
        values = [total, fwd_sum, bwd_sum]
        text = [
            f"{total:.2f} ms",
            f"{fwd_sum:.2f} ms<br>{fwd_sum / total * 100:.1f}% of step",
            f"{bwd_sum:.2f} ms<br>{bwd_sum / total * 100:.1f}% of step",
        ]
        for p, v in zip(FWD_PHASES, fwd_med):
            labels.append(p.replace("hstu ", ""))
            parents.append("fwd")
            values.append(v)
            pct_fwd = (v / fwd_sum * 100) if fwd_sum else 0.0
            pct_step = (v / total * 100) if total else 0.0
            text.append(
                f"{v:.2f} ms<br>{pct_fwd:.1f}% of fwd<br>{pct_step:.1f}% of step"
            )
        for p, v in zip(BWD_PHASES, bwd_med):
            labels.append(p)
            parents.append("bwd")
            values.append(v)
            pct_bwd = (v / bwd_sum * 100) if bwd_sum else 0.0
            pct_step = (v / total * 100) if total else 0.0
            text.append(
                f"{v:.2f} ms<br>{pct_bwd:.1f}% of bwd<br>{pct_step:.1f}% of step"
            )
        # Idle slice (only when outer marker gave us wall-clock context).
        if has_idle and idle_ms > 0:
            labels.append("idle")
            parents.append("e2e step")
            values.append(idle_ms)
            pct_step = (idle_ms / total * 100) if total else 0.0
            text.append(
                f"{idle_ms:.2f} ms<br>{pct_step:.1f}% of step<br>"
                "(GPU idle — fwd→bwd gap, host stalls)"
            )

        title_suffix = (
            f"GPU wall-clock; idle = {idle_ms:.2f} ms"
            if has_idle
            else "(no outer marker; busy time only)"
        )
        fig = go.Figure(
            go.Sunburst(
                labels=labels,
                parents=parents,
                values=values,
                branchvalues="total",
                text=text,
                hovertemplate="<b>%{label}</b><br>%{text}<extra></extra>",
                insidetextorientation="radial",
            )
        )
        fig.update_layout(
            title=f"HSTU Layer Step Breakdown — {label}<br>"
            f"<sup>median step; total = {total:.2f} ms {title_suffix}</sup>",
            margin=dict(t=80, l=10, r=10, b=10),
            width=900,
            height=900,
        )

        if output_html:
            fig.write_html(output_html)
            print(f"\nWrote {output_html}")

        if output_png:
            try:
                fig.write_image(output_png, scale=2)
                print(f"Wrote {output_png}")
            except Exception as e:
                print(
                    f"  (plotly PNG export unavailable: {e.__class__.__name__}); "
                    "falling back to matplotlib nested pie.",
                    file=sys.stderr,
                )
                _matplotlib_nested_pie(
                    output_png,
                    fwd_med,
                    bwd_med,
                    total,
                    label,
                    n_iters=n_iters,
                    idle_ms=idle_ms if has_idle else 0.0,
                )
    except ImportError:
        print(
            "plotly not installed — skipping interactive sunburst HTML. "
            "Falling back to matplotlib nested pie for PNG.",
            file=sys.stderr,
        )
        if output_png:
            _matplotlib_nested_pie(
                output_png, fwd_med, bwd_med, total, label, n_iters=n_iters
            )


def _matplotlib_nested_pie(
    output_png: str,
    fwd_med: List[float],
    bwd_med: List[float],
    total: float,
    label: str,
    n_iters: int,
    idle_ms: float = 0.0,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fwd_sum = sum(fwd_med)
    bwd_sum = sum(bwd_med)

    fig, ax = plt.subplots(figsize=(11, 11), subplot_kw=dict(aspect="equal"))

    # Inner ring: fwd / bwd / idle
    if idle_ms > 0:
        inner_sizes = [fwd_sum, bwd_sum, idle_ms]
        inner_labels = [
            f"fwd\n{fwd_sum:.2f} ms\n{fwd_sum / total * 100:.1f}%",
            f"bwd\n{bwd_sum:.2f} ms\n{bwd_sum / total * 100:.1f}%",
            f"idle\n{idle_ms:.2f} ms\n{idle_ms / total * 100:.1f}%",
        ]
        inner_colors = ["#4C72B0", "#DD8452", "#888888"]
    else:
        inner_sizes = [fwd_sum, bwd_sum]
        inner_labels = [
            f"fwd\n{fwd_sum:.2f} ms\n{fwd_sum / total * 100:.1f}%",
            f"bwd\n{bwd_sum:.2f} ms\n{bwd_sum / total * 100:.1f}%",
        ]
        inner_colors = ["#4C72B0", "#DD8452"]

    ax.pie(
        inner_sizes,
        radius=0.6,
        labels=inner_labels,
        labeldistance=0.45,
        colors=inner_colors,
        wedgeprops=dict(width=0.6, edgecolor="white"),
        textprops=dict(
            ha="center", va="center", fontsize=12, fontweight="bold", color="white"
        ),
        startangle=90,
        counterclock=False,
    )

    # Outer ring: 4 fwd phases + 4 bwd phases + optional blank idle ring
    outer_sizes = fwd_med + bwd_med
    outer_names = [
        "ln+linear_bias+silu",
        "attn",
        "norm mul dropout",
        "linear_residual",
        "linear_residual",
        "norm_mul_dropout",
        "attn",
        "ln_linear_silu",
    ]
    cmap_fwd = plt.cm.Blues(range(80, 240, 40))
    cmap_bwd = plt.cm.Oranges(range(80, 240, 40))
    outer_colors = list(cmap_fwd) + list(cmap_bwd)
    outer_labels = [
        f"{n}\n{v:.2f} ms ({v / total * 100:.1f}%)"
        for n, v in zip(outer_names, outer_sizes)
    ]
    if idle_ms > 0:
        outer_sizes = outer_sizes + [idle_ms]
        outer_names = outer_names + [""]
        outer_colors = outer_colors + ["#BBBBBB"]
        outer_labels = outer_labels + [""]
    ax.pie(
        outer_sizes,
        radius=1.0,
        labels=outer_labels,
        labeldistance=1.08,
        colors=outer_colors,
        wedgeprops=dict(width=0.4, edgecolor="white"),
        textprops=dict(ha="center", va="center", fontsize=9),
        startangle=90,
        counterclock=False,
    )

    subtitle = (
        f"GPU wall-clock step = {total:.2f} ms;  idle = {idle_ms:.2f} ms"
        if idle_ms > 0
        else f"fwd+bwd busy time = {total:.2f} ms (no outer marker → no idle)"
    )
    ax.set_title(
        f"HSTU Layer Step Breakdown — {label}\n{subtitle}",
        fontsize=13,
        fontweight="bold",
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(output_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_png} (matplotlib fallback)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite_path")
    ap.add_argument("--output", default="sunburst.html")
    ap.add_argument(
        "--png", default=None, help="Also write a static PNG (needs kaleido)"
    )
    ap.add_argument("--label", default="plus_fused (A100 80GB PCIe, bf16)")
    args = ap.parse_args()
    build(args.sqlite_path, args.output, args.png, args.label)
    return 0


if __name__ == "__main__":
    sys.exit(main())
