#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HSTU Benchmark Results Analyzer

Extract FLOPS and MFU metrics from experiment logs and visualize comparisons.

Usage:
    python analyze_results.py <results_dir> [options]

Examples:
    # Analyze all experiments in a batch result directory
    python analyze_results.py training/benchmark/results/20260205_072123
    
    # Save plot to file instead of displaying
    python analyze_results.py training/benchmark/results/20260205_072123 --output comparison.png
    
    # Use bar chart (default)
    python analyze_results.py training/benchmark/results/20260205_072123 --plot-type bar
    
    # Use line chart
    python analyze_results.py training/benchmark/results/20260205_072123 --plot-type line
    
    # Skip first N iterations (warmup)
    python analyze_results.py training/benchmark/results/20260205_072123 --skip-warmup 1
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Try to import matplotlib, provide helpful message if not available
try:
    import matplotlib

    # IMPORTANT: Must set backend BEFORE importing pyplot
    # Use non-interactive backend for headless servers (no display)
    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def extract_metrics_from_log(
    log_path: str, skip_warmup: int = 0
) -> Tuple[Optional[float], Optional[float]]:
    """
    Extract max FLOPS and MFU from a log file.

    Args:
        log_path: Path to log file
        skip_warmup: Number of initial iterations to skip (warmup)

    Returns:
        Tuple of (max_flops, max_mfu) or (None, None) if not found
    """
    # Regex patterns
    # Pattern: "achieved FLOPS 67.81 TFLOPS, MFU 10.87%"
    flops_pattern = r"achieved FLOPS\s+([\d.]+)\s+TFLOPS"
    mfu_pattern = r"MFU\s+([\d.]+)%"

    flops_values = []
    mfu_values = []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                flops_match = re.search(flops_pattern, line)
                mfu_match = re.search(mfu_pattern, line)

                if flops_match and mfu_match:
                    flops_values.append(float(flops_match.group(1)))
                    mfu_values.append(float(mfu_match.group(1)))
    except Exception as e:
        print(f"Warning: Error reading {log_path}: {e}")
        return None, None

    if not flops_values:
        return None, None

    # Skip warmup iterations
    if skip_warmup > 0 and len(flops_values) > skip_warmup:
        flops_values = flops_values[skip_warmup:]
        mfu_values = mfu_values[skip_warmup:]

    return max(flops_values), max(mfu_values)


def find_log_files(results_dir: str) -> Dict[str, str]:
    """
    Find all experiment log files in the results directory.

    Args:
        results_dir: Path to results directory

    Returns:
        Dict mapping experiment names to log file paths
    """
    experiments: Dict[str, str] = {}
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"Error: Directory not found: {results_dir}")
        return experiments

    # Look for experiment subdirectories
    for exp_dir in sorted(results_path.iterdir()):
        if not exp_dir.is_dir():
            continue

        # Skip summary.txt and other non-directory items
        if exp_dir.name in ["summary.txt"]:
            continue

        # Find .log files in the experiment directory
        log_files = list(exp_dir.glob("*.log"))
        if log_files:
            # Use the first (or only) log file
            experiments[exp_dir.name] = str(log_files[0])

    return experiments


def analyze_experiments(
    results_dir: str, skip_warmup: int = 0
) -> Dict[str, Dict[str, Any]]:
    """
    Analyze all experiments in the results directory.

    Args:
        results_dir: Path to results directory
        skip_warmup: Number of warmup iterations to skip

    Returns:
        Dict mapping experiment names to their metrics
    """
    experiments = find_log_files(results_dir)

    if not experiments:
        print(f"No experiments found in {results_dir}")
        return {}

    results = {}
    print(f"\nAnalyzing {len(experiments)} experiments...")
    print("-" * 60)

    for exp_name, log_path in experiments.items():
        max_flops, max_mfu = extract_metrics_from_log(log_path, skip_warmup)

        if max_flops is not None and max_mfu is not None:
            results[exp_name] = {
                "max_flops": max_flops,
                "max_mfu": max_mfu,
                "log_path": log_path,
            }
            print(
                f"  {exp_name:30s} | FLOPS: {max_flops:7.2f} TFLOPS | MFU: {max_mfu:6.2f}%"
            )
        else:
            print(f"  {exp_name:30s} | No metrics found")

    print("-" * 60)
    return results


def plot_comparison(
    results: Dict[str, Dict[str, Any]],
    output_path: Optional[str] = None,
    plot_type: str = "bar",
    title: str = "HSTU Benchmark Comparison",
) -> None:
    """
    Plot comparison of FLOPS and MFU across experiments.

    Args:
        results: Dict of experiment results
        output_path: Path to save the plot (if None, display interactively)
        plot_type: 'bar' or 'line'
        title: Plot title
    """
    if not results:
        print("No results to plot")
        return

    # Sort experiments by name for consistent ordering
    exp_names = sorted(results.keys())
    flops_values = [results[exp]["max_flops"] for exp in exp_names]
    mfu_values = [results[exp]["max_mfu"] for exp in exp_names]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Color palette
    colors = plt.cm.Set2(range(len(exp_names)))

    if plot_type == "bar":
        # Bar chart
        x = range(len(exp_names))

        # FLOPS subplot
        bars1 = ax1.bar(x, flops_values, color=colors, edgecolor="black", linewidth=0.5)
        ax1.set_xlabel("Experiment", fontsize=12)
        ax1.set_ylabel("Max FLOPS (TFLOPS)", fontsize=12)
        ax1.set_title("Maximum Achieved FLOPS", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=10)
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars1, flops_values):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        # MFU subplot
        bars2 = ax2.bar(x, mfu_values, color=colors, edgecolor="black", linewidth=0.5)
        ax2.set_xlabel("Experiment", fontsize=12)
        ax2.set_ylabel("Max MFU (%)", fontsize=12)
        ax2.set_title("Maximum Model FLOPs Utilization", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=10)
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars2, mfu_values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{val:.2f}%",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    else:  # line plot
        x = range(len(exp_names))

        # FLOPS subplot
        ax1.plot(x, flops_values, "o-", color="#2196F3", linewidth=2, markersize=8)
        ax1.set_xlabel("Experiment", fontsize=12)
        ax1.set_ylabel("Max FLOPS (TFLOPS)", fontsize=12)
        ax1.set_title("Maximum Achieved FLOPS", fontsize=14, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=10)
        ax1.grid(alpha=0.3)

        # Add value labels
        for i, val in enumerate(flops_values):
            ax1.annotate(
                f"{val:.1f}",
                (i, val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

        # MFU subplot
        ax2.plot(x, mfu_values, "s-", color="#4CAF50", linewidth=2, markersize=8)
        ax2.set_xlabel("Experiment", fontsize=12)
        ax2.set_ylabel("Max MFU (%)", fontsize=12)
        ax2.set_title("Maximum Model FLOPs Utilization", fontsize=14, fontweight="bold")
        ax2.set_xticks(x)
        ax2.set_xticklabels(exp_names, rotation=45, ha="right", fontsize=10)
        ax2.grid(alpha=0.3)

        # Add value labels
        for i, val in enumerate(mfu_values):
            ax2.annotate(
                f"{val:.2f}%",
                (i, val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=9,
            )

    # Overall title
    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {output_path}")
    else:
        plt.show()


def print_summary_table(results: Dict[str, Dict[str, Any]]) -> None:
    """Print a summary table of results."""
    if not results:
        return

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Experiment':<30} | {'Max FLOPS (TFLOPS)':>18} | {'Max MFU (%)':>12}")
    print("-" * 70)

    # Sort by FLOPS descending
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["max_flops"], reverse=True
    )

    best_flops = sorted_results[0][1]["max_flops"]
    best_mfu = max(r["max_mfu"] for r in results.values())

    for exp_name, metrics in sorted_results:
        flops_marker = " 🏆" if metrics["max_flops"] == best_flops else ""
        mfu_marker = " 🏆" if metrics["max_mfu"] == best_mfu else ""
        print(
            f"{exp_name:<30} | {metrics['max_flops']:>15.2f}{flops_marker:>3} | {metrics['max_mfu']:>9.2f}%{mfu_marker}"
        )

    print("=" * 70)

    # Calculate speedup relative to first experiment
    if len(sorted_results) > 1:
        baseline_name = sorted_results[-1][0]  # Worst performer as baseline
        baseline_flops = sorted_results[-1][1]["max_flops"]

        print(f"\nSpeedup relative to {baseline_name}:")
        for exp_name, metrics in sorted_results:
            if exp_name != baseline_name:
                speedup = metrics["max_flops"] / baseline_flops
                print(f"  {exp_name}: {speedup:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze HSTU benchmark results and visualize comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "results_dir",
        help="Path to results directory (e.g., training/benchmark/results/20260205_072123)",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output file path for the plot (e.g., comparison.png). If not specified, display interactively.",
    )
    parser.add_argument(
        "--plot-type",
        choices=["bar", "line"],
        default="bar",
        help="Type of plot: bar or line (default: bar)",
    )
    parser.add_argument(
        "--skip-warmup",
        type=int,
        default=1,
        help="Number of warmup iterations to skip (default: 1)",
    )
    parser.add_argument(
        "--title", default="HSTU Benchmark Comparison", help="Plot title"
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only print summary, do not generate plot",
    )

    args = parser.parse_args()

    # Analyze experiments
    results = analyze_experiments(args.results_dir, args.skip_warmup)

    if not results:
        print("No valid results found. Please check the log files.")
        sys.exit(1)

    # Print summary table
    print_summary_table(results)

    # Generate plot
    if not args.no_plot:
        plot_comparison(results, args.output, args.plot_type, args.title)


if __name__ == "__main__":
    main()
