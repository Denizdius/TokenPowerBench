#!/usr/bin/env python3
"""Generate paper-quality TokenPowerBench figures (min/max-excluded averages)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_figures"

CAMPAIGNS = {
    "alpaca": {
        "analysis_csv": ROOT
        / "tokenbench_qwen3_14b_27b_noeager_multi_run_result"
        / "analysis"
        / "csv",
        "dataset_label": "Alpaca",
        "models": ["Qwen3-14B", "Qwen3.5-27B"],
        "filename_infix": "",
    },
    "longbench": {
        "analysis_csv": ROOT
        / "tokenbench_qwen3_14b_longbench_noeager_multi_run_result"
        / "analysis"
        / "csv",
        "dataset_label": "LongBench",
        "models": ["Qwen3-14B"],
        "filename_infix": "longbench_",
    },
}

# noeager = torch.compile / CUDA graphs enabled; eager = enforce-eager
COLOR_COMPILE_ON = "#2A9D8F"
COLOR_COMPILE_OFF = "#E76F51"
DURATION_DIFF_THRESHOLD_PCT = 10.0

CONFIG_ORDER = [
    "Single GPU",
    "TP2",
    "PP2",
    "DP2",
    "TP2 × DP2",
    "PP2 × DP2",
    "TP2 × PP2",
    "TP4",
    "PP4",
    "DP4",
]

WORKLOADS = [
    ("bs128_out500", "Batch Size 128, Output Tokens Limit 500"),
    ("bs256_out500", "Batch Size 256, Output Tokens Limit 500"),
    ("bs256_out2000", "Batch Size 256, Output Tokens Limit 2000"),
]

PALETTE = [
    "#4C72B0",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#CCB974",
    "#64B5CD",
    "#8C8C8C",
    "#E17C05",
    "#2CA02C",
    "#D62728",
]


def setup_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times", "serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sort_configs(configs: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(CONFIG_ORDER)}
    return sorted(configs, key=lambda c: order.get(c, 999))


def model_slug(name: str) -> str:
    return name.lower().replace(".", "").replace("-", "_")


def plot_figure(
    df: pd.DataFrame,
    model: str,
    workload: str,
    workload_title: str,
    dataset_label: str,
    out_stem: Path,
) -> None:
    sub = df[(df["model_name"] == model) & (df["workload"] == workload)].copy()
    if sub.empty:
        print(f"No data for {model} {workload}")
        return

    configs = sort_configs(sub["config"].tolist())
    by_cfg = {r["config"]: r for _, r in sub.iterrows()}

    latency = [by_cfg[c]["duration"] for c in configs]
    throughput = [by_cfg[c]["throughput"] for c in configs]
    power = [by_cfg[c]["gpu_avg_power"] for c in configs]
    mj = [by_cfg[c]["mj_per_token"] for c in configs]

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(configs))]
    x = np.arange(len(configs))

    width = 10.5 if len(configs) <= 4 else 12.5
    fig, axes = plt.subplots(2, 2, figsize=(width, 7.2))
    fig.suptitle(
        f"{model} — {dataset_label}\n{workload_title}",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )

    panels = [
        (axes[0, 0], latency, "Latency (s)", "{:.1f}"),
        (axes[0, 1], throughput, "Throughput (tok/s)", "{:.0f}"),
        (axes[1, 0], power, "Average Power of GPUs (W)", "{:.0f}"),
        (axes[1, 1], mj, "Energy per Token (mJ/token)", "{:.1f}"),
    ]

    for ax, values, ylabel, fmt in panels:
        bars = ax.bar(
            x,
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.4,
            width=0.72,
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right")
        ax.set_xlim(-0.6, len(configs) - 0.4)
        ymax = max(values) if values else 1.0
        ax.set_ylim(0, ymax * 1.18)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + ymax * 0.015,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    for ext in ("pdf", "png"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def plot_torch_compile_figure(
    df: pd.DataFrame,
    model: str,
    workload: str,
    workload_title: str,
    dataset_label: str,
    out_stem: Path,
) -> None:
    """Grouped On/Off bars for configs with at least 10% duration difference."""
    sub = df[(df["model_name"] == model) & (df["workload"] == workload)].copy()
    if sub.empty:
        print(f"No torch-compile diffs for {model} {workload}")
        return

    configs = sort_configs(sub["config"].tolist())
    by_cfg = {r["config"]: r for _, r in sub.iterrows()}

    # On = noeager, Off = eager
    metrics = [
        (
            "Latency (s)",
            "{:.1f}",
            [by_cfg[c]["duration_noeager"] for c in configs],
            [by_cfg[c]["duration_eager"] for c in configs],
        ),
        (
            "Throughput (tok/s)",
            "{:.0f}",
            [by_cfg[c]["throughput_noeager"] for c in configs],
            [by_cfg[c]["throughput_eager"] for c in configs],
        ),
        (
            "Average Power of GPUs (W)",
            "{:.0f}",
            [by_cfg[c]["gpu_avg_power_noeager"] for c in configs],
            [by_cfg[c]["gpu_avg_power_eager"] for c in configs],
        ),
        (
            "Energy per Token (mJ/token)",
            "{:.1f}",
            [by_cfg[c]["mj_per_token_noeager"] for c in configs],
            [by_cfg[c]["mj_per_token_eager"] for c in configs],
        ),
    ]

    n = len(configs)
    width = 9.5 if n <= 2 else (11.5 if n <= 4 else 13.0)
    fig, axes = plt.subplots(2, 2, figsize=(width, 7.4))
    fig.suptitle(
        f"{model} — {dataset_label}\n{workload_title}\nTorch Compile On vs Off",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    x = np.arange(n)
    bar_w = 0.36

    for ax, (ylabel, fmt, on_vals, off_vals) in zip(axes.flat, metrics):
        bars_on = ax.bar(
            x - bar_w / 2,
            on_vals,
            width=bar_w,
            color=COLOR_COMPILE_ON,
            edgecolor="black",
            linewidth=0.4,
            label="Torch Compile On",
        )
        bars_off = ax.bar(
            x + bar_w / 2,
            off_vals,
            width=bar_w,
            color=COLOR_COMPILE_OFF,
            edgecolor="black",
            linewidth=0.4,
            label="Torch Compile Off",
        )
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=30, ha="right")
        ax.set_xlim(-0.7, n - 0.3)
        ymax = max(max(on_vals), max(off_vals)) if on_vals else 1.0
        ax.set_ylim(0, ymax * 1.22)
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.55)
        ax.set_axisbelow(True)
        for bars, vals in ((bars_on, on_vals), (bars_off, off_vals)):
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ymax * 0.015,
                    fmt.format(val),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="bold",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.915),
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    for ext in ("pdf", "png"):
        path = out_stem.with_suffix(f".{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Wrote {path}")
    plt.close(fig)


def load_torch_compile_diffs(
    compare_csv: Path,
    threshold_pct: float = DURATION_DIFF_THRESHOLD_PCT,
) -> pd.DataFrame:
    df = pd.read_csv(compare_csv)
    pct = (df["duration_noeager"] - df["duration_eager"]).abs() / df["duration_eager"] * 100.0
    return df[pct >= threshold_pct].copy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TokenPowerBench paper figures")
    parser.add_argument(
        "--campaign",
        choices=CAMPAIGNS,
        default="alpaca",
        help="Dataset campaign to render (default: alpaca)",
    )
    parser.add_argument(
        "--all-comparisons",
        action="store_true",
        help="Render Torch Compile comparisons for every configuration",
    )
    args = parser.parse_args()

    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    campaign = CAMPAIGNS[args.campaign]
    analysis_csv = campaign["analysis_csv"]
    dataset_label = campaign["dataset_label"]
    filename_infix = campaign["filename_infix"]
    df = pd.read_csv(analysis_csv / "summary_min_max_excluded.csv")

    for model in campaign["models"]:
        for workload, title in WORKLOADS:
            stem = OUT / f"{model_slug(model)}_{filename_infix}{workload}"
            plot_figure(df, model, workload, title, dataset_label, stem)

    threshold_pct = 0.0 if args.all_comparisons else DURATION_DIFF_THRESHOLD_PCT
    cmp_df = load_torch_compile_diffs(
        analysis_csv / "eager_vs_noeager_absolute.csv",
        threshold_pct=threshold_pct,
    )
    print(
        f"\nTorch Compile diffs with duration |Δ| >= {threshold_pct}%: "
        f"{len(cmp_df)} configs"
    )
    print(
        cmp_df[["model_name", "workload", "config"]].to_string(index=False)
    )
    workload_titles = dict(WORKLOADS)
    for (model, workload), group in cmp_df.groupby(["model_name", "workload"], sort=False):
        title = workload_titles.get(workload, workload)
        stem = OUT / f"{model_slug(model)}_{filename_infix}{workload}_torch_compile"
        plot_torch_compile_figure(group, model, workload, title, dataset_label, stem)


if __name__ == "__main__":
    main()

