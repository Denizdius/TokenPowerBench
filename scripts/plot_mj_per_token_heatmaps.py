#!/usr/bin/env python3
"""Paper-style mJ/token heatmaps for eager and no-eager multi-run results.

Uses average-min/max-excluded values. Separate figures for:
  - eager vs no-eager
  - Single GPU / 2 GPUs / 4 GPUs

Within each figure, TP and PP configs are adjacent for easy comparison.
Layout mirrors TokenPowerBench Figure 5: three workload panels side by side.

Depends only on numpy + matplotlib (no pandas) for faster startup.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parents[1]

WORKLOADS = [
    ("bs128_out500", "Standard Load\n(BS: 128, T: 500)"),
    ("bs256_out500", "High Concurrency\n(BS: 256, T: 500)"),
    ("bs256_out2000", "High Throughput\n(BS: 256, T: 2000)"),
]

MODEL_ORDER = ["Qwen3-14B", "Qwen3.5-27B"]

# TP and PP kept adjacent within each GPU-count group.
GPU_GROUPS = {
    "1gpu": {
        "title": "Single GPU",
        "configs": ["Single GPU"],
        "labels": ["Single GPU"],
    },
    "2gpu": {
        "title": "2 GPUs",
        "configs": ["TP2", "PP2", "DP2"],
        "labels": ["TP=2", "PP=2", "DP=2"],
    },
    "4gpu": {
        "title": "4 GPUs",
        "configs": ["TP4", "PP4", "TP2 × PP2", "TP2 × DP2", "PP2 × DP2", "DP4"],
        "labels": ["TP=4", "PP=4", "TP=2×PP=2", "TP=2×DP=2", "PP=2×DP=2", "DP=4"],
    },
}

DATASETS = {
    "eager": {
        "csv": ROOT
        / "tokenbench_qwen3_14b_27b_multi_run_result"
        / "analysis"
        / "csv"
        / "summary_min_max_excluded.csv",
        "out_dir": ROOT
        / "tokenbench_qwen3_14b_27b_multi_run_result"
        / "analysis"
        / "plots"
        / "heatmaps",
        "mode_label": "Eager",
    },
    "noeager": {
        "csv": ROOT
        / "tokenbench_qwen3_14b_27b_noeager_multi_run_result"
        / "analysis"
        / "csv"
        / "summary_min_max_excluded.csv",
        "out_dir": ROOT
        / "tokenbench_qwen3_14b_27b_noeager_multi_run_result"
        / "analysis"
        / "plots"
        / "heatmaps",
        "mode_label": "No-Eager",
    },
}


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def models_with_data(rows: list[dict[str, str]], configs: list[str]) -> list[str]:
    """Keep only models that have at least one value for this GPU group."""
    present = {
        r["model_name"]
        for r in rows
        if r["config"] in configs and r["model_name"] in MODEL_ORDER
    }
    return [m for m in MODEL_ORDER if m in present]


def build_matrix(
    rows: list[dict[str, str]],
    workload: str,
    configs: list[str],
    models: list[str],
) -> np.ndarray:
    """Rows = models, cols = configs; NaN where missing."""
    mat = np.full((len(models), len(configs)), np.nan)
    lookup = {
        (r["model_name"], r["workload"], r["config"]): float(r["mj_per_token"])
        for r in rows
    }
    for i, model in enumerate(models):
        for j, cfg in enumerate(configs):
            key = (model, workload, cfg)
            if key in lookup:
                mat[i, j] = lookup[key]
    return mat


def annotate_cells(ax, mat: np.ndarray, norm: Normalize, fontsize: int = 11) -> None:
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            if np.isnan(val):
                ax.text(
                    j, i, "—", ha="center", va="center", color="#666666", fontsize=fontsize
                )
                continue
            rgba = plt.cm.RdYlGn_r(norm(val))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if luminance < 0.45 else "black"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
            )


def plot_gpu_group(
    rows: list[dict[str, str]],
    *,
    mode_key: str,
    mode_label: str,
    group_key: str,
    out_dir: Path,
) -> Path | None:
    group = GPU_GROUPS[group_key]
    configs = group["configs"]
    labels = group["labels"]

    models = models_with_data(rows, configs)
    if not models:
        print(f"  skip {mode_key}/{group_key}: no models with data", flush=True)
        return None

    matrices = [build_matrix(rows, wl, configs, models) for wl, _ in WORKLOADS]
    finite_parts = [m[~np.isnan(m)] for m in matrices if np.any(~np.isnan(m))]
    if not finite_parts:
        print(f"  skip {mode_key}/{group_key}: all NaN", flush=True)
        return None
    finite = np.concatenate(finite_parts)

    vmin, vmax = float(finite.min()), float(finite.max())
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    norm = Normalize(vmin=vmin, vmax=vmax)

    n_cols = len(WORKLOADS)
    n_models = len(models)
    # Single-GPU gets a bit more vertical room so the one available row reads clearly.
    if group_key == "1gpu":
        fig_w = 9.5
        fig_h = 3.4
        cell_fontsize = 13
        label_fontsize = 11
    else:
        fig_w = max(3.0 * n_cols + 0.8, 1.1 * len(configs) + 5.5)
        fig_h = max(2.6, 0.95 * n_models + 2.0)
        cell_fontsize = 11
        label_fontsize = 10

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w, fig_h), squeeze=False)
    axes = axes[0]

    for ax, mat, (_, title) in zip(axes, matrices, WORKLOADS):
        ax.imshow(mat, cmap="RdYlGn_r", norm=norm, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(models, fontsize=label_fontsize)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_models, 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
        annotate_cells(ax, mat, norm, fontsize=cell_fontsize)

    for ax in axes[1:]:
        ax.set_yticklabels([])

    fig.suptitle(
        f"Heatmap of Energy per Token — {mode_label} — {group['title']}\n"
        f"(average, min/max run excluded)",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{mode_key}_{group_key}_mj_per_token.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}", flush=True)
    return out_path


def main() -> None:
    written: list[Path] = []
    for mode_key, meta in DATASETS.items():
        csv_path = meta["csv"]
        if not csv_path.exists():
            print(f"Missing {csv_path}, skipping {mode_key}", flush=True)
            continue
        rows = load_rows(csv_path)
        print(f"\n{meta['mode_label']} ({len(rows)} rows from {csv_path.name})", flush=True)
        for group_key in GPU_GROUPS:
            path = plot_gpu_group(
                rows,
                mode_key=mode_key,
                mode_label=meta["mode_label"],
                group_key=group_key,
                out_dir=meta["out_dir"],
            )
            if path:
                written.append(path)

    print(f"\nDone. Generated {len(written)} heatmap figures.", flush=True)


if __name__ == "__main__":
    main()
