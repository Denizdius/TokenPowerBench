#!/usr/bin/env python3
"""Plot TokenPowerBench results: latency, throughput, mJ/token, avg power."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = [
    ("bs128_out500", 128, 500, "Batch 128 · max 500 output tokens"),
    ("bs256_out500", 256, 500, "Batch 256 · max 500 output tokens"),
    ("bs256_out2000", 256, 2000, "Batch 256 · max 2000 output tokens"),
]

MODEL_DIRS = {
    "Qwen3-14B": ROOT / "tokenbench_qwen3_14b",
    "Qwen3.5-27B": ROOT / "tokenbench_qwen3.5_27b",
}

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


def config_label(record: dict) -> str:
    tp = record["tensor_parallel_size"]
    pp = record["pipeline_parallel_size"]
    dp = record["data_parallel_size"]
    parts: list[str] = []
    if tp > 1:
        parts.append(f"TP{tp}")
    if pp > 1:
        parts.append(f"PP{pp}")
    if dp > 1:
        parts.append(f"DP{dp}")
    return " × ".join(parts) if parts else "Single GPU"


def file_timestamp(path: Path) -> int:
    m = re.search(r"_(\d{8}_\d{6})\.json$", path.name)
    if not m:
        return 0
    return int(m.group(1).replace("_", ""))


def load_results(result_dir: Path) -> list[dict]:
    best: dict[tuple, dict] = {}
    for path in result_dir.glob("*.json"):
        with path.open() as fh:
            payload = json.load(fh)
        for key, record in payload.items():
            if not key.startswith("batch_"):
                continue
            row = dict(record)
            row["_source"] = path.name
            row["config"] = config_label(row)
            row["throughput_tok_s"] = (
                row["total_output_tokens"] / row["duration_s"]
                if row["duration_s"]
                else 0.0
            )
            row["mj_per_token"] = (
                row.get("total_mj_per_token") or row.get("gpu_mj_per_token") or 0.0
            )
            dedupe_key = (
                row["batch_size"],
                row["output_tokens"],
                row["config"],
            )
            ts = file_timestamp(path)
            if dedupe_key not in best or ts >= best[dedupe_key]["_ts"]:
                row["_ts"] = ts
                best[dedupe_key] = row
    return list(best.values())


def sort_configs(configs: list[str]) -> list[str]:
    order = {name: i for i, name in enumerate(CONFIG_ORDER)}
    return sorted(configs, key=lambda c: order.get(c, 999))


def plot_scenario(model_name: str, rows: list[dict], batch_size: int, output_tokens: int,
                  title_suffix: str, out_path: Path) -> None:
    subset = [
        r for r in rows
        if r["batch_size"] == batch_size and r["output_tokens"] == output_tokens
    ]
    if not subset:
        print(f"No data for {model_name} {title_suffix}")
        return

    configs = sort_configs([r["config"] for r in subset])
    by_cfg = {r["config"]: r for r in subset}

    latency = [by_cfg[c]["duration_s"] for c in configs]
    throughput = [by_cfg[c]["throughput_tok_s"] for c in configs]
    mj_per_token = [by_cfg[c]["mj_per_token"] for c in configs]
    avg_power = [by_cfg[c]["gpu_avg_power_w"] for c in configs]

    x = np.arange(len(configs))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"{model_name} — {title_suffix}", fontsize=14, fontweight="bold")

    panels = [
        (axes[0, 0], latency, "Latency (s)", "#4C72B0", "Total benchmark duration"),
        (axes[0, 1], throughput, "Throughput (tok/s)", "#55A868", "Estimated output tokens / s"),
        (axes[1, 0], mj_per_token, "mJ per token", "#C44E52", "GPU energy per output token"),
        (axes[1, 1], avg_power, "Avg GPU power (W)", "#8172B3", "NVML average over active GPUs"),
    ]

    for ax, values, ylabel, color, subtitle in panels:
        bars = ax.bar(x, values, color=color, edgecolor="white", linewidth=0.6)
        ax.set_title(subtitle, fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    for model_name, result_dir in MODEL_DIRS.items():
        if not result_dir.exists():
            print(f"Skip missing dir: {result_dir}")
            continue
        rows = load_results(result_dir)
        plots_dir = result_dir / "plots"
        for slug, batch_size, output_tokens, title in SCENARIOS:
            out_file = plots_dir / f"{model_name.replace('.', '')}_{slug}.png"
            plot_scenario(model_name, rows, batch_size, output_tokens, title, out_file)


if __name__ == "__main__":
    main()
