#!/usr/bin/env python3
"""
Analyze TokenPowerBench multi-run benchmark results.

For each of 10 runs, 2 models, and 3 workloads (normal / high concurrency / high
throughput), produces bar-chart plots and CSV summaries. Aggregates across runs
with mean, average min max excluded, percentiles, min–max spread, and
ANOVA / factorial statistics.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

SCENARIOS = [
    ("bs128_out500", 128, 500, "Normal (batch 128 · 500 output tokens)"),
    ("bs256_out500", 256, 500, "High concurrency (batch 256 · 500 output tokens)"),
    ("bs256_out2000", 256, 2000, "High throughput (batch 256 · 2000 output tokens)"),
]

MODELS = {
    "qwen3_14b": "Qwen3-14B",
    "qwen3.5_27b": "Qwen3.5-27B",
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

METRICS = [
    ("duration_s", "Duration (s)", "#4C72B0"),
    ("throughput_tok_s", "Throughput (tok/s)", "#55A868"),
    ("mj_per_token", "Energy per token (mJ)", "#C44E52"),
    ("gpu_avg_power_w", "Avg GPU power (W)", "#8172B3"),
]

JSON_NAME_RE = re.compile(
    r"^(qwen3(?:\.5_27b|_14b))_(.+)_bs(\d+)_out(\d+)_(\d{8}_\d{6})\.json$"
)


@dataclass(frozen=True)
class RunRecord:
    run_id: int
    model_key: str
    model_name: str
    strategy: str
    config: str
    batch_size: int
    output_tokens: int
    workload: str
    duration_s: float
    throughput_tok_s: float
    mj_per_token: float
    gpu_avg_power_w: float
    source_file: str


def config_label(tp: int, pp: int, dp: int) -> str:
    parts: list[str] = []
    if tp > 1:
        parts.append(f"TP{tp}")
    if pp > 1:
        parts.append(f"PP{pp}")
    if dp > 1:
        parts.append(f"DP{dp}")
    return " × ".join(parts) if parts else "Single GPU"


def workload_slug(batch_size: int, output_tokens: int) -> str:
    return f"bs{batch_size}_out{output_tokens}"


def sort_configs(configs: Iterable[str]) -> list[str]:
    order = {name: i for i, name in enumerate(CONFIG_ORDER)}
    return sorted(set(configs), key=lambda c: order.get(c, 999))


def parse_json_record(path: Path, run_id: int) -> RunRecord | None:
    m = JSON_NAME_RE.match(path.name)
    if not m:
        return None
    model_key, strategy, batch_s, out_s, _ts = m.groups()
    if model_key not in MODELS:
        return None

    with path.open() as fh:
        payload = json.load(fh)

    record = None
    for key, row in payload.items():
        if key.startswith("batch_"):
            record = row
            break
    if record is None:
        return None

    duration = float(record["duration_s"])
    total_tokens = float(record["total_output_tokens"])
    batch_size = int(record["batch_size"])
    output_tokens = int(record["output_tokens"])

    return RunRecord(
        run_id=run_id,
        model_key=model_key,
        model_name=MODELS[model_key],
        strategy=strategy,
        config=config_label(
            int(record["tensor_parallel_size"]),
            int(record["pipeline_parallel_size"]),
            int(record["data_parallel_size"]),
        ),
        batch_size=batch_size,
        output_tokens=output_tokens,
        workload=workload_slug(batch_size, output_tokens),
        duration_s=duration,
        throughput_tok_s=total_tokens / duration if duration else 0.0,
        mj_per_token=float(
            record.get("total_mj_per_token") or record.get("gpu_mj_per_token") or 0.0
        ),
        gpu_avg_power_w=float(record["gpu_avg_power_w"]),
        source_file=path.name,
    )


def load_all_records(data_root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for run_dir in sorted(data_root.glob("run_*")):
        if not run_dir.is_dir():
            continue
        run_id = int(run_dir.name.split("_", 1)[1])
        json_dir = run_dir / "jsons"
        if not json_dir.is_dir():
            continue
        for path in sorted(json_dir.glob("*.json")):
            rec = parse_json_record(path, run_id)
            if rec:
                rows.append(rec.__dict__)
    if not rows:
        raise FileNotFoundError(f"No JSON results under {data_root}/run_*/jsons/")
    return pd.DataFrame(rows)


def average_min_max_excluded(values: np.ndarray) -> float:
    """Mean after dropping one minimum and one maximum (needs >= 3 samples)."""
    arr = np.asarray(values, dtype=float)
    if arr.size <= 2:
        return float(np.mean(arr))
    sorted_vals = np.sort(arr)
    return float(np.mean(sorted_vals[1:-1]))


def summarize_group(g: pd.DataFrame, metric: str) -> dict:
    vals = g[metric].to_numpy(dtype=float)
    return {
        "n_runs": len(vals),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "sem": float(stats.sem(vals)) if len(vals) > 1 else 0.0,
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "range": float(np.max(vals) - np.min(vals)),
        "average_min_max_excluded": average_min_max_excluded(vals),
        "p25": float(np.percentile(vals, 25)),
        "p50": float(np.percentile(vals, 50)),
        "p75": float(np.percentile(vals, 75)),
        "p95": float(np.percentile(vals, 95)),
        "p99": float(np.percentile(vals, 99)),
        "cv_pct": float(100.0 * np.std(vals, ddof=1) / np.mean(vals))
        if len(vals) > 1 and np.mean(vals) != 0
        else 0.0,
    }


def plot_metric_bars(
    *,
    configs: list[str],
    values: list[float],
    yerr: list[float] | None,
    title: str,
    ylabel: str,
    color: str,
    out_path: Path,
) -> None:
    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(max(8, len(configs) * 0.9), 5))
    bars = ax.bar(
        x,
        values,
        yerr=yerr,
        capsize=4 if yerr else 0,
        color=color,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.92,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
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
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_scenario_grid(
    *,
    model_name: str,
    scenario_title: str,
    configs: list[str],
    metric_values: dict[str, list[float]],
    yerr_values: dict[str, list[float] | None] | None,
    suptitle: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    x = np.arange(len(configs))

    for ax, (metric_key, ylabel, color) in zip(axes.flat, METRICS):
        vals = metric_values[metric_key]
        yerr = yerr_values.get(metric_key) if yerr_values else None
        bars = ax.bar(
            x,
            vals,
            yerr=yerr,
            capsize=3 if yerr else 0,
            color=color,
            edgecolor="white",
            linewidth=0.6,
        )
        ax.set_title(ylabel, fontsize=11)
        ax.set_ylabel(ylabel.split(" (")[0])
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{val:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_per_run(df: pd.DataFrame, out_root: Path) -> None:
    for run_id in sorted(df["run_id"].unique()):
        run_df = df[df["run_id"] == run_id]
        for model_key, model_name in MODELS.items():
            model_df = run_df[run_df["model_key"] == model_key]
            for slug, batch_size, output_tokens, title in SCENARIOS:
                subset = model_df[
                    (model_df["batch_size"] == batch_size)
                    & (model_df["output_tokens"] == output_tokens)
                ]
                if subset.empty:
                    continue
                configs = sort_configs(subset["config"].tolist())
                by_cfg = {r["config"]: r for _, r in subset.iterrows()}
                metric_values = {
                    m: [by_cfg[c][m] for c in configs]
                    for m, _, _ in METRICS
                }
                plot_dir = out_root / "plots" / "per_run" / f"run_{run_id}" / model_key / slug
                plot_scenario_grid(
                    model_name=model_name,
                    scenario_title=title,
                    configs=configs,
                    metric_values=metric_values,
                    yerr_values=None,
                    suptitle=f"{model_name} — run {run_id} — {title}",
                    out_path=plot_dir / "all_metrics.png",
                )
                for metric_key, ylabel, color in METRICS:
                    plot_metric_bars(
                        configs=configs,
                        values=metric_values[metric_key],
                        yerr=None,
                        title=f"{model_name} — run {run_id} — {title}",
                        ylabel=ylabel,
                        color=color,
                        out_path=plot_dir / f"{metric_key}.png",
                    )


def plot_aggregated(df: pd.DataFrame, out_root: Path, exclude_min_max: bool) -> None:
    stat_key = "average_min_max_excluded" if exclude_min_max else "mean"
    subdir = "avg_min_max_excluded" if exclude_min_max else "mean"

    for model_key, model_name in MODELS.items():
        model_df = df[df["model_key"] == model_key]
        for slug, batch_size, output_tokens, title in SCENARIOS:
            subset = model_df[
                (model_df["batch_size"] == batch_size)
                & (model_df["output_tokens"] == output_tokens)
            ]
            if subset.empty:
                continue

            configs = sort_configs(subset["config"].unique())
            summary_rows = []
            for cfg in configs:
                g = subset[subset["config"] == cfg]
                row = {"config": cfg}
                for metric_key, _, _ in METRICS:
                    stats_row = summarize_group(g, metric_key)
                    row[metric_key] = stats_row[stat_key]
                    row[f"{metric_key}_sem"] = stats_row["sem"]
                summary_rows.append(row)

            metric_values = {m: [r[m] for r in summary_rows] for m, _, _ in METRICS}
            yerr_values = {
                m: [r[f"{m}_sem"] for r in summary_rows] for m, _, _ in METRICS
            }
            agg_label = (
                "average min max excluded"
                if exclude_min_max
                else "mean ± SEM"
            )
            plot_dir = out_root / "plots" / "aggregated" / subdir / model_key / slug
            plot_scenario_grid(
                model_name=model_name,
                scenario_title=title,
                configs=configs,
                metric_values=metric_values,
                yerr_values=yerr_values if not exclude_min_max else None,
                suptitle=f"{model_name} — {title} — {agg_label} over {subset['run_id'].nunique()} runs",
                out_path=plot_dir / "all_metrics.png",
            )
            for metric_key, ylabel, color in METRICS:
                plot_metric_bars(
                    configs=configs,
                    values=metric_values[metric_key],
                    yerr=yerr_values[metric_key] if not exclude_min_max else None,
                    title=f"{model_name} — {title} — {agg_label}",
                    ylabel=ylabel,
                    color=color,
                    out_path=plot_dir / f"{metric_key}.png",
                )


def write_raw_csv(df: pd.DataFrame, path: Path) -> None:
    cols = [
        "run_id",
        "model_name",
        "model_key",
        "strategy",
        "config",
        "workload",
        "batch_size",
        "output_tokens",
        "duration_s",
        "throughput_tok_s",
        "mj_per_token",
        "gpu_avg_power_w",
        "source_file",
    ]
    df[cols].sort_values(
        ["model_key", "workload", "config", "run_id"]
    ).to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL)


def write_summary_csv(df: pd.DataFrame, path: Path) -> None:
    rows: list[dict] = []
    group_cols = ["model_name", "model_key", "workload", "batch_size", "output_tokens", "config", "strategy"]
    for keys, g in df.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        for metric_key, metric_label, _ in METRICS:
            s = summarize_group(g, metric_key)
            rows.append(
                {
                    **base,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    **s,
                }
            )
    pd.DataFrame(rows).sort_values(
        ["model_key", "workload", "config", "metric"]
    ).to_csv(path, index=False)


def write_min_max_spread_csv(df: pd.DataFrame, path: Path) -> None:
    rows: list[dict] = []
    metrics_of_interest = ["throughput_tok_s", "mj_per_token", "duration_s"]
    group_cols = ["model_name", "model_key", "workload", "config", "strategy"]
    for keys, g in df.groupby(group_cols, sort=False):
        base = dict(zip(group_cols, keys))
        for metric in metrics_of_interest:
            vals = g[metric].to_numpy(dtype=float)
            mean_v = float(np.mean(vals))
            min_v = float(np.min(vals))
            max_v = float(np.max(vals))
            spread_abs = max_v - min_v
            spread_pct = (100.0 * spread_abs / mean_v) if mean_v else 0.0
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean_v,
                    "min": min_v,
                    "max": max_v,
                    "spread_abs": spread_abs,
                    "spread_pct_of_mean": spread_pct,
                    "min_run_id": int(g.loc[g[metric].idxmin(), "run_id"]),
                    "max_run_id": int(g.loc[g[metric].idxmax(), "run_id"]),
                }
            )
    pd.DataFrame(rows).sort_values(
        ["model_key", "workload", "config", "metric"]
    ).to_csv(path, index=False)


def one_way_anova(groups: list[np.ndarray]) -> dict:
    groups = [g for g in groups if len(g) > 0]
    if len(groups) < 2:
        return {"f_statistic": np.nan, "p_value": np.nan, "eta_squared": np.nan}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        f_stat, p_val = stats.f_oneway(*groups)
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    eta_sq = ss_between / ss_total if ss_total else np.nan
    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_val),
        "eta_squared": float(eta_sq),
    }


def two_way_anova(
    df: pd.DataFrame,
    factor_a: str,
    factor_b: str,
    metric: str,
) -> dict:
    """Balanced two-way ANOVA (fixed effects) via sums of squares."""
    sub = df[[factor_a, factor_b, metric]].dropna()
    levels_a = sorted(sub[factor_a].unique())
    levels_b = sorted(sub[factor_b].unique())
    if len(levels_a) < 2 or len(levels_b) < 2:
        return {}

    grand_mean = sub[metric].mean()
    n_total = len(sub)

    ss_a = sum(
        len(sub[sub[factor_a] == a]) * (sub.loc[sub[factor_a] == a, metric].mean() - grand_mean) ** 2
        for a in levels_a
    )
    ss_b = sum(
        len(sub[sub[factor_b] == b]) * (sub.loc[sub[factor_b] == b, metric].mean() - grand_mean) ** 2
        for b in levels_b
    )

    cell_means = sub.groupby([factor_a, factor_b])[metric].mean()
    cell_counts = sub.groupby([factor_a, factor_b])[metric].count()
    ss_ab = 0.0
    for a in levels_a:
        for b in levels_b:
            if (a, b) not in cell_means.index:
                continue
            n_cell = cell_counts.loc[(a, b)]
            ss_ab += n_cell * (cell_means.loc[(a, b)] - grand_mean) ** 2
    ss_ab -= ss_a + ss_b

    ss_within = 0.0
    for (a, b), g in sub.groupby([factor_a, factor_b]):
        cell_mean = g[metric].mean()
        ss_within += ((g[metric] - cell_mean) ** 2).sum()

    df_a = len(levels_a) - 1
    df_b = len(levels_b) - 1
    df_ab = df_a * df_b
    df_within = n_total - len(levels_a) * len(levels_b)
    if df_within <= 0:
        return {}

    ms_a = ss_a / df_a
    ms_b = ss_b / df_b
    ms_ab = ss_ab / df_ab if df_ab else np.nan
    ms_within = ss_within / df_within

    return {
        "factor_a": factor_a,
        "factor_b": factor_b,
        "metric": metric,
        "f_a": float(ms_a / ms_within),
        "p_a": float(stats.f.sf(ms_a / ms_within, df_a, df_within)),
        "f_b": float(ms_b / ms_within),
        "p_b": float(stats.f.sf(ms_b / ms_within, df_b, df_within)),
        "f_interaction": float(ms_ab / ms_within) if df_ab else np.nan,
        "p_interaction": float(stats.f.sf(ms_ab / ms_within, df_ab, df_within))
        if df_ab
        else np.nan,
        "eta_squared_a": float(ss_a / (ss_a + ss_b + ss_ab + ss_within)),
        "eta_squared_b": float(ss_b / (ss_a + ss_b + ss_ab + ss_within)),
        "eta_squared_interaction": float(ss_ab / (ss_a + ss_b + ss_ab + ss_within)),
        "n": n_total,
    }


def write_anova_csv(df: pd.DataFrame, path: Path) -> None:
    rows: list[dict] = []

    # One-way: parallel config within model × workload
    for (model_key, workload), g in df.groupby(["model_key", "workload"]):
        if g["config"].nunique() < 2:
            continue
        for metric_key, metric_label, _ in METRICS:
            groups = [grp[metric_key].to_numpy() for _, grp in g.groupby("config")]
            res = one_way_anova(groups)
            rows.append(
                {
                    "test": "one_way_config",
                    "factor_tested": "config",
                    "model_key": model_key,
                    "model_name": MODELS[model_key],
                    "workload": workload,
                    "config": "(all configs)",
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "n_groups": g["config"].nunique(),
                    "n_observations": len(g),
                    **res,
                }
            )

    # One-way: workload within model × config
    for (model_key, config), g in df.groupby(["model_key", "config"]):
        if g["workload"].nunique() < 2:
            continue
        for metric_key, metric_label, _ in METRICS:
            groups = [grp[metric_key].to_numpy() for _, grp in g.groupby("workload")]
            res = one_way_anova(groups)
            rows.append(
                {
                    "test": "one_way_workload",
                    "factor_tested": "workload",
                    "model_key": model_key,
                    "model_name": MODELS[model_key],
                    "workload": "(all workloads)",
                    "config": config,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    "n_groups": g["workload"].nunique(),
                    "n_observations": len(g),
                    **res,
                }
            )

    out = pd.DataFrame(rows)
    out.sort_values(
        ["model_key", "test", "workload", "config", "metric"],
        inplace=True,
    )
    out.to_csv(path, index=False)


def write_factorial_csv(df: pd.DataFrame, path: Path) -> None:
    rows: list[dict] = []
    for model_key, g in df.groupby("model_key"):
        for metric_key, metric_label, _ in METRICS:
            res = two_way_anova(g, "config", "workload", metric_key)
            if res:
                rows.append(
                    {
                        "model_key": model_key,
                        "metric": metric_key,
                        "metric_label": metric_label,
                        **res,
                    }
                )

    # Cross-model comparison: model × workload (collapse configs to run-level means per config first
    # would lose detail; instead test model effect within each config×workload cell)
    for (config, workload), g in df.groupby(["config", "workload"]):
        if g["model_key"].nunique() < 2:
            continue
        for metric_key, metric_label, _ in METRICS:
            groups = [grp[metric_key].to_numpy() for _, grp in g.groupby("model_key")]
            res = one_way_anova(groups)
            rows.append(
                {
                    "test": "one_way_model",
                    "config": config,
                    "workload": workload,
                    "metric": metric_key,
                    "metric_label": metric_label,
                    **res,
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)


def write_report_md(df: pd.DataFrame, out_root: Path, paths: dict[str, Path]) -> None:
    n_runs = df["run_id"].nunique()
    lines = [
        "# TokenPowerBench multi-run analysis report",
        "",
        f"- Runs analyzed: **{n_runs}**",
        f"- Models: {', '.join(MODELS.values())}",
        f"- Workloads: normal (bs128/out500), high concurrency (bs256/out500), high throughput (bs256/out2000)",
        "",
        "## Outputs",
        "",
        "| File | Description |",
        "|------|-------------|",
        f"| `{paths['raw'].name}` | All per-run measurements |",
        f"| `{paths['summary'].name}` | Mean, std, percentiles, average min max excluded per config |",
        f"| `{paths['spread'].name}` | Min/max spread for throughput, energy, duration |",
        f"| `{paths['anova'].name}` | One-way ANOVA |",
        f"| `{paths['factorial'].name}` | Two-way config×workload ANOVA per model |",
        "",
        "## Plots",
        "",
        "- `plots/per_run/run_<N>/<model>/<workload>/` — single-run bar charts",
        "- `plots/aggregated/mean/` — mean ± SEM across runs",
        "- `plots/aggregated/avg_min_max_excluded/` — average with min/max run excluded",
        "",
        "## Statistics notes",
        "",
        "- **Average min max excluded**: for 10 runs, drops the single lowest and highest value before averaging.",
        "- **One-way ANOVA (config)**: tests whether parallel strategy affects each metric within a model×workload (`factor_tested=config`, workload fixed).",
        "- **One-way ANOVA (workload)**: tests whether workload affects each metric within a model×config (`factor_tested=workload`, config fixed).",
        "- Rows use `(all configs)` or `(all workloads)` when that factor is the one being compared.",
        "- **Two-way ANOVA**: config × workload interaction per model; η² reports effect size.",
        "- **p99 / p95**: run-to-run percentiles from repeated benchmark executions, not request latency.",
        "",
    ]
    (out_root / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze multi-run TokenPowerBench results")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "tokenbench_qwen3_14b_27b_multi_run_result",
        help="Root folder containing run_1 … run_N subdirectories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Analysis output directory (default: <data-root>/analysis)",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_root = (args.output_dir or data_root / "analysis").resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {data_root} …")
    df = load_all_records(data_root)
    print(f"Loaded {len(df)} records across {df['run_id'].nunique()} runs")

    csv_dir = out_root / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "raw": csv_dir / "raw_measurements.csv",
        "summary": csv_dir / "summary_by_config.csv",
        "spread": csv_dir / "min_max_spread.csv",
        "anova": csv_dir / "anova_results.csv",
        "factorial": csv_dir / "factorial_anova.csv",
    }

    write_raw_csv(df, paths["raw"])
    write_summary_csv(df, paths["summary"])
    write_min_max_spread_csv(df, paths["spread"])
    write_anova_csv(df, paths["anova"])
    write_factorial_csv(df, paths["factorial"])
    write_report_md(df, out_root, paths)

    print("Generating per-run plots …")
    plot_per_run(df, out_root)
    print("Generating aggregated mean plots …")
    plot_aggregated(df, out_root, exclude_min_max=False)
    print("Generating aggregated average min max excluded plots …")
    plot_aggregated(df, out_root, exclude_min_max=True)

    print(f"\nDone. Analysis written to {out_root}")


if __name__ == "__main__":
    main()
