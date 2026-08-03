# TokenPowerBench multi-run analysis report

- Runs analyzed: **10**
- Models: Qwen3-14B, Qwen3.5-27B
- Workloads: normal (bs128/out500), high concurrency (bs256/out500), high throughput (bs256/out2000)

## Outputs

| File | Description |
|------|-------------|
| `raw_measurements.csv` | All per-run measurements |
| `summary_by_config.csv` | One row per config with average duration, throughput, mj_per_token, gpu_avg_power |
| `summary_min_max_excluded.csv` | Same as summary but average min/max run excluded |
| `min_max_spread.csv` | Min/max spread per config (metrics as columns) |
| `anova_results.csv` | One-way ANOVA (metrics as columns) |
| `factorial_anova.csv` | Two-way config×workload ANOVA per model (metrics as columns) |
| `model_anova.csv` | Model comparison ANOVA per config×workload (metrics as columns) |
| `eager_vs_noeager_absolute.csv` | Eager vs no-eager (min/max excluded): values and absolute diff |
| `eager_vs_noeager_percent.csv` | Eager vs no-eager percent diff (↑ higher, ↓ lower) |

## Plots

- `plots/per_run/run_<N>/<model>/<workload>/` — single-run bar charts
- `plots/aggregated/mean/` — mean ± SEM across runs
- `plots/aggregated/avg_min_max_excluded/` — average with min/max run excluded
- `plots/eager_vs_noeager/values/` — eager vs no-eager side-by-side (min/max excluded)
- `plots/eager_vs_noeager/percent_diff/` — percent change with ↑/↓ labels

## Statistics notes

- **Average min max excluded**: for 10 runs, drops the single lowest and highest value before averaging.
- **One-way ANOVA**: compares configs (workload fixed, config blank) or workloads (config fixed, workload blank) per metric.
- **Two-way ANOVA**: config × workload interaction per model; η² reports effect size.
- **p99 / p95**: run-to-run percentiles from repeated benchmark executions, not request latency.
