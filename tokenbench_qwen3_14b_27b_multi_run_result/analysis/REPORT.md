# TokenPowerBench multi-run analysis report

- Runs analyzed: **10**
- Models: Qwen3-14B, Qwen3.5-27B
- Workloads: normal (bs128/out500), high concurrency (bs256/out500), high throughput (bs256/out2000)

## Outputs

| File | Description |
|------|-------------|
| `raw_measurements.csv` | All per-run measurements |
| `summary_by_config.csv` | One row per config with average duration, throughput, mj_per_token, gpu_avg_power |
| `min_max_spread.csv` | Min/max spread per config (metrics as columns) |
| `anova_results.csv` | One-way ANOVA (metrics as columns) |
| `factorial_anova.csv` | Two-way config×workload ANOVA per model (metrics as columns) |
| `model_anova.csv` | Model comparison ANOVA per config×workload (metrics as columns) |

## Plots

- `plots/per_run/run_<N>/<model>/<workload>/` — single-run bar charts
- `plots/aggregated/mean/` — mean ± SEM across runs
- `plots/aggregated/avg_min_max_excluded/` — average with min/max run excluded

## Statistics notes

- **Average min max excluded**: for 10 runs, drops the single lowest and highest value before averaging.
- **One-way ANOVA**: compares configs (workload fixed, config blank) or workloads (config fixed, workload blank) per metric.
- **Two-way ANOVA**: config × workload interaction per model; η² reports effect size.
- **p99 / p95**: run-to-run percentiles from repeated benchmark executions, not request latency.
