# TokenPowerBench multi-run analysis report

- Runs analyzed: **10**
- Models: Qwen3-14B, Qwen3.5-27B
- Workloads: normal (bs128/out500), high concurrency (bs256/out500), high throughput (bs256/out2000)

## Outputs

| File | Description |
|------|-------------|
| `raw_measurements.csv` | All per-run measurements |
| `summary_by_config.csv` | Mean, std, percentiles, average min max excluded per config |
| `min_max_spread.csv` | Min/max spread for throughput, energy, duration |
| `anova_results.csv` | One-way ANOVA |
| `factorial_anova.csv` | Two-way config×workload ANOVA per model |

## Plots

- `plots/per_run/run_<N>/<model>/<workload>/` — single-run bar charts
- `plots/aggregated/mean/` — mean ± SEM across runs
- `plots/aggregated/avg_min_max_excluded/` — average with min/max run excluded

## Statistics notes

- **Average min max excluded**: for 10 runs, drops the single lowest and highest value before averaging.
- **One-way ANOVA (config)**: tests whether parallel strategy affects each metric within a model×workload (`factor_tested=config`, workload fixed).
- **One-way ANOVA (workload)**: tests whether workload affects each metric within a model×config (`factor_tested=workload`, config fixed).
- Rows use `(all configs)` or `(all workloads)` when that factor is the one being compared.
- **Two-way ANOVA**: config × workload interaction per model; η² reports effect size.
- **p99 / p95**: run-to-run percentiles from repeated benchmark executions, not request latency.
