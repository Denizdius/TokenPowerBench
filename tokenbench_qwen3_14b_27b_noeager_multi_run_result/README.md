# TokenBench 10-run results (no `--enforce-eager`)

Same layout as `tokenbench_qwen3_14b_27b_multi_run_result/`, but uses
`run_noeager_*` benchmark scripts (CUDA graphs / torch.compile enabled).

## Submit

```bash
cd /gpfs/projects/etur83
sbatch run_10_tokenbench_qwen3_14b_27b_noeager.sh
```

## Configs per run

| Model | Scripts | Notes |
|-------|---------|--------|
| Qwen3-14B | 12 | all `run_noeager_qwen3_14b_*` |
| Qwen3.5-27B | 24 | all `run_noeager_qwen3.5_27b_*` **except** standalone DP2/DP4 |

**Skipped (27B OOM without enforce-eager):**
- `run_noeager_qwen3.5_27b_dp2_*` (3 configs)
- `run_noeager_qwen3.5_27b_dp4_*` (3 configs)

**Still included:** `tp2_dp2`, `pp2_dp2`, and all other 27B parallel modes.

**Total:** 36 configs × 10 runs = 360 benchmark jobs.

## Layout

```
tokenbench_qwen3_14b_27b_noeager_multi_run_result/
  logs/master_noeager_10run_<jobid>.log
  run_1/
    qwen3_14b_tokenbench_result/logs_run_noeager_*.txt
    qwen3.5_27b_tokenbench_result/logs_run_noeager_*.txt
    jsons/*_noeager_*.json
  run_2/
  ...
  run_10/
```

JSON run tags end with `_noeager` (e.g. `qwen3_14b_tp2_bs128_out500_noeager`).
