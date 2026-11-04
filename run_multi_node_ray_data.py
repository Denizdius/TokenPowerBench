#!/usr/bin/env python3
"""
Multi-node distributed LLM energy benchmark (vLLM + Ray Data LLM).

Uses ray.data.llm.vLLMEngineProcessorConfig and build_processor.
Does not replace run_multi_node.py (driver-side LLM + Ray executor).

Usage
-----
    python run_multi_node_ray_data.py \
        --models Qwen2.5-72B-Instruct \
        --model-dir /gpfs/projects/etur83/my_volume/lm-evaluation-harness \
        --datasets alpaca \
        --batch-sizes 128 \
        --tensor-parallel 8 \
        --pipeline-parallel 1 \
        --data-parallel 1 \
        --concurrency 1 \
        --num-samples 1024 \
        --max-tokens 500 \
        --gpu-memory-utilization 0.90 \
        --monitor gpu_only
"""

import argparse
import json
import os
import time
from pathlib import Path

# Let Ray assign GPUs across both nodes. SLURM would otherwise pin this
# driver to the 4 local devices and vLLM would refuse TP/PP > 4.
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

from tokenpowerbench.data import DatasetLoader
from tokenpowerbench.distributed.ray_cluster import RayClusterConfig, ensure_ray_initialized
from tokenpowerbench.distributed.vllm_ray_data import VLLMRayDataEngine
from tokenpowerbench.energy import create_monitor


def parse_args():
    p = argparse.ArgumentParser(
        description="Multi-node LLM energy benchmark (Ray Data LLM processor)"
    )

    p.add_argument("--models", required=True,
                   help="Comma-separated model names, e.g. 'Qwen2.5-72B-Instruct'")
    p.add_argument("--model-dir", default=os.path.expanduser("~/models"),
                   help="Base directory containing model folders (default: ~/models)")

    p.add_argument("--datasets", default="alpaca",
                   help="Comma-separated dataset names (alpaca, dolly, longbench, humaneval)")
    p.add_argument(
        "--dataset-path",
        default=None,
        help="Local dataset path for offline use (see run_single_node.py --dataset-path)",
    )
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--max-words", type=int, default=300)

    p.add_argument("--tensor-parallel", default="8",
                   help="Comma-separated TP sizes, e.g. '4,8,16'")
    p.add_argument("--pipeline-parallel", default="1",
                   help="Comma-separated PP sizes, e.g. '1,2,4'")
    p.add_argument("--data-parallel", default="1",
                   help=(
                       "DP replica count. DP>1 maps to processor concurrency "
                       "(one local LLM(tp, pp) per replica; not LLM(dp=k))."
                   ))
    p.add_argument("--concurrency", default="1",
                   help="Ray Data processor replicas (fixed pool of this size)")
    p.add_argument("--batch-sizes", default="256",
                   help="Comma-separated batch sizes")

    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)

    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="vLLM fraction of GPU memory for KV/weights (default: 0.90).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max_model_len; default: from model config.",
    )
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help=(
            "vLLM enforce_eager=True: disable CUDA graphs / torch.compile. "
            "Default is False (vLLM optimized path)."
        ),
    )
    p.add_argument(
        "--language-model-only",
        action="store_true",
        help=(
            "vLLM language_model_only=True: skip multimodal encoder/processor "
            "init for hybrid models (e.g. Qwen3.5)."
        ),
    )
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="vLLM model dtype (default: auto).",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip the one-prompt Ray Data processor warmup.",
    )

    p.add_argument("--ray-head-address", default=None,
                   help="Override Ray head address (default: read from env or SLURM)")
    p.add_argument("--ray-head-port", type=int, default=6379)

    p.add_argument("--monitor", default="auto",
                   choices=["auto", "gpu_only", "full_node"],
                   help="Energy monitor mode (default: auto)")
    p.add_argument(
        "--save-gpu-usage",
        action="store_true",
        help=(
            "Save per-sample GPU power (W), memory used (MB), and utilization "
            "(percent) for active GPUs as three JSON traces next to results."
        ),
    )

    p.add_argument("--output-dir", default="./results")
    p.add_argument("--verbose", action="store_true")

    return p.parse_args()


def run():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = [m.strip() for m in args.models.split(",")]
    datasets = [d.strip() for d in args.datasets.split(",")]
    tp_sizes = [int(x) for x in args.tensor_parallel.split(",")]
    pp_sizes = [int(x) for x in args.pipeline_parallel.split(",")]
    dp_sizes = [int(x) for x in args.data_parallel.split(",")]
    concurrencies = [int(x) for x in args.concurrency.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        print("Error: --gpu-memory-utilization must be in (0, 1].")
        return
    if any(v < 1 for v in tp_sizes + pp_sizes + dp_sizes + concurrencies + batch_sizes):
        print("Error: TP, PP, DP, concurrency, and batch size must be >= 1.")
        return

    if args.ray_head_address:
        cluster = RayClusterConfig(
            head_address=args.ray_head_address,
            head_port=args.ray_head_port,
        )
    else:
        cluster = RayClusterConfig.from_slurm()

    print(f"Ray cluster: {cluster.ray_init_address}")
    print("Executor: Ray Data LLM (vLLMEngineProcessorConfig + build_processor)")
    ensure_ray_initialized(cluster)

    loader = DatasetLoader()
    all_results = {}

    for model_name in models:
        model_path = os.path.join(args.model_dir, model_name)
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}  — skipping.")
            continue

        model_results = {}

        for dataset_name in datasets:
            prompts = loader.load(
                dataset_name,
                num_samples=args.num_samples,
                min_words=args.min_words,
                max_words=args.max_words,
                dataset_path=args.dataset_path,
            )
            if not prompts:
                print(f"No prompts for {dataset_name} — skipping.")
                continue

            for tp in tp_sizes:
                for pp in pp_sizes:
                    for dp in dp_sizes:
                        for concurrency in concurrencies:
                            for batch_size in batch_sizes:
                                cfg_key = (
                                    f"TP{tp}_PP{pp}_DP{dp}_C{concurrency}_B{batch_size}"
                                )
                                print(f"\n{'='*70}")
                                print(
                                    f"Model: {model_name}  Dataset: {dataset_name}  "
                                    f"Config: {cfg_key}  executor=ray_data_llm"
                                )
                                print(f"{'='*70}")

                                engine_config = dict(
                                    model_path=model_path,
                                    tensor_parallel_size=tp,
                                    pipeline_parallel_size=pp,
                                    data_parallel_size=dp,
                                    concurrency=concurrency,
                                    batch_size=batch_size,
                                    max_tokens=args.max_tokens,
                                    temperature=args.temperature,
                                    top_p=args.top_p,
                                    gpu_memory_utilization=args.gpu_memory_utilization,
                                    max_model_len=args.max_model_len,
                                    enforce_eager=args.enforce_eager,
                                    language_model_only=args.language_model_only,
                                    dtype=args.dtype,
                                    warmup=not args.no_warmup,
                                    verbose=args.verbose,
                                )

                                engine = VLLMRayDataEngine(cluster, engine_config)
                                if not engine.prepare():
                                    print(f"Failed to load model for {cfg_key}")
                                    engine.shutdown()
                                    continue
                                monitor = create_monitor(
                                    args.monitor,
                                    tensor_parallel_size=tp,
                                    pipeline_parallel_size=pp,
                                    data_parallel_size=dp,
                                    use_ray_cluster=True,
                                )

                                monitor.start()
                                result = engine.run_generate(prompts)
                                time.sleep(2.0)
                                monitor.stop()
                                engine.shutdown()

                                if result is None:
                                    print(f"Benchmark failed for {cfg_key}")
                                    continue

                                perf = result["performance_metrics"]
                                duration = perf["total_time_s"]
                                total_tokens = perf["total_tokens"]
                                num_responses = perf.get("total_prompts", len(prompts))

                                energy = monitor.compute_metrics(
                                    duration, total_tokens, num_responses
                                )
                                print(energy.summary())

                                slug = model_name.replace("/", "_")
                                ts = time.strftime("%Y%m%d_%H%M%S")
                                stem = (
                                    f"{slug}_{dataset_name}_raydata"
                                    f"_tp{tp}_pp{pp}_dp{dp}"
                                    f"_c{concurrency}_b{batch_size}_{ts}"
                                )
                                fname = output_dir / f"{stem}.json"

                                gpu_usage_files = None
                                if args.save_gpu_usage:
                                    gpu_usage_files = monitor.save_gpu_usage(
                                        output_dir / stem
                                    )

                                result["energy_metrics"] = {
                                    "monitor_mode": args.monitor,
                                    "duration_s": duration,
                                    "gpu_avg_power_w": energy.gpu_avg_power_w,
                                    "gpu_energy_j": energy.gpu_energy_j,
                                    "gpu_mj_per_token": energy.gpu_mj_per_token,
                                    "per_gpu_power_w": energy.per_gpu_power_w,
                                    "gpus_included_for_energy": energy.gpus_included_for_energy,
                                    "cpu_avg_power_w": energy.cpu_avg_power_w,
                                    "cpu_energy_j": energy.cpu_energy_j,
                                    "dram_avg_power_w": energy.dram_avg_power_w,
                                    "dram_energy_j": energy.dram_energy_j,
                                    "system_avg_power_w": energy.system_avg_power_w,
                                    "system_energy_j": energy.system_energy_j,
                                    "total_energy_j": energy.total_energy_j,
                                    "total_mj_per_token": energy.total_mj_per_token,
                                    "gpu_memory_utilization": args.gpu_memory_utilization,
                                    "max_model_len_cli": args.max_model_len,
                                    "enforce_eager": args.enforce_eager,
                                    "language_model_only": args.language_model_only,
                                    "dtype": args.dtype,
                                    "warmup": not args.no_warmup,
                                    "executor": "ray_data_llm",
                                    "gpu_power_file": (
                                        gpu_usage_files["power"] if gpu_usage_files else None
                                    ),
                                    "gpu_memory_file": (
                                        gpu_usage_files["memory"] if gpu_usage_files else None
                                    ),
                                    "gpu_utilization_file": (
                                        gpu_usage_files["utilization"]
                                        if gpu_usage_files
                                        else None
                                    ),
                                }

                                with open(fname, "w") as f:
                                    json.dump(result, f, indent=2, ensure_ascii=False)
                                print(f"Saved: {fname}")

                                model_results.setdefault(dataset_name, {})[cfg_key] = result

        if model_results:
            all_results[model_name] = model_results

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f"summary_raydata_{ts}.json"
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == "__main__":
    run()
