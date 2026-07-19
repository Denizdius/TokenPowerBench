#!/usr/bin/env python3
"""
Single-node LLM energy benchmark.

Usage
-----
    python run_single_node.py \
        --model /path/to/model \
        --engine vllm \
        --batch-sizes 128,256 \
        --num-samples 5000 \
        --output-tokens 500 \
        --gpu-memory-utilization 0.85 \
        --tensor-parallel-size 1 \
        --pipeline-parallel-size 1 \
        --data-parallel-size 1 \
        --dataset alpaca \
        --dataset-path /data/alpaca \
        --monitor auto        # "auto" | "gpu_only" | "full_node"

Monitor modes
-------------
  auto       Automatically use full_node if RAPL is accessible, else gpu_only.
  gpu_only   GPU power via NVML only. No root required.
  full_node  GPU + CPU (Intel RAPL) + total node (IPMI).
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch

from tokenpowerbench.data import DatasetLoader
from tokenpowerbench.energy import create_monitor
from tokenpowerbench.engines import VLLMEngine


def parse_args():
    p = argparse.ArgumentParser(description="Single-node LLM energy benchmark")

    p.add_argument(
        "--model",
        required=True,
        help=(
            "Hugging Face Hub model id (e.g. unsloth/Qwen3-8B-Base-unsloth-bnb-4bit) "
            "or local model directory path"
        ),
    )
    p.add_argument("--engine", default="vllm", choices=["vllm"],
                   help="Inference engine (default: vllm)")

    p.add_argument("--dataset", default="alpaca",
                   choices=["alpaca", "dolly", "longbench", "humaneval"],
                   help="Dataset name: selects how rows are parsed into prompts")
    p.add_argument(
        "--dataset-path",
        default=None,
        help=(
            "Local dataset path (offline). Directory from datasets.save_to_disk(), "
            "a .json/.jsonl file, or a .txt file (one prompt per line). "
            "When set, Hugging Face Hub is not used."
        ),
    )
    p.add_argument("--num-samples", type=int, default=5000,
                   help="Number of inference requests")
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--max-words", type=int, default=300)

    p.add_argument("--batch-sizes", default="256",
                   help="Comma-separated list of batch sizes, e.g. '128,256,512'")
    p.add_argument("--output-tokens", type=int, default=500,
                   help="Max tokens to generate per prompt")

    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="vLLM fraction of GPU memory for KV/weights (default: 0.85).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="vLLM max_model_len; default: from model config.",
    )
    p.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help="vLLM tensor_parallel_size (default: all visible GPUs).",
    )
    p.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=1,
        help="vLLM pipeline_parallel_size (default: 1).",
    )
    p.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help=(
            "vLLM data_parallel_size (default: 1). Values > 1 spawn one process "
            "per DP rank. TP=PP=1 uses vLLM coordinated DP; TP/PP>1 uses "
            "independent replicas. Requires TP×PP×DP visible GPUs."
        ),
    )
    p.add_argument(
        "--dp-num-nodes",
        type=int,
        default=1,
        help="Total nodes for multi-node data parallel (default: 1).",
    )
    p.add_argument(
        "--dp-node-rank",
        type=int,
        default=0,
        help="This node's rank for multi-node data parallel (default: 0).",
    )
    p.add_argument(
        "--dp-master-addr",
        default="",
        help="Master IP for multi-node DP coordination (required if dp-num-nodes > 1).",
    )
    p.add_argument(
        "--dp-master-port",
        type=int,
        default=0,
        help="Master port for multi-node DP (default: auto).",
    )
    p.add_argument(
        "--language-model-only",
        action="store_true",
        help=(
            "vLLM language_model_only=True: skip multimodal encoder/processor "
            "init for hybrid models and run text-only inference (faster load)."
        ),
    )
    p.add_argument(
        "--enforce-eager",
        action="store_true",
        help=(
            "vLLM enforce_eager=True: disable CUDA graphs / torch.compile capture. "
            "Slower but more compatible; default is False (vLLM optimized path)."
        ),
    )
    p.add_argument(
        "--dtype",
        default="auto",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        help="vLLM model dtype (default: auto).",
    )
    p.add_argument(
        "--lora-path",
        default=None,
        help=(
            "Local path to LoRA adapter directory. Enables vLLM LoRA "
            "(enable_lora=True) and applies this adapter on every generate call."
        ),
    )
    p.add_argument(
        "--max-lora-rank",
        type=int,
        default=64,
        help="vLLM max_lora_rank when --lora-path is set (default: 64).",
    )
    p.add_argument(
        "--lora-name",
        default="finetuned",
        help="LoRA adapter name passed to vLLM LoRARequest (default: finetuned).",
    )

    p.add_argument("--monitor", default="auto",
                   choices=["auto", "gpu_only", "full_node"],
                   help="Energy monitor mode (default: auto)")
    p.add_argument(
        "--save-gpu-usage",
        action="store_true",
        help=(
            "Save per-sample GPU power (W), memory used (MB), and utilization "
            "(%) for active GPUs (TP×PP×DP) as three JSON traces next to results."
        ),
    )

    p.add_argument("--output-dir", default="./results",
                   help="Directory for result JSON files")
    p.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Optional label used as the result JSON filename stem "
            "(e.g. qwen3_14b_dp2_bs128_out500). When set, output is "
            "{run_tag}_{timestamp}.json instead of the auto-generated name."
        ),
    )

    return p.parse_args()


def run():
    args = parse_args()
    batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        print("Error: --gpu-memory-utilization must be in (0, 1].")
        return
    if args.tensor_parallel_size is not None and args.tensor_parallel_size < 1:
        print("Error: --tensor-parallel-size must be >= 1.")
        return
    if args.pipeline_parallel_size < 1 or args.data_parallel_size < 1:
        print("Error: --pipeline-parallel-size and --data-parallel-size must be >= 1.")
        return
    if args.dp_num_nodes < 1 or args.dp_node_rank < 0:
        print("Error: --dp-num-nodes must be >= 1 and --dp-node-rank must be >= 0.")
        return
    if args.data_parallel_size % args.dp_num_nodes != 0:
        print(
            "Error: --data-parallel-size must be divisible by --dp-num-nodes."
        )
        return
    if args.dp_num_nodes > 1 and not args.dp_master_addr:
        print("Error: --dp-master-addr is required when --dp-num-nodes > 1.")
        return
    if args.lora_path and args.max_lora_rank < 1:
        print("Error: --max-lora-rank must be >= 1 when --lora-path is set.")
        return

    n_gpus = max(torch.cuda.device_count(), 1) if torch.cuda.is_available() else 1
    tp = args.tensor_parallel_size if args.tensor_parallel_size is not None else n_gpus
    pp = args.pipeline_parallel_size
    dp = args.data_parallel_size
    if tp * pp * dp > n_gpus:
        print(
            f"Error: TP×PP×DP = {tp}×{pp}×{dp} = {tp * pp * dp} "
            f"exceeds visible GPU count ({n_gpus})."
        )
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = DatasetLoader()
    prompts = loader.load(
        args.dataset,
        num_samples=args.num_samples,
        min_words=args.min_words,
        max_words=args.max_words,
        dataset_path=args.dataset_path,
    )
    if not prompts:
        print("No prompts loaded. Exiting.")
        return

    engine = VLLMEngine()
    if not engine.available:
        print("vLLM is not installed. Run: pip install vllm")
        return

    try:
        model = engine.setup_model(
            args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            tensor_parallel_size=tp,
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            enforce_eager=args.enforce_eager,
            dtype=args.dtype,
            lora_path=args.lora_path,
            max_lora_rank=args.max_lora_rank,
            lora_name=args.lora_name,
            dp_num_nodes=args.dp_num_nodes,
            dp_node_rank=args.dp_node_rank,
            dp_master_addr=args.dp_master_addr,
            dp_master_port=args.dp_master_port,
            language_model_only=args.language_model_only,
        )
        if model is None:
            print(f"Failed to load model from {args.model}")
            return

        print("Running warmup pass…")
        engine.run_inference([prompts[0]], batch_size=1, max_tokens=20)

        all_results = {}

        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"Batch size: {batch_size}  |  monitor: {args.monitor}")
            print(f"{'='*60}")

            monitor = create_monitor(
                args.monitor,
                tensor_parallel_size=tp,
                pipeline_parallel_size=pp,
                data_parallel_size=dp,
            )
            monitor.start()

            outputs, t0, t1 = engine.run_benchmark(
                prompts, args.num_samples, batch_size, args.output_tokens
            )

            time.sleep(2.0)
            monitor.stop()

            duration = t1 - t0
            total_tokens = engine.estimate_tokens(outputs)
            metrics = monitor.compute_metrics(duration, total_tokens, len(outputs))

            print(metrics.summary())

            gpu_usage_files = None
            if args.save_gpu_usage:
                model_slug = os.path.basename(args.model.rstrip("/"))
                ts = time.strftime("%Y%m%d_%H%M%S")
                stem = args.run_tag if args.run_tag else f"{model_slug}_{args.engine}"
                gpu_usage_files = monitor.save_gpu_usage(
                    output_dir / f"{stem}_batch{batch_size}_{ts}"
                )

            all_results[f"batch_{batch_size}"] = {
                "model": args.model,
                "engine": args.engine,
                "dataset": args.dataset,
                "dataset_path": args.dataset_path,
                "batch_size": batch_size,
                "num_samples": args.num_samples,
                "output_tokens": args.output_tokens,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "max_model_len_cli": args.max_model_len,
                "tensor_parallel_size": tp,
                "pipeline_parallel_size": pp,
                "data_parallel_size": dp,
                "dp_num_nodes": args.dp_num_nodes,
                "dp_node_rank": args.dp_node_rank,
                "enforce_eager": args.enforce_eager,
                "language_model_only": args.language_model_only,
                "dtype": args.dtype,
                "lora_path": args.lora_path,
                "max_lora_rank": args.max_lora_rank if args.lora_path else None,
                "lora_name": args.lora_name if args.lora_path else None,
                "monitor_mode": args.monitor,
                "duration_s": duration,
                "total_output_tokens": total_tokens,
                "num_responses": len(outputs),
                "gpu_avg_power_w": metrics.gpu_avg_power_w,
                "gpu_energy_j": metrics.gpu_energy_j,
                "gpu_mj_per_token": metrics.gpu_mj_per_token,
                "per_gpu_power_w": metrics.per_gpu_power_w,
                "gpus_included_for_energy": metrics.gpus_included_for_energy,
                "cpu_avg_power_w": metrics.cpu_avg_power_w,
                "cpu_energy_j": metrics.cpu_energy_j,
                "dram_avg_power_w": metrics.dram_avg_power_w,
                "dram_energy_j": metrics.dram_energy_j,
                "system_avg_power_w": metrics.system_avg_power_w,
                "system_energy_j": metrics.system_energy_j,
                "total_energy_j": metrics.total_energy_j,
                "total_mj_per_token": metrics.total_mj_per_token,
                "gpu_power_file": (
                    gpu_usage_files["power"] if gpu_usage_files else None
                ),
                "gpu_memory_file": (
                    gpu_usage_files["memory"] if gpu_usage_files else None
                ),
                "gpu_utilization_file": (
                    gpu_usage_files["utilization"] if gpu_usage_files else None
                ),
            }

        model_slug = os.path.basename(args.model.rstrip("/"))
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        if args.run_tag:
            out_file = output_dir / f"{args.run_tag}_{timestamp}.json"
        else:
            out_file = output_dir / (
                f"{model_slug}_{args.engine}_"
                f"b{'_'.join(str(b) for b in batch_sizes)}_{timestamp}.json"
            )
        with open(out_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {out_file}")
    finally:
        engine.shutdown()


if __name__ == "__main__":
    run()
