"""Command-line interface to the TokenPowerBench Python API."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from . import api
from .runtime import runtime_identity


def positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _benchmark_arguments(args):
    return {key: value for key, value in vars(args).items() if key not in ("engine", "check_monitor")}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Benchmark LLM inference energy with explicit sensor scope and optional phase timing.")
    p.add_argument("--version", action="version", version=f"tokenpowerbench {__version__}")
    p.add_argument("--model", help="Local model directory or Hugging Face model ID")
    p.add_argument("--engine", default="vllm", choices=["vllm"])
    source = p.add_mutually_exclusive_group()
    source.add_argument("--dataset", choices=["alpaca", "dolly", "longbench", "humaneval"], help="Dataset source (default: alpaca when no prompt file is supplied)")
    source.add_argument("--prompts-file", type=Path, help="JSON array of prompt strings (no dataset download)")
    p.add_argument("--dataset-path", type=Path, help="Local file or directory for --dataset. The dataset name selects the parser; the path can be any local Alpaca or LongBench export.")
    p.add_argument("--language-model-only", action="store_true", help="Skip multimodal encoder init and run text-only inference.")
    p.add_argument("--num-samples", type=positive_int, help="Request count (default: prompt file length, or 1000 for a dataset)")
    p.add_argument("--min-words", type=int, default=2)
    p.add_argument("--max-words", type=positive_int, default=300)
    p.add_argument("--batch-sizes", default="1")
    p.add_argument("--output-tokens", type=positive_int, default=128)
    p.add_argument("--max-model-len", type=positive_int, help="Explicit model context limit")
    p.add_argument("--tensor-parallel-size", type=positive_int, help="vLLM tensor parallel size. With pipeline and data parallel left at 1, it must match the CUDA-visible GPU count.")
    p.add_argument("--pipeline-parallel-size", type=positive_int, default=1, help="vLLM pipeline parallel size (default: 1).")
    p.add_argument("--data-parallel-size", type=positive_int, default=1, help="Data parallel size. Values above 1 spawn one worker per rank and need tensor_parallel_size x pipeline_parallel_size x data_parallel_size visible GPUs.")
    p.add_argument("--dp-num-nodes", type=positive_int, default=1, help="Node count for multi-node data parallel (default: 1).")
    p.add_argument("--dp-node-rank", type=int, default=0, help="This node's rank for multi-node data parallel (default: 0).")
    p.add_argument("--dp-master-addr", default="", help="Master address for multi-node data parallel. Required when --dp-num-nodes is greater than 1.")
    p.add_argument("--dp-master-port", type=int, default=0, help="Master port for multi-node data parallel (default: auto).")
    p.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graph and torch.compile capture.")
    p.add_argument("--dtype", default="auto", choices=["auto", "half", "float16", "bfloat16", "float", "float32"], help="vLLM model dtype (default: auto).")
    p.add_argument("--lora-path", type=Path, help="Local LoRA adapter directory applied to every generate call.")
    p.add_argument("--max-lora-rank", type=positive_int, default=64, help="vLLM max LoRA rank when --lora-path is set (default: 64).")
    p.add_argument("--lora-name", default="finetuned", help="LoRA adapter name (default: finetuned).")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--phase-profiling", action="store_true", help="Serial first-token phase profiling; requires --batch-sizes 1")
    p.add_argument("--monitor", default="auto", choices=["auto", "gpu_only", "full_node"],
                   help="auto: probe access; gpu_only: NVML; full_node: require IPMI, with optional CPU RAPL")
    p.add_argument("--check-monitor", action="store_true", help="Print process identity and sensor access without loading a model")
    p.add_argument("--save-gpu-usage", action="store_true", help="Save per-sample GPU power (W), memory (MB), and utilization (percent) beside the results.")
    p.add_argument("--run-tag", help="Also write {run-tag}_{timestamp}.json in the output directory.")
    p.add_argument("--output-dir", type=Path, default=Path("results"))
    args = p.parse_args(argv)
    if not args.model and not args.check_monitor:
        p.error("--model is required unless --check-monitor is used")
    try:
        args.batch_sizes = [int(value) for value in args.batch_sizes.split(",")]
        options = _benchmark_arguments(args)
        options.update(model=args.model if args.model is not None else "environment-check", prompts=None)
        api._validate_options(options)
    except (TypeError, ValueError) as exc:
        p.error(str(exc))
    return args


def main(argv=None):
    """Run the shared API; return 0, 1 for failure, or 130 for interruption."""
    args = parse_args(argv)
    try:
        # Identity stays observable even if the subsequent sensor probe fails.
        print(json.dumps({"runtime": runtime_identity()}, indent=2))
        if args.check_monitor:
            report = api.check_environment(monitor=args.monitor)
            print(json.dumps({"sensors": report["sensors"]}, indent=2))
        else:
            result = api.benchmark(**_benchmark_arguments(args))
            print(f"Results and raw samples: {result.output_dir}")
        return 0
    except KeyboardInterrupt:
        print("Benchmark interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Benchmark failed: {exc}", file=sys.stderr)
        return 1
