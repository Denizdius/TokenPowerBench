"""
vLLM data-parallel worker process (spawned by VLLMEngine).

Two modes when DP > 1:

- **Coordinated** (TP=PP=1): vLLM external_launcher + native ``data_parallel_size``.
- **Replica** (TP>1 or PP>1): independent ``LLM(tp=…, dp=1)`` per worker with
  ``CUDA_VISIBLE_DEVICES`` pinned (lm_eval-style). Avoids NCCL deadlocks where
  vLLM expects ``TP×DP`` processes but only ``DP`` are spawned.
"""

from __future__ import annotations

import os
import time
import traceback
from typing import Any, List, Optional, Tuple

_DP_ENV_KEYS = (
    "VLLM_DP_RANK",
    "VLLM_DP_RANK_LOCAL",
    "VLLM_DP_SIZE",
    "VLLM_DP_MASTER_IP",
    "VLLM_DP_MASTER_PORT",
    "RANK",
    "LOCAL_RANK",
    "WORLD_SIZE",
    "MASTER_ADDR",
    "MASTER_PORT",
)


def _load_llm(config: dict):
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    lora_path = config.pop("lora_path", None)
    lora_name = config.pop("lora_name", "finetuned")
    lora_int_id = config.pop("lora_int_id", 1)
    max_lora_rank = config.pop("max_lora_rank", 64)

    if lora_path:
        config["enable_lora"] = True
        config["max_lora_rank"] = max_lora_rank
        lora_request = LoRARequest(lora_name, lora_int_id, lora_path)
    else:
        lora_request = None

    llm = LLM(**config)
    return llm, lora_request


def _run_batches(
    llm,
    prompts: List[str],
    batch_size: int,
    max_tokens: int,
    temperature: float,
    lora_request: Optional[Any],
) -> List[Any]:
    from vllm import SamplingParams

    params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
    gen_kw: dict = {}
    if lora_request is not None:
        gen_kw["lora_request"] = lora_request

    results: List[Any] = []
    for i in range(0, len(prompts), batch_size):
        results.extend(
            llm.generate(
                prompts[i : i + batch_size],
                params,
                **gen_kw,
            )
        )
    return results


def _clear_distributed_env() -> None:
    for key in _DP_ENV_KEYS:
        os.environ.pop(key, None)


def configure_coordinated_dp_env(
    *,
    global_dp_rank: int,
    local_dp_rank: int,
    dp_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    torch_master_port: int,
) -> None:
    """Env vars for vLLM external_launcher when TP=PP=1 and DP>1."""
    _clear_distributed_env()
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["RANK"] = str(global_dp_rank)
    os.environ["LOCAL_RANK"] = str(local_dp_rank)
    os.environ["WORLD_SIZE"] = str(dp_size)
    os.environ["MASTER_ADDR"] = dp_master_ip
    os.environ["MASTER_PORT"] = str(torch_master_port)


def configure_replica_env(*, cuda_visible_devices: str) -> None:
    """Pin GPUs and clear vLLM DP env leaked from the parent process."""
    _clear_distributed_env()
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


def cuda_visible_devices_for_replica(
    global_dp_rank: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
) -> str:
    gpus_per_replica = tensor_parallel_size * pipeline_parallel_size
    start = global_dp_rank * gpus_per_replica
    return ",".join(str(start + i) for i in range(gpus_per_replica))


def vllm_dp_worker_entry(
    global_dp_rank: int,
    local_dp_rank: int,
    dp_size: int,
    use_coordinated_dp: bool,
    cuda_visible_devices: Optional[str],
    dp_master_ip: str,
    dp_master_port: int,
    torch_master_port: int,
    config: dict,
    task_queue,
    result_queue,
    ready_queue,
    error_queue,
) -> None:
    """Entry point for one data-parallel vLLM worker process."""
    try:
        if use_coordinated_dp:
            configure_coordinated_dp_env(
                global_dp_rank=global_dp_rank,
                local_dp_rank=local_dp_rank,
                dp_size=dp_size,
                dp_master_ip=dp_master_ip,
                dp_master_port=dp_master_port,
                torch_master_port=torch_master_port,
            )
            mode = "coordinated"
        else:
            assert cuda_visible_devices is not None
            configure_replica_env(cuda_visible_devices=cuda_visible_devices)
            mode = f"replica gpus=[{cuda_visible_devices}]"

        print(
            f"[VLLMDPWorker rank={global_dp_rank}] Loading model "
            f"(local_rank={local_dp_rank}, dp_size={dp_size}, mode={mode})…"
        )
        llm, lora_request = _load_llm(dict(config))
        print(f"[VLLMDPWorker rank={global_dp_rank}] Model ready.")
        ready_queue.put(global_dp_rank)

        while True:
            task = task_queue.get()
            if task is None:
                break

            start_idx, prompts, batch_size, max_tokens, temperature, placeholder = task
            if placeholder:
                _run_batches(
                    llm,
                    prompts,
                    batch_size=1,
                    max_tokens=min(max_tokens, 4),
                    temperature=temperature,
                    lora_request=lora_request,
                )
                result_queue.put((start_idx, [], placeholder))
                continue

            outputs = _run_batches(
                llm,
                prompts,
                batch_size,
                max_tokens,
                temperature,
                lora_request,
            )
            result_queue.put((start_idx, outputs, placeholder))

        time.sleep(1.0)
    except Exception:
        traceback.print_exc()
        error_queue.put((global_dp_rank, "worker failed; see traceback above"))
        raise


def shard_range(length: int, dp_size: int, rank: int) -> Tuple[int, int]:
    """Return [start, end) indices for ``rank`` out of ``dp_size`` shards."""
    floor = length // dp_size
    remainder = length % dp_size

    def _start(r: int) -> int:
        return r * floor + min(r, remainder)

    return _start(rank), _start(rank + 1)
