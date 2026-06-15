"""
vLLM single-node inference engine.
"""

from __future__ import annotations

import os
import time
from multiprocessing import get_context
from pathlib import Path
from typing import Any, List, Optional, Tuple

import torch

from .base import InferenceEngine
from .vllm_dp_worker import shard_range, vllm_dp_worker_entry

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False
    LoRARequest = None  # type: ignore

try:
    from vllm.utils.network_utils import get_open_port as _vllm_get_open_port
except ImportError:
    import socket

    def _vllm_get_open_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return sock.getsockname()[1]


class VLLMEngine(InferenceEngine):
    """
    vLLM inference engine for single-node benchmarking.

    ``model`` may be a local directory or a Hugging Face Hub model id
    (e.g. ``unsloth/Qwen3-8B-Base-unsloth-bnb-4bit``).

    When ``data_parallel_size > 1`` with ``tensor_parallel_size == 1`` and
    ``pipeline_parallel_size == 1``, spawns one vLLM process per DP rank
    (external-launcher pattern). Combined TP/PP+DP uses a single ``LLM()``.
    """

    def __init__(self) -> None:
        self._llm: Optional[LLM] = None
        self._lora_request: Optional[Any] = None
        self._dp_size: int = 1
        self._dp_processes: List[Any] = []
        self._dp_active_ranks: List[int] = []
        self._dp_task_queues: List[Any] = []
        self._dp_result_queue: Any = None
        self._dp_ready_queue: Any = None
        self._dp_error_queue: Any = None
        self._dp_ctx: Any = None
        self._use_dp_workers: bool = False

    @property
    def available(self) -> bool:
        return _VLLM_AVAILABLE

    @property
    def uses_data_parallel(self) -> bool:
        return self._dp_size > 1

    @staticmethod
    def _needs_dp_worker_launcher(dp: int, tp: int, pp: int) -> bool:
        """Pure DP (TP=PP=1) needs multi-process external launcher in vLLM."""
        return dp > 1 and tp == 1 and pp == 1

    def setup_model(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enforce_eager: bool = False,
        dtype: str = "auto",
        lora_path: Optional[str] = None,
        max_lora_rank: int = 64,
        lora_name: str = "finetuned",
        lora_int_id: int = 1,
        dp_num_nodes: int = 1,
        dp_node_rank: int = 0,
        dp_master_addr: str = "",
        dp_master_port: int = 0,
        dp_worker_timeout_s: int = 3600,
        language_model_only: bool = False,
    ) -> Optional[Any]:
        if not self.available:
            print("[VLLMEngine] vLLM is not installed.")
            return None

        torch.cuda.empty_cache()
        n_gpus = max(torch.cuda.device_count(), 1)
        tp = tensor_parallel_size if tensor_parallel_size is not None else n_gpus
        pp = pipeline_parallel_size
        dp = data_parallel_size
        self._dp_size = dp
        self._dp_worker_timeout_s = dp_worker_timeout_s
        self._use_dp_workers = self._needs_dp_worker_launcher(dp, tp, pp)

        if max_model_len is None:
            max_model_len = _read_max_position_embeddings(model_path)

        lora_resolved: Optional[str] = None
        if lora_path:
            lora_resolved = str(Path(lora_path).expanduser().resolve())
            if not Path(lora_resolved).exists():
                print(f"[VLLMEngine] Error: LoRA path does not exist: {lora_resolved}")
                return None
            self._lora_request = LoRARequest(
                lora_name, lora_int_id, lora_resolved
            )
        else:
            self._lora_request = None

        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        print(
            f"[VLLMEngine] Loading {model_path}  "
            f"TP={tp} PP={pp} DP={dp}  max_len={max_model_len}  "
            f"dtype={dtype}  enforce_eager={enforce_eager}  "
            f"language_model_only={language_model_only}"
        )
        if self._lora_request:
            print(
                f"[VLLMEngine] LoRA enabled  rank={max_lora_rank}  "
                f"path={self._lora_request.lora_path}"
            )

        if self._use_dp_workers:
            print("[VLLMEngine] Using multi-process DP launcher (TP=PP=1).")
            return self._setup_data_parallel(
                model_path=model_path,
                tp=tp,
                pp=pp,
                dp=dp,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                enforce_eager=enforce_eager,
                dtype=dtype,
                lora_resolved=lora_resolved,
                max_lora_rank=max_lora_rank,
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                dp_num_nodes=dp_num_nodes,
                dp_node_rank=dp_node_rank,
                dp_master_addr=dp_master_addr,
                dp_master_port=dp_master_port,
                language_model_only=language_model_only,
            )

        llm_kwargs: dict = _language_model_only_kwargs(language_model_only)
        if pp != 1:
            llm_kwargs["pipeline_parallel_size"] = pp
        if dp != 1:
            llm_kwargs["data_parallel_size"] = dp
        if self._lora_request is not None:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = max_lora_rank

        try:
            self._llm = LLM(
                model=model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
                enforce_eager=enforce_eager,
                dtype=dtype,
                **llm_kwargs,
            )
            print("[VLLMEngine] Model loaded.")
            return self._llm
        except Exception as exc:
            import traceback
            print(f"[VLLMEngine] Failed to load model: {exc}")
            traceback.print_exc()
            return None

    def _setup_data_parallel(
        self,
        *,
        model_path: str,
        tp: int,
        pp: int,
        dp: int,
        gpu_memory_utilization: float,
        max_model_len: int,
        enforce_eager: bool,
        dtype: str,
        lora_resolved: Optional[str],
        max_lora_rank: int,
        lora_name: str,
        lora_int_id: int,
        dp_num_nodes: int,
        dp_node_rank: int,
        dp_master_addr: str,
        dp_master_port: int,
        language_model_only: bool,
    ) -> Optional[str]:
        if dp % dp_num_nodes != 0:
            print(
                f"[VLLMEngine] Error: data_parallel_size ({dp}) must be divisible "
                f"by dp_num_nodes ({dp_num_nodes})."
            )
            return None

        dp_per_node = dp // dp_num_nodes
        if dp_node_rank * dp_per_node >= dp:
            print(
                f"[VLLMEngine] Error: dp_node_rank ({dp_node_rank}) out of range "
                f"for dp_size={dp}, dp_num_nodes={dp_num_nodes}."
            )
            return None

        if dp_num_nodes == 1:
            dp_master_ip = "127.0.0.1"
            dp_master_port_val = _vllm_get_open_port()
            torch_master_port_val = _vllm_get_open_port()
        else:
            if not dp_master_addr:
                print(
                    "[VLLMEngine] Error: --dp-master-addr is required when "
                    "--dp-num-nodes > 1."
                )
                return None
            dp_master_ip = dp_master_addr
            dp_master_port_val = dp_master_port or _vllm_get_open_port()
            torch_master_port_val = _vllm_get_open_port()

        print(
            f"[VLLMEngine] Starting {dp_per_node} DP worker process(es) on this node "
            f"(global DP={dp}, dp_master={dp_master_ip}:{dp_master_port_val}, "
            f"torch_master={dp_master_ip}:{torch_master_port_val})…"
        )

        worker_config = {
            "model": model_path,
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "trust_remote_code": True,
            "enforce_eager": enforce_eager,
            "dtype": dtype,
            "data_parallel_size": dp,
            "distributed_executor_backend": "external_launcher",
            "lora_path": lora_resolved,
            "lora_name": lora_name,
            "lora_int_id": lora_int_id,
            "max_lora_rank": max_lora_rank,
            **_language_model_only_kwargs(language_model_only),
        }
        if pp != 1:
            worker_config["pipeline_parallel_size"] = pp

        self._dp_ctx = get_context("spawn")
        self._dp_task_queues = [self._dp_ctx.Queue() for _ in range(dp)]
        self._dp_result_queue = self._dp_ctx.Queue()
        self._dp_ready_queue = self._dp_ctx.Queue()
        self._dp_error_queue = self._dp_ctx.Queue()
        self._dp_processes = []

        global_ranks = list(range(
            dp_node_rank * dp_per_node,
            (dp_node_rank + 1) * dp_per_node,
        ))
        self._dp_active_ranks = global_ranks
        for local_dp_rank, global_dp_rank in enumerate(global_ranks):
            proc = self._dp_ctx.Process(
                target=vllm_dp_worker_entry,
                args=(
                    global_dp_rank,
                    local_dp_rank,
                    dp,
                    tp,
                    pp,
                    dp_master_ip,
                    dp_master_port_val,
                    torch_master_port_val,
                    worker_config,
                    self._dp_task_queues[global_dp_rank],
                    self._dp_result_queue,
                    self._dp_ready_queue,
                    self._dp_error_queue,
                ),
                name=f"vllm-dp-rank-{global_dp_rank}",
            )
            proc.start()
            self._dp_processes.append(proc)

        try:
            for _ in global_ranks:
                ready_rank = self._dp_ready_queue.get(timeout=self._dp_worker_timeout_s)
                print(f"[VLLMEngine] DP worker rank {ready_rank} ready.")
        except Exception as exc:
            self._report_dp_errors()
            self.shutdown()
            print(f"[VLLMEngine] DP worker startup failed: {exc}")
            return None

        for proc in self._dp_processes:
            if proc.exitcode not in (None, 0):
                self._report_dp_errors()
                self.shutdown()
                print(
                    f"[VLLMEngine] DP worker pid={proc.pid} exited with "
                    f"code {proc.exitcode} during startup."
                )
                return None

        print("[VLLMEngine] All DP workers loaded.")
        return "data_parallel"

    def _report_dp_errors(self) -> None:
        if self._dp_error_queue is None:
            return
        while not self._dp_error_queue.empty():
            rank, msg = self._dp_error_queue.get_nowait()
            print(f"[VLLMEngine] DP worker rank {rank} error: {msg}")

    def shutdown(self) -> None:
        if not self._use_dp_workers:
            self._llm = None
            return

        for queue in self._dp_task_queues:
            try:
                queue.put(None)
            except Exception:
                pass

        for proc in self._dp_processes:
            proc.join(timeout=30)
            if proc.is_alive():
                print(f"[VLLMEngine] Killing stuck DP worker pid={proc.pid}")
                proc.kill()
                proc.join(timeout=5)

        self._dp_processes = []
        self._dp_active_ranks = []
        self._dp_task_queues = []
        self._dp_result_queue = None
        self._dp_ready_queue = None
        self._dp_error_queue = None
        self._dp_ctx = None

    def run_inference(
        self,
        prompts: List[str],
        batch_size: int,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> List[Any]:
        if self._use_dp_workers:
            return self._dp_run_inference(
                prompts, batch_size, max_tokens, temperature
            )
        if self._llm is None:
            return []
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        gen_kw: dict = {}
        if self._lora_request is not None:
            gen_kw["lora_request"] = self._lora_request

        results = []
        for i in range(0, len(prompts), batch_size):
            results.extend(
                self._llm.generate(
                    prompts[i:i + batch_size],
                    params,
                    **gen_kw,
                )
            )
        return results

    def _dp_run_inference(
        self,
        prompts: List[str],
        batch_size: int,
        max_tokens: int,
        temperature: float,
    ) -> List[Any]:
        if not self._dp_processes:
            return []

        active_ranks = self._dp_active_ranks
        pending = len(active_ranks)
        for rank in active_ranks:
            start, end = shard_range(len(prompts), self._dp_size, rank)
            shard = prompts[start:end]
            placeholder = False
            if not shard:
                shard = [" "]
                placeholder = True
            self._dp_task_queues[rank].put(
                (start, shard, batch_size, max_tokens, temperature, placeholder)
            )

        merged: dict[int, Any] = {}
        while pending:
            start_idx, outputs, placeholder = self._dp_result_queue.get(
                timeout=self._dp_worker_timeout_s
            )
            if not placeholder:
                for offset, out in enumerate(outputs):
                    merged[start_idx + offset] = out
            pending -= 1

        self._check_dp_processes()
        return [merged[i] for i in range(len(prompts)) if i in merged]

    def _check_dp_processes(self) -> None:
        for proc in self._dp_processes:
            if proc.exitcode not in (None, 0):
                self._report_dp_errors()
                raise RuntimeError(
                    f"DP worker pid={proc.pid} exited with code {proc.exitcode}"
                )

    def run_benchmark(
        self,
        prompts: List[str],
        num_samples: int,
        batch_size: int,
        max_tokens: int,
    ) -> Tuple[List[Any], float, float]:
        if self._use_dp_workers and not self._dp_processes:
            return [], 0.0, 0.0
        if not self._use_dp_workers and self._llm is None:
            return [], 0.0, 0.0

        full: List[str] = []
        while len(full) < num_samples:
            full.extend(prompts)
        full = full[:num_samples]

        t0 = time.time()
        all_outputs = self.run_inference(full, batch_size, max_tokens)
        t1 = time.time()
        return all_outputs, t0, t1

    def estimate_tokens(self, outputs: List[Any]) -> int:
        total = 0
        for out in outputs:
            if hasattr(out, "outputs") and out.outputs:
                total += int(len(out.outputs[0].text.split()) * 1.3)
        return total


def _language_model_only_kwargs(language_model_only: bool) -> dict:
    """Pass through to vLLM only when enabled (older vLLM may lack the flag)."""
    if language_model_only:
        return {"language_model_only": True}
    return {}


def _read_max_position_embeddings(model_path: str, default: int = 2048) -> int:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return getattr(cfg, "max_position_embeddings", default)
    except Exception:
        return default
