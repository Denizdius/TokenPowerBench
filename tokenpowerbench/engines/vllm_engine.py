"""vLLM inference with optional serial, host-observed phase profiling."""

from __future__ import annotations

import importlib.util
import gc
import math
import sys
import time
import uuid
from typing import Any, List, Optional, Tuple

from .base import InferenceEngine
from ..phases import PhaseTimingError, build_phase_event


class VLLMEngine(InferenceEngine):
    """Single-node vLLM wrapper; CUDA dependencies load only when used."""

    def __init__(self) -> None:
        self._llm: Optional[Any] = None
        self._phase_profiling = False
        self._seed = 0
        self._temperature = 0.7
        self.resolved_config: dict = {}
        self._lora_request: Optional[Any] = None
        self._dp_size = 1
        self._use_dp_workers = False
        self._dp_worker_timeout_s = 3600
        self._dp_processes: List[Any] = []
        self._dp_active_ranks: List[int] = []
        self._dp_task_queues: List[Any] = []
        self._dp_result_queue: Any = None
        self._dp_ready_queue: Any = None
        self._dp_error_queue: Any = None
        self._dp_ctx: Any = None

    @property
    def available(self) -> bool:
        return importlib.util.find_spec("vllm") is not None

    def setup_model(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        *,
        phase_profiling: bool = False,
        seed: int = 0,
        temperature: float = 0.7,
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
    ) -> Any:
        """Load vLLM, with isolated scheduling when phase profiling is enabled.

        Phase mode requests disabled prefix caching and chunked prefill and
        admits one sequence at a time. Effective settings are recorded because
        some vLLM versions override requested settings. Observed boundaries
        include host/scheduler overhead and cannot establish exact GPU kernel
        execution intervals.
        """
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not 0 < gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        if max_model_len is not None:
            _positive_int("max_model_len", max_model_len)
        if tensor_parallel_size is not None:
            _positive_int("tensor_parallel_size", tensor_parallel_size)
        _positive_int("pipeline_parallel_size", pipeline_parallel_size)
        _positive_int("data_parallel_size", data_parallel_size)
        _positive_int("dp_num_nodes", dp_num_nodes)
        if isinstance(dp_node_rank, bool) or not isinstance(dp_node_rank, int) or dp_node_rank < 0:
            raise ValueError("dp_node_rank must be a nonnegative integer")
        if data_parallel_size % dp_num_nodes != 0:
            raise ValueError("data_parallel_size must be divisible by dp_num_nodes")
        if phase_profiling and data_parallel_size > 1:
            raise ValueError("Phase profiling does not support data_parallel_size > 1")
        if not self.available:
            raise RuntimeError("vLLM is not installed. Install tokenpowerbench[vllm].")

        import torch
        from vllm import LLM

        torch.cuda.empty_cache()
        tp = tensor_parallel_size or max(torch.cuda.device_count(), 1)
        self._dp_size = data_parallel_size
        self._dp_worker_timeout_s = dp_worker_timeout_s
        self._use_dp_workers = data_parallel_size > 1
        self._seed = seed
        self._temperature = temperature
        self._phase_profiling = phase_profiling
        if self._use_dp_workers:
            return self._setup_data_parallel(
                model_path=model_path, tp=tp, pp=pipeline_parallel_size, dp=data_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                enforce_eager=enforce_eager, dtype=dtype, lora_path=lora_path,
                max_lora_rank=max_lora_rank, lora_name=lora_name, lora_int_id=lora_int_id,
                dp_num_nodes=dp_num_nodes, dp_node_rank=dp_node_rank,
                dp_master_addr=dp_master_addr, dp_master_port=dp_master_port,
                language_model_only=language_model_only, seed=seed,
            )
        kwargs = {
            "model": model_path,
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": gpu_memory_utilization,
            "seed": seed,
            "trust_remote_code": True,
        }
        if pipeline_parallel_size != 1:
            kwargs["pipeline_parallel_size"] = pipeline_parallel_size
        if enforce_eager:
            kwargs["enforce_eager"] = True
        if dtype != "auto":
            kwargs["dtype"] = dtype
        if language_model_only:
            kwargs["language_model_only"] = True
        if lora_path:
            from pathlib import Path
            resolved = str(Path(lora_path).expanduser().resolve())
            if not Path(resolved).exists():
                raise ValueError(f"LoRA path does not exist: {resolved}")
            from vllm.lora.request import LoRARequest
            kwargs["enable_lora"] = True
            kwargs["max_lora_rank"] = max_lora_rank
            self._lora_request = LoRARequest(lora_name, lora_int_id, resolved)
        else:
            self._lora_request = None
        if max_model_len is not None:
            kwargs["max_model_len"] = max_model_len
        if phase_profiling:
            kwargs.update(
                enable_prefix_caching=False,
                enable_chunked_prefill=False,
                max_num_seqs=1,
            )

        self._llm = LLM(**kwargs)
        self.resolved_config = _engine_config(self._llm, kwargs)
        self._phase_profiling = phase_profiling
        self._seed = seed
        self._temperature = temperature
        return self._llm

    def shutdown(self) -> None:
        """Release data-parallel workers and the in-process engine."""
        self.close()

    def close(self) -> None:
        """Shut down data-parallel workers and the vLLM engine core.

        V1 exposes synchronous shutdown on its engine-core client. V0 uses
        model_executor.shutdown(), also called by its LLMEngine destructor.
        Both paths are inspected against vLLM 0.10.1.1. Collection after
        shutdown releases in-process model references before emptying any
        initialized CUDA allocator cache.
        """
        self._shutdown_dp_workers()
        llm = self._llm
        if llm is None:
            return
        self._llm = None
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        shutdown = getattr(core, "shutdown", None)
        executor = None
        if not callable(shutdown):
            executor = getattr(engine, "model_executor", None)
            shutdown = getattr(executor, "shutdown", None)
        try:
            if not callable(shutdown):
                raise RuntimeError("This vLLM engine exposes neither engine_core.shutdown() nor model_executor.shutdown()")
            shutdown()
        finally:
            if executor is not None:
                # Its destructor calls the same method; avoid a second shutdown.
                engine.model_executor = None
            llm = engine = core = executor = shutdown = None
            gc.collect()
            cuda = getattr(sys.modules.get("torch"), "cuda", None)
            is_initialized = getattr(cuda, "is_initialized", None)
            if callable(is_initialized) and is_initialized():
                cuda.empty_cache()

    def _sampling_params(
        self, max_tokens: int, temperature: Optional[float] = None, *, profile: bool = False
    ) -> Any:
        from vllm import SamplingParams

        kwargs = {
            "max_tokens": max_tokens,
            "temperature": self._temperature if temperature is None else temperature,
            "seed": self._seed,
            "n": 1,
        }
        if profile:
            from vllm.sampling_params import RequestOutputKind

            kwargs["output_kind"] = RequestOutputKind.CUMULATIVE
        return SamplingParams(**kwargs)

    def _require_model(self) -> Any:
        if self._llm is None:
            raise RuntimeError("Call setup_model before running inference.")
        return self._llm

    def run_inference(
        self,
        prompts: List[str],
        batch_size: int,
        max_tokens: int = 200,
        temperature: Optional[float] = None,
    ) -> List[Any]:
        _validate_workload(prompts, len(prompts), batch_size, max_tokens)
        if temperature is not None and (not math.isfinite(temperature) or temperature < 0):
            raise ValueError("temperature must be finite and nonnegative")
        if self._use_dp_workers:
            used_temperature = self._temperature if temperature is None else temperature
            return self._dp_run_inference(prompts, batch_size, max_tokens, used_temperature)
        llm = self._require_model()
        params = self._sampling_params(max_tokens, temperature)
        results = []
        generate_kwargs = {}
        if self._lora_request is not None:
            generate_kwargs["lora_request"] = self._lora_request
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            outputs = llm.generate(batch, params, **generate_kwargs)
            if len(outputs) != len(batch):
                raise RuntimeError("vLLM returned a different number of responses than submitted prompts.")
            results.extend(outputs)
        return results

    def run_benchmark(
        self,
        prompts: List[str],
        num_samples: int,
        batch_size: int,
        max_tokens: int,
    ) -> Tuple[List[Any], float, float]:
        _validate_workload(prompts, num_samples, batch_size, max_tokens)
        if self._use_dp_workers:
            if not self._dp_processes:
                raise RuntimeError("Call setup_model before running inference.")
        else:
            self._require_model()
        full = [prompts[i % len(prompts)] for i in range(num_samples)]
        all_outputs: List[Any] = []
        t0 = time.perf_counter()
        for i in range(0, num_samples, batch_size):
            batch = full[i:i + batch_size]
            all_outputs.extend(self.run_inference(batch, batch_size, max_tokens))
        t1 = time.perf_counter()
        return all_outputs, t0, t1

    def run_profiled_benchmark(
        self,
        prompts: List[str],
        num_samples: int,
        batch_size: int,
        max_tokens: int,
    ) -> Tuple[List[Any], float, float, List[dict]]:
        """Observe each request's first token using incremental engine steps.

        Each step timestamp is taken after the synchronous engine call returns.
        These are host-observed proxies, not GPU kernel timestamps. A request
        is completed before submitting the next to avoid mixed request phases.
        Engines that coalesce the first output tokens are rejected.
        """
        _validate_workload(prompts, num_samples, batch_size, max_tokens)
        llm = self._require_model()
        if batch_size != 1:
            raise ValueError("Phase profiling requires batch_size=1 for isolated requests.")
        if not self._phase_profiling:
            raise PhaseTimingError("Load the model with phase_profiling=True first.")
        engine = getattr(llm, "llm_engine", None)
        methods = ("add_request", "step", "has_unfinished_requests", "abort_request")
        if engine is None or any(not callable(getattr(engine, name, None)) for name in methods):
            raise PhaseTimingError("This vLLM version lacks the required incremental engine API.")
        if engine.has_unfinished_requests():
            raise PhaseTimingError("Phase profiling requires an idle engine.")

        full = [prompts[i % len(prompts)] for i in range(num_samples)]
        params = self._sampling_params(max_tokens, profile=True)
        run_id = uuid.uuid4().hex
        all_outputs, events = [], []
        t0 = time.perf_counter()
        for index, prompt in enumerate(full):
            request_id = f"tpbench-{run_id}-{index}"
            submitted = time.perf_counter()
            first_token = None
            final_output = None
            previous_ids: tuple = ()
            try:
                engine.add_request(request_id, prompt, params)
                # V1 may already be executing in a background process by now.
                # The submission timestamp is the start of the prefill proxy.
                dispatch_completed = time.perf_counter()
                while engine.has_unfinished_requests():
                    step_outputs = engine.step()
                    observed = time.perf_counter()
                    for output in step_outputs:
                        if str(getattr(output, "request_id", "")) != request_id:
                            raise PhaseTimingError("Unexpected concurrent request in phase profiling.")
                        completions = getattr(output, "outputs", None)
                        if not completions:
                            if getattr(output, "finished", False):
                                raise PhaseTimingError("Request finished without generated token IDs.")
                            continue
                        if len(completions) != 1:
                            raise PhaseTimingError("Phase profiling requires one completion per request.")
                        token_ids = getattr(completions[0], "token_ids", None)
                        if token_ids is None:
                            raise PhaseTimingError("vLLM output does not expose generated token IDs.")
                        current_ids = tuple(token_ids)
                        if current_ids[:len(previous_ids)] != previous_ids:
                            raise PhaseTimingError("Engine outputs are not cumulative token IDs.")
                        if current_ids and first_token is None:
                            if len(current_ids) != 1:
                                raise PhaseTimingError(
                                    "First engine observation contains multiple tokens; "
                                    "the prefill/decode boundary cannot be measured."
                                )
                            first_token = observed
                        previous_ids = current_ids
                        if getattr(output, "finished", False):
                            if first_token is None:
                                raise PhaseTimingError("Request finished without a first token.")
                            if final_output is not None:
                                raise PhaseTimingError("Duplicate finished output for a request.")
                            input_ids = getattr(output, "prompt_token_ids", None)
                            if input_ids is None:
                                raise PhaseTimingError("vLLM output does not expose prompt token IDs.")
                            event = build_phase_event(
                                request_id=request_id,
                                submitted_s=submitted,
                                dispatch_completed_s=dispatch_completed,
                                first_token_s=first_token,
                                finished_s=observed,
                                input_tokens=len(input_ids),
                                output_tokens=len(current_ids),
                            )
                            final_output = output
                    if final_output is not None:
                        if engine.has_unfinished_requests():
                            raise PhaseTimingError("Engine is not idle after the profiled request finished.")
                        break
                if final_output is None:
                    raise PhaseTimingError("Engine stopped without a completed request output.")
            except BaseException:
                try:
                    engine.abort_request([request_id])
                except Exception:
                    pass
                raise
            all_outputs.append(final_output)
            events.append(event)
        t1 = time.perf_counter()
        return all_outputs, t0, t1, events

    def estimate_tokens(self, outputs: List[Any]) -> int:
        """Count generated token IDs exactly; retained name for compatibility."""
        total = 0
        for output in outputs:
            completions = getattr(output, "outputs", None)
            if not completions:
                raise ValueError("Cannot count tokens: response has no completions.")
            for completion in completions:
                token_ids = getattr(completion, "token_ids", None)
                if token_ids is None:
                    raise ValueError("Cannot count tokens: response has no generated token IDs.")
                total += len(token_ids)
        return total

    def _setup_data_parallel(
        self, *, model_path, tp, pp, dp, gpu_memory_utilization, max_model_len,
        enforce_eager, dtype, lora_path, max_lora_rank, lora_name, lora_int_id,
        dp_num_nodes, dp_node_rank, dp_master_addr, dp_master_port, language_model_only, seed,
    ):
        """Spawn one worker per DP rank. TP=PP=1 uses coordinated vLLM DP."""
        from multiprocessing import get_context
        from pathlib import Path

        from .vllm_dp_worker import cuda_visible_devices_for_replica, vllm_dp_worker_entry

        dp_per_node = dp // dp_num_nodes
        if dp_node_rank * dp_per_node >= dp:
            raise ValueError("dp_node_rank is outside the data-parallel group")
        use_coordinated = tp == 1 and pp == 1
        if dp_num_nodes == 1:
            master_ip = "127.0.0.1"
            master_port = _open_port()
            torch_port = _open_port()
        else:
            if not dp_master_addr:
                raise ValueError("dp_master_addr is required when dp_num_nodes > 1")
            master_ip = dp_master_addr
            master_port = dp_master_port or _open_port()
            torch_port = _open_port()
        lora_resolved = None
        if lora_path:
            lora_resolved = str(Path(lora_path).expanduser().resolve())
            if not Path(lora_resolved).exists():
                raise ValueError(f"LoRA path does not exist: {lora_resolved}")
        worker_config = {
            "model": model_path,
            "tensor_parallel_size": tp,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": True,
            "enforce_eager": enforce_eager,
            "dtype": dtype,
            "seed": seed,
            "lora_path": lora_resolved,
            "lora_name": lora_name,
            "lora_int_id": lora_int_id,
            "max_lora_rank": max_lora_rank,
        }
        if max_model_len is not None:
            worker_config["max_model_len"] = max_model_len
        if pp != 1:
            worker_config["pipeline_parallel_size"] = pp
        if language_model_only:
            worker_config["language_model_only"] = True
        if use_coordinated:
            worker_config["data_parallel_size"] = dp
            worker_config["distributed_executor_backend"] = "external_launcher"

        self._dp_ctx = get_context("spawn")
        self._dp_task_queues = [self._dp_ctx.Queue() for _ in range(dp)]
        self._dp_result_queue = self._dp_ctx.Queue()
        self._dp_ready_queue = self._dp_ctx.Queue()
        self._dp_error_queue = self._dp_ctx.Queue()
        self._dp_processes = []
        global_ranks = list(range(dp_node_rank * dp_per_node, (dp_node_rank + 1) * dp_per_node))
        self._dp_active_ranks = global_ranks
        for local_rank, global_rank in enumerate(global_ranks):
            visible = None
            if not use_coordinated:
                visible = cuda_visible_devices_for_replica(global_rank, tp, pp)
            process = self._dp_ctx.Process(
                target=vllm_dp_worker_entry,
                args=(
                    global_rank, local_rank, dp, use_coordinated, visible,
                    master_ip, master_port, torch_port, worker_config,
                    self._dp_task_queues[global_rank], self._dp_result_queue,
                    self._dp_ready_queue, self._dp_error_queue,
                ),
                name=f"vllm-dp-rank-{global_rank}",
            )
            process.start()
            self._dp_processes.append(process)
        try:
            for _ in global_ranks:
                ready_rank = self._dp_ready_queue.get(timeout=self._dp_worker_timeout_s)
                print(f"[VLLMEngine] DP worker rank {ready_rank} ready.")
        except Exception as exc:
            self._report_dp_errors()
            self._shutdown_dp_workers()
            raise RuntimeError(f"DP worker startup failed: {exc}") from exc
        for process in self._dp_processes:
            if process.exitcode not in (None, 0):
                self._report_dp_errors()
                self._shutdown_dp_workers()
                raise RuntimeError(f"DP worker pid={process.pid} exited with code {process.exitcode}")
        print("[VLLMEngine] All DP workers loaded.")
        return "data_parallel"

    def _report_dp_errors(self) -> None:
        if self._dp_error_queue is None:
            return
        while not self._dp_error_queue.empty():
            rank, message = self._dp_error_queue.get_nowait()
            print(f"[VLLMEngine] DP worker rank {rank} error: {message}")

    def _shutdown_dp_workers(self) -> None:
        for queue in self._dp_task_queues:
            try:
                queue.put(None)
            except Exception:
                pass
        for process in self._dp_processes:
            process.join(timeout=30)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        self._dp_processes = []
        self._dp_active_ranks = []
        self._dp_task_queues = []
        self._dp_result_queue = None
        self._dp_ready_queue = None
        self._dp_error_queue = None
        self._dp_ctx = None
        self._use_dp_workers = False

    def _dp_run_inference(self, prompts, batch_size, max_tokens, temperature):
        from .vllm_dp_worker import shard_range

        if not self._dp_processes:
            raise RuntimeError("Call setup_model before running inference.")
        pending = len(self._dp_active_ranks)
        for rank in self._dp_active_ranks:
            start, end = shard_range(len(prompts), self._dp_size, rank)
            shard = prompts[start:end]
            placeholder = False
            if not shard:
                shard = [" "]
                placeholder = True
            self._dp_task_queues[rank].put(
                (start, shard, batch_size, max_tokens, temperature, placeholder)
            )
        merged = {}
        while pending:
            start_idx, outputs, placeholder = self._dp_result_queue.get(timeout=self._dp_worker_timeout_s)
            if not placeholder:
                for offset, output in enumerate(outputs):
                    merged[start_idx + offset] = output
            pending -= 1
        for process in self._dp_processes:
            if process.exitcode not in (None, 0):
                self._report_dp_errors()
                raise RuntimeError(f"DP worker pid={process.pid} exited with code {process.exitcode}")
        return [merged[index] for index in range(len(prompts)) if index in merged]


def _language_model_only_kwargs(language_model_only: bool) -> dict:
    """Pass the flag through only when enabled; older vLLM builds may lack it."""
    if language_model_only:
        return {"language_model_only": True}
    return {}


def _read_max_position_embeddings(model_path: str, default: int = 2048) -> int:
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        return int(getattr(config, "max_position_embeddings", default))
    except Exception:
        return default


def _open_port() -> int:
    try:
        from vllm.utils.network_utils import get_open_port
        return int(get_open_port())
    except Exception:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            return int(sock.getsockname()[1])


def _engine_config(llm: Any, requested: dict) -> dict:
    """Keep requested and effective vLLM settings distinct in run artifacts."""
    engine = getattr(llm, "llm_engine", None)
    config = getattr(engine, "vllm_config", None)
    fields = {
        "scheduler_config": ("enable_chunked_prefill", "max_num_seqs", "max_num_batched_tokens"),
        "cache_config": ("enable_prefix_caching", "gpu_memory_utilization"),
        "model_config": ("max_model_len",),
        "parallel_config": ("tensor_parallel_size",),
    }
    effective = {}
    for section, names in fields.items():
        values = getattr(config, section, None)
        if values is None:
            # Older engines expose these configs directly on LLMEngine.
            values = getattr(engine, section, None)
        for name in names:
            effective[name] = getattr(values, name, None)
    return {
        "requested": dict(requested),
        "effective": effective,
        "engine_class": None if engine is None else f"{type(engine).__module__}.{type(engine).__qualname__}",
    }


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _validate_workload(prompts: List[str], num_samples: int, batch_size: int, max_tokens: int) -> None:
    for name, value in (("num_samples", num_samples), ("batch_size", batch_size), ("max_tokens", max_tokens)):
        _positive_int(name, value)
    if not prompts or any(not isinstance(prompt, str) or not prompt.strip() for prompt in prompts):
        raise ValueError("prompts must contain nonempty strings")
