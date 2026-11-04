"""
VLLMRayDataEngine — multi-node vLLM via Ray Data LLM processors.

Uses the current Ray Data API:

    from ray.data.llm import vLLMEngineProcessorConfig, build_processor

This is a parallel path to VLLMDistributedEngine (driver-side LLM).
It does not replace run_multi_node.py.

Cross-node TP/PP is requested with:
    engine_kwargs["distributed_executor_backend"] = "ray"
and a PACK placement group of TP×PP×DP GPU bundles.
"""

from __future__ import annotations

import inspect
import os
import time
from typing import Any, Callable, Dict, List, Optional

import ray
from packaging.version import Version

from .ray_cluster import RayClusterConfig
from tokenpowerbench.engines.vllm_engine import (
    _language_model_only_kwargs,
    _read_max_position_embeddings,
)

_MIN_RAY_VERSION = "2.22.0"


def _import_ray_data_llm():
    try:
        import ray.data.llm as ray_llm
        from ray.data.llm import vLLMEngineProcessorConfig
    except ImportError as exc:
        raise ImportError(
            "ray.data.llm.vLLMEngineProcessorConfig is not available. "
            "Need a Ray build with the Data LLM extra "
            "(e.g. vllm-v0.27.1-cu130-official-ray-support.sif)."
        ) from exc

    build_fn = getattr(ray_llm, "build_processor", None) or getattr(
        ray_llm, "build_llm_processor", None
    )
    if build_fn is None:
        raise ImportError(
            "Neither build_processor nor build_llm_processor is available "
            "in ray.data.llm."
        )
    return vLLMEngineProcessorConfig, build_fn


def _filter_ctor_kwargs(cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    allowed = {name for name in params if name != "self"}
    dropped = sorted(k for k in kwargs if k not in allowed)
    if dropped:
        print(f"[VLLMRayDataEngine] Skipping unsupported config keys: {dropped}")
    return {k: v for k, v in kwargs.items() if k in allowed}


class VLLMRayDataEngine:
    """Ray Data LLM processor: vLLMEngineProcessorConfig + build_processor."""

    def __init__(self, cluster: RayClusterConfig, config: Dict[str, Any]) -> None:
        self.cluster = cluster
        self.config = config
        self.model_path = config["model_path"]
        self.tensor_parallel_size = config["tensor_parallel_size"]
        self.pipeline_parallel_size = config["pipeline_parallel_size"]
        self.data_parallel_size = int(config.get("data_parallel_size", 1))
        self.concurrency = int(config["concurrency"])
        self.batch_size = config["batch_size"]
        self.gpu_memory_utilization = float(config.get("gpu_memory_utilization", 0.90))
        self.max_model_len = config.get("max_model_len")
        if self.max_model_len is None:
            self.max_model_len = _read_max_position_embeddings(self.model_path)
        self.enforce_eager = bool(config.get("enforce_eager", False))
        self.language_model_only = bool(config.get("language_model_only", False))
        self.dtype = config.get("dtype", "auto")
        self.warmup = bool(config.get("warmup", True))
        self.verbose = config.get("verbose", False)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 512)
        self._gpus_per_node = 0
        self._processor: Optional[Callable] = None

    def prepare(self) -> bool:
        """Ray connect + processor + warmup. Not timed. Monitor must be OFF."""
        try:
            if not self._init_ray():
                return False
            self._processor = self._make_processor()
            print(
                f"[VLLMRayDataEngine] Processor object id={id(self._processor)}. "
                "Generate must reuse this object."
            )
            if self.warmup:
                print(
                    "[VLLMRayDataEngine] Warmup on the same processor object "
                    "(not a second engine)."
                )
                warm = ray.data.from_items(
                    [{"prompt": "warmup", "row_id": -1}]
                )
                self._processor(warm).take_all()
                print(
                    "[VLLMRayDataEngine] Warmup complete. If generate later logs "
                    "another LLM load, Ray Data reloaded the engine — do not use "
                    "those numbers next to single-node 14B/27B."
                )
            return True
        except Exception as exc:
            print(f"[VLLMRayDataEngine] Prepare failed: {exc}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return False

    def run_generate(self, prompts: List[str]) -> Optional[Dict[str, Any]]:
        """Generation only (timed). Call after prepare(); shutdown after monitor.stop()."""
        try:
            return self._run_inference(prompts)
        except Exception as exc:
            print(f"[VLLMRayDataEngine] Benchmark failed: {exc}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def shutdown(self) -> None:
        self._processor = None

    def run_benchmark(self, prompts: List[str]) -> Optional[Dict[str, Any]]:
        """Prepare + generate, no shutdown (caller owns monitor/shutdown)."""
        if not self.prepare():
            return None
        return self.run_generate(prompts)

    def _init_ray(self) -> bool:
        if Version(ray.__version__) < Version(_MIN_RAY_VERSION):
            print(
                f"[VLLMRayDataEngine] Ray >= {_MIN_RAY_VERSION} required, "
                f"found {ray.__version__}"
            )
            return False

        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

        if not ray.is_initialized():
            print(
                f"[VLLMRayDataEngine] Connecting to Ray at "
                f"{self.cluster.ray_init_address}"
            )
            ray.init(
                address=self.cluster.ray_init_address,
                ignore_reinit_error=True,
            )
        else:
            print("[VLLMRayDataEngine] Ray already initialized.")

        time.sleep(2)

        total_gpus = 0
        alive: List[Any] = []
        for attempt in range(5):
            try:
                nodes = ray.nodes()
                alive = [n for n in nodes if n["Alive"]]
                resources = ray.cluster_resources()
                node_gpus = sum(
                    int(n.get("Resources", {}).get("GPU", 0)) for n in alive
                )
                total_gpus = max(int(resources.get("GPU", 0)), node_gpus)
                if total_gpus > 0:
                    break
                print(
                    f"[VLLMRayDataEngine] No GPUs detected "
                    f"(attempt {attempt + 1}/5), retrying…"
                )
                time.sleep(3)
            except Exception as exc:
                print(
                    f"[VLLMRayDataEngine] Cluster query failed "
                    f"(attempt {attempt + 1}): {exc}"
                )
                time.sleep(2)

        print(
            f"[VLLMRayDataEngine] Cluster: {len(alive)} node(s), "
            f"{total_gpus} GPU(s)"
        )
        gpu_counts: List[int] = []
        for n in alive:
            hostname = n.get("NodeManagerHostname", "?")
            gpus = int(n.get("Resources", {}).get("GPU", 0))
            cpus = int(n.get("Resources", {}).get("CPU", 0))
            gpu_counts.append(gpus)
            print(f"  {hostname}: {gpus} GPUs, {cpus} CPUs")
        self._gpus_per_node = min(gpu_counts) if gpu_counts else 0

        replica_gpus = self.tensor_parallel_size * self.pipeline_parallel_size
        replica_count = self.data_parallel_size if self.data_parallel_size > 1 else self.concurrency
        required = replica_gpus * replica_count
        if self.data_parallel_size > 1 and replica_gpus > self._gpus_per_node:
            print(
                f"[VLLMRayDataEngine] DP replica does not fit on one node: "
                f"TP×PP={replica_gpus}, node has {self._gpus_per_node}."
            )
            return False
        if total_gpus < required:
            print(
                f"[VLLMRayDataEngine] Insufficient GPUs: need {required}, "
                f"have {total_gpus}. "
                f"(TP={self.tensor_parallel_size}, PP={self.pipeline_parallel_size}, "
                f"DP={self.data_parallel_size}, concurrency={self.concurrency})"
            )
            return False
        return True

    def _gpus_per_replica(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    def _replica_count(self) -> int:
        if self.data_parallel_size > 1:
            return self.data_parallel_size
        return self.concurrency

    def _engine_kwargs(self) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = dict(
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=self.enforce_eager,
            dtype=self.dtype,
            disable_custom_all_reduce=True,
            **_language_model_only_kwargs(self.language_model_only),
        )
        if self.pipeline_parallel_size != 1:
            kwargs["pipeline_parallel_size"] = self.pipeline_parallel_size
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        # Cross-node TP/PP (replica larger than one node): Ray executor.
        # Local DP replicas: default/mp executor — do not nest Ray, and never
        # pass data_parallel_size (vLLM 0.28 rejects LLM(dp=k)).
        if self.data_parallel_size == 1 and self._gpus_per_replica() > (self._gpus_per_node or 4):
            kwargs["distributed_executor_backend"] = "ray"
        return kwargs

    def _processor_config_kwargs(self) -> Dict[str, Any]:
        n_gpus = self._gpus_per_replica()
        replica_count = self._replica_count()
        strategy = "STRICT_PACK" if n_gpus <= (self._gpus_per_node or 4) else "PACK"
        concurrency = (replica_count, replica_count)

        cfg: Dict[str, Any] = dict(
            engine_kwargs=self._engine_kwargs(),
            concurrency=concurrency,
            batch_size=self.batch_size,
            chat_template_stage=False,
            runtime_env={
                "env_vars": {
                    "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                }
            },
            placement_group_config={
                "bundles": [{"GPU": 1, "CPU": 1}] * n_gpus,
                "strategy": strategy,
            },
        )
        return cfg

    def _make_processor(self) -> Callable:
        config_cls, build_fn = _import_ray_data_llm()
        raw = self._processor_config_kwargs()
        # Current docs: model_source=. Older Ray Data LLM: model=.
        model_key = "model_source"
        try:
            params = inspect.signature(config_cls).parameters
            if "model_source" not in params and "model" in params:
                model_key = "model"
        except (TypeError, ValueError):
            pass
        raw[model_key] = self.model_path

        optional_drop_order = (
            "placement_group_config",
            "chat_template_stage",
            "runtime_env",
        )
        last_error: Optional[Exception] = None
        for _ in range(len(optional_drop_order) + 1):
            filtered = _filter_ctor_kwargs(config_cls, raw)
            try:
                config = config_cls(**filtered)
                print(
                    f"[VLLMRayDataEngine] Ray Data LLM processor  "
                    f"builder={build_fn.__name__}  model_key={model_key}"
                )
                print(
                    f"  TP={self.tensor_parallel_size}  "
                    f"PP={self.pipeline_parallel_size}  "
                    f"DP={self.data_parallel_size}  "
                    f"concurrency={self.concurrency}  "
                    f"batch={self.batch_size}  "
                    f"enforce_eager={self.enforce_eager}  "
                    f"gpu_memory_utilization={self.gpu_memory_utilization}  "
                    f"max_model_len={self.max_model_len}  "
                    f"chat_template_stage={filtered.get('chat_template_stage', 'default')}"
                )
                sampling = dict(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    detokenize=True,
                )

                def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
                    return {
                        "prompt": row["prompt"],
                        "row_id": row["row_id"],
                        "sampling_params": sampling,
                    }

                def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
                    text = row.get("generated_text") or ""
                    n_tok = int(len(str(text).split()) * 1.3)
                    return {
                        "row_id": row.get("row_id"),
                        "prompt": row.get("prompt", ""),
                        "generated_text": text,
                        "tokens_generated": n_tok,
                    }

                return build_fn(
                    config,
                    preprocess=preprocess,
                    postprocess=postprocess,
                )
            except TypeError as exc:
                last_error = exc
                dropped = False
                for key in optional_drop_order:
                    if key in raw:
                        print(
                            f"[VLLMRayDataEngine] {key} not accepted by this "
                            f"Ray Data version; retrying without it."
                        )
                        raw.pop(key)
                        dropped = True
                        break
                if not dropped:
                    raise
        raise TypeError(
            f"Could not construct vLLMEngineProcessorConfig: {last_error}"
        )

    def _run_inference(self, prompts: List[str]) -> Optional[Dict[str, Any]]:
        if self._processor is None:
            print("[VLLMRayDataEngine] Processor is not loaded; call prepare() first.")
            return None
        print(
            f"[VLLMRayDataEngine] Starting Ray Data LLM inference  "
            f"model={os.path.basename(self.model_path)}  "
            f"TP={self.tensor_parallel_size}  PP={self.pipeline_parallel_size}  "
            f"DP={self.data_parallel_size}  batch={self.batch_size}  "
            f"prompts={len(prompts)}  processor_id={id(self._processor)}"
        )

        ds = ray.data.from_items(
            [{"prompt": p, "row_id": i} for i, p in enumerate(prompts)]
        )
        t0 = time.time()
        try:
            out_ds = self._processor(ds)
            rows = out_ds.take_all()
            t1 = time.time()
            return self._process_results(rows, t0, t1, prompts)
        except Exception as exc:
            print(f"[VLLMRayDataEngine] Inference failed: {exc}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            return None

    def _process_results(
        self,
        rows: List[Dict[str, Any]],
        t0: float,
        t1: float,
        prompts: List[str],
    ) -> Dict[str, Any]:
        total_time = t1 - t0

        def _row_id(row: Dict[str, Any], fallback: int) -> int:
            value = row.get("row_id", fallback)
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        ordered = sorted(rows, key=lambda r: _row_id(r, 0))
        results = []
        total_tokens = 0
        for i, row in enumerate(ordered):
            idx = _row_id(row, i)
            instruction = row.get("prompt") or (
                prompts[idx] if 0 <= idx < len(prompts) else ""
            )
            response = row.get("generated_text") or ""
            n_tok = int(len(str(response).split()) * 1.3)
            total_tokens += n_tok
            results.append({
                "id": idx + 1,
                "prompt": instruction,
                "response": response,
                "tokens_generated": n_tok,
                "processing_time_s": 0.0,
                "tokens_per_second": 0.0,
            })

        metrics = {
            "total_prompts": len(results),
            "total_time_s": total_time,
            "total_tokens": total_tokens,
            "avg_processing_time_s": 0.0,
            "throughput_tokens_per_s": total_tokens / total_time if total_time > 0 else 0,
            "success_rate": len(results) / len(prompts) if prompts else 0,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "concurrency": self.concurrency,
            "batch_size": self.batch_size,
            "executor": "ray_data_llm",
        }
        print(
            f"[VLLMRayDataEngine] Done: {metrics['total_prompts']} prompts, "
            f"{metrics['throughput_tokens_per_s']:.1f} tok/s, "
            f"success={metrics['success_rate']:.1%}"
        )
        return {
            "results": results,
            "performance_metrics": metrics,
            "configuration": {
                "model_path": self.model_path,
                "tensor_parallel_size": self.tensor_parallel_size,
                "pipeline_parallel_size": self.pipeline_parallel_size,
                "data_parallel_size": self.data_parallel_size,
                "concurrency": self.concurrency,
                "batch_size": self.batch_size,
                "gpu_memory_utilization": self.gpu_memory_utilization,
                "max_model_len": self.max_model_len,
                "enforce_eager": self.enforce_eager,
                "language_model_only": self.language_model_only,
                "dtype": self.dtype,
                "warmup": self.warmup,
                "distributed_executor_backend": "ray",
                "processor": "vLLMEngineProcessorConfig",
            },
        }
