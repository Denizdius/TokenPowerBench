"""
vLLM single-node inference engine.
"""

from __future__ import annotations

import os
import time
from typing import Any, List, Optional, Tuple

import torch

from .base import InferenceEngine

try:
    from vllm import LLM, SamplingParams
    _VLLM_AVAILABLE = True
except ImportError:
    _VLLM_AVAILABLE = False


class VLLMEngine(InferenceEngine):
    """
    vLLM inference engine for single-node benchmarking.

    Automatically uses all available GPUs via tensor parallelism.
    """

    def __init__(self) -> None:
        self._llm: Optional[LLM] = None

    @property
    def available(self) -> bool:
        return _VLLM_AVAILABLE

    def setup_model(
        self,
        model_path: str,
        gpu_memory_utilization: float = 0.85,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        enforce_eager: bool = False,
    ) -> Optional[LLM]:
        if not self.available:
            print("[VLLMEngine] vLLM is not installed.")
            return None

        torch.cuda.empty_cache()
        n_gpus = max(torch.cuda.device_count(), 1)
        tp = tensor_parallel_size if tensor_parallel_size is not None else n_gpus
        pp = pipeline_parallel_size
        dp = data_parallel_size

        if max_model_len is None:
            max_model_len = _read_max_position_embeddings(model_path)

        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        print(
            f"[VLLMEngine] Loading {model_path}  "
            f"TP={tp} PP={pp} DP={dp}  max_len={max_model_len}  "
            f"enforce_eager={enforce_eager}"
        )

        llm_kwargs: dict = {}
        if pp != 1:
            llm_kwargs["pipeline_parallel_size"] = pp
        if dp != 1:
            llm_kwargs["data_parallel_size"] = dp

        try:
            self._llm = LLM(
                model=model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len,
                trust_remote_code=True,
                enforce_eager=enforce_eager,
                **llm_kwargs,
            )
            print("[VLLMEngine] Model loaded.")
            return self._llm
        except Exception as exc:
            import traceback
            print(f"[VLLMEngine] Failed to load model: {exc}")
            traceback.print_exc()
            return None

    def run_inference(
        self,
        prompts: List[str],
        batch_size: int,
        max_tokens: int = 200,
        temperature: float = 0.7,
    ) -> List[Any]:
        if self._llm is None:
            return []
        params = SamplingParams(max_tokens=max_tokens, temperature=temperature)
        results = []
        for i in range(0, len(prompts), batch_size):
            results.extend(self._llm.generate(prompts[i:i + batch_size], params))
        return results

    def run_benchmark(
        self,
        prompts: List[str],
        num_samples: int,
        batch_size: int,
        max_tokens: int,
    ) -> Tuple[List[Any], float, float]:
        if self._llm is None:
            return [], 0.0, 0.0

        full: List[str] = []
        while len(full) < num_samples:
            full.extend(prompts)
        full = full[:num_samples]

        all_outputs: List[Any] = []
        t0 = time.time()
        for i in range(0, num_samples, batch_size):
            batch = full[i:i + batch_size]
            all_outputs.extend(self.run_inference(batch, batch_size, max_tokens))
        t1 = time.time()
        return all_outputs, t0, t1

    def estimate_tokens(self, outputs: List[Any]) -> int:
        total = 0
        for out in outputs:
            if hasattr(out, "outputs") and out.outputs:
                total += int(len(out.outputs[0].text.split()) * 1.3)
        return total


def _read_max_position_embeddings(model_path: str, default: int = 2048) -> int:
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path)
        return getattr(cfg, "max_position_embeddings", default)
    except Exception:
        return default
