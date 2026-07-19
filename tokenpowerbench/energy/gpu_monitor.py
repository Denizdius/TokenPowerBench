"""
GPU-only energy monitor via NVIDIA NVML.

No elevated privileges required. Works for any data-center or workstation
user who has CUDA access.
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .base import EnergyMonitor, EnergyMetrics

try:
    import pynvml
    _NVML_AVAILABLE = True
except ImportError:
    _NVML_AVAILABLE = False

_GPU_SAMPLE_INTERVAL = 0.1  # 100 ms — NVML updates at ~100 ms granularity
_EDGE_TRIM_FRAC = 0.10      # Drop first/last 10% of samples (ramp-up / idle tail)


def trim_edges(readings: list, frac: float = _EDGE_TRIM_FRAC) -> list:
    """Remove the first and last `frac` fraction of a reading list."""
    n = len(readings)
    if n == 0:
        return readings
    buf = max(1, int(n * frac))
    if n <= buf * 2:
        return readings
    return readings[buf:-buf]


class GPUEnergyMonitor(EnergyMonitor):
    """
    Monitors GPU power draw via NVML at 100 ms intervals.

    Suitable for any user with CUDA access — no root required.
    """

    def __init__(
        self,
        *,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        num_gpus_for_energy: Optional[int] = None,
    ) -> None:
        if not _NVML_AVAILABLE:
            raise RuntimeError(
                "pynvml is not installed. Run: pip install nvidia-ml-py"
            )
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]

        if num_gpus_for_energy is not None:
            needed = max(1, num_gpus_for_energy)
        else:
            needed = max(
                1,
                tensor_parallel_size * pipeline_parallel_size * data_parallel_size,
            )
        if needed > n:
            print(
                f"[GPUEnergyMonitor] Warning: TP×PP×DP needs {needed} GPU(s) but "
                f"only {n} visible; capping energy accounting to {n}."
            )
            needed = n
        self._gpu_indices_for_energy = list(range(needed))
        self._parallel = (tensor_parallel_size, pipeline_parallel_size, data_parallel_size)

        print(f"[GPUEnergyMonitor] Found {n} GPU(s):")
        for i, h in enumerate(self._handles):
            name = pynvml.nvmlDeviceGetName(h)
            role = (
                "used for energy"
                if i in self._gpu_indices_for_energy
                else "idle (excluded from totals)"
            )
            print(f"  GPU {i}: {name}  [{role}]")
        tp, pp, dp = self._parallel
        idxs = ", ".join(str(i) for i in self._gpu_indices_for_energy)
        print(
            f"[GPUEnergyMonitor] Energy totals use GPU(s) [{idxs}] "
            f"(TP={tp} × PP={pp} × DP={dp} → {len(self._gpu_indices_for_energy)} GPU(s))"
        )

        self._lock = threading.Lock()
        self._active = False
        self._thread: threading.Thread | None = None
        self._reset_gpu_buffers()

    def _reset_gpu_buffers(self) -> None:
        self._readings: List[List[float]] = []
        self._memory_readings: List[List[float]] = []
        self._util_readings: List[List[float]] = []
        self._timestamps: List[float] = []
        self._sample_t0: Optional[float] = None

    def start(self) -> None:
        self._reset_gpu_buffers()
        self._active = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        time.sleep(2.0)  # let GPU settle before inference starts

    def stop(self) -> None:
        self._active = False
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    def compute_metrics(
        self,
        duration: float,
        total_output_tokens: int,
        num_responses: int,
    ) -> EnergyMetrics:
        with self._lock:
            readings = list(self._readings)

        readings = trim_edges(readings)
        if not readings:
            return EnergyMetrics(
                duration=duration,
                total_output_tokens=total_output_tokens,
                num_responses=num_responses,
            )

        arr = np.array(readings)
        per_gpu_avg = np.mean(arr, axis=0)
        active = self._gpu_indices_for_energy
        total_gpu_w = float(np.sum(per_gpu_avg[active]))

        return EnergyMetrics(
            duration=duration,
            total_output_tokens=total_output_tokens,
            num_responses=num_responses,
            gpu_avg_power_w=total_gpu_w,
            gpu_energy_j=total_gpu_w * duration,
            per_gpu_power_w={i: float(v) for i, v in enumerate(per_gpu_avg)},
            gpus_included_for_energy=list(active),
        )

    def _meta(self, metric: str, unit: str, num_samples: int) -> Dict[str, Any]:
        tp, pp, dp = self._parallel
        return {
            "metric": metric,
            "unit": unit,
            "sample_interval_s": _GPU_SAMPLE_INTERVAL,
            "gpus_included": list(self._gpu_indices_for_energy),
            "tensor_parallel_size": tp,
            "pipeline_parallel_size": pp,
            "data_parallel_size": dp,
            "num_samples": num_samples,
        }

    def _series_payload(
        self,
        metric: str,
        unit: str,
        values: List[List[float]],
        timestamps: List[float],
    ) -> Dict[str, Any]:
        active = list(self._gpu_indices_for_energy)
        samples = []
        n = min(len(values), len(timestamps))
        for i in range(n):
            gpus: Dict[str, float] = {}
            for gpu_idx in active:
                v = values[i][gpu_idx] if gpu_idx < len(values[i]) else 0.0
                gpus[str(gpu_idx)] = float(v)
            samples.append({"t_s": float(timestamps[i]), "gpus": gpus})
        payload = self._meta(metric, unit, len(samples))
        payload["samples"] = samples
        return payload

    def export_gpu_traces(self) -> Dict[str, Dict[str, Any]]:
        """
        Return three separate traces for active GPUs (TP×PP×DP):
        power (W), memory (MB), utilization (%).
        """
        with self._lock:
            power = list(self._readings)
            memory = list(self._memory_readings)
            util = list(self._util_readings)
            timestamps = list(self._timestamps)

        return {
            "power": self._series_payload("power", "W", power, timestamps),
            "memory": self._series_payload("memory_used", "MB", memory, timestamps),
            "utilization": self._series_payload(
                "utilization", "%", util, timestamps
            ),
        }

    def save_gpu_usage(self, path_stem: Union[str, Path]) -> Dict[str, str]:
        """
        Write three JSON traces from ``path_stem``:

            {stem}_gpu_power.json
            {stem}_gpu_memory.json
            {stem}_gpu_utilization.json

        Returns a dict mapping metric name → file path.
        """
        stem = Path(path_stem)
        # Allow callers to pass "..._gpu_usage" / "..._gpu_usage.json" for compat.
        name = stem.name
        for suffix in ("_gpu_usage.json", "_gpu_usage", ".json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        stem = stem.with_name(name)
        stem.parent.mkdir(parents=True, exist_ok=True)

        traces = self.export_gpu_traces()
        paths: Dict[str, str] = {}
        for key, filename_key in (
            ("power", "gpu_power"),
            ("memory", "gpu_memory"),
            ("utilization", "gpu_utilization"),
        ):
            out = stem.parent / f"{stem.name}_{filename_key}.json"
            with open(out, "w") as f:
                json.dump(traces[key], f, indent=2)
            paths[key] = str(out)
            print(
                f"[GPUEnergyMonitor] Saved {key} trace "
                f"({traces[key]['num_samples']} samples, "
                f"GPUs {traces[key]['gpus_included']}) → {out}"
            )
        return paths

    def _sample_loop(self) -> None:
        while self._active:
            now = time.time()
            if self._sample_t0 is None:
                self._sample_t0 = now
            power_sample: List[float] = []
            memory_sample: List[float] = []
            util_sample: List[float] = []
            for handle in self._handles:
                try:
                    mw = pynvml.nvmlDeviceGetPowerUsage(handle)
                    power_sample.append(mw / 1000.0)
                except Exception:
                    power_sample.append(0.0)
                try:
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_sample.append(mem.used / (1024.0 * 1024.0))
                except Exception:
                    memory_sample.append(0.0)
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    util_sample.append(float(util.gpu))
                except Exception:
                    util_sample.append(0.0)
            with self._lock:
                self._readings.append(power_sample)
                self._memory_readings.append(memory_sample)
                self._util_readings.append(util_sample)
                self._timestamps.append(now - self._sample_t0)
            time.sleep(_GPU_SAMPLE_INTERVAL)
