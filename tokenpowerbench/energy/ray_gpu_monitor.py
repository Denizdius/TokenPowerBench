"""Cluster-wide GPU energy monitor: one NVML sampler actor per Ray GPU node.

Samples use wall-clock timestamps because actor ``perf_counter`` values are not
comparable with the driver. ``compute_metrics`` integrates those samples with
the 1.0 power integral. A requested driver ``perf_counter`` window that does
not overlap the traces falls back to the collected trace span.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from tokenpowerbench.distributed.ray_cluster import gpu_alive_nodes

from .base import EnergyMetrics, EnergyMonitor
from .gpu_monitor import integrate_power


class RayClusterGPUEnergyMonitor(EnergyMonitor):
    """NVML on every GPU node. Indices are offset across nodes in hostname order."""

    def __init__(
        self,
        *,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        data_parallel_size: int = 1,
        device_indices: Optional[list] = None,
        num_gpus_for_energy: Optional[int] = None,
    ) -> None:
        self._parallel = (tensor_parallel_size, pipeline_parallel_size, data_parallel_size)
        if device_indices is not None:
            self._needed_gpus = len(device_indices)
            self._explicit_indices = list(device_indices)
        elif num_gpus_for_energy is not None:
            self._needed_gpus = max(1, num_gpus_for_energy)
            self._explicit_indices = None
        else:
            self._needed_gpus = max(1, tensor_parallel_size * pipeline_parallel_size * data_parallel_size)
            self._explicit_indices = None
        self._samplers: list[Any] = []
        self._samples = {"gpu": [], "rapl": [], "ipmi": []}
        self._closed = False
        self._active = False
        self.capabilities = {
            "gpu": {
                "available": True,
                "reason": "Ray actors sample NVML on each GPU node",
                "scope": "ray_cluster_gpus",
                "warning": "Cluster samples use wall time. Sub-second NVML windows stay unresolved.",
            }
        }

    @property
    def samples(self):
        return {"gpu": list(self._samples["gpu"]), "rapl": [], "ipmi": []}

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("Monitor is closed")
        if not ray.is_initialized():
            raise RuntimeError("RayClusterGPUEnergyMonitor requires ray.init() first")
        nodes = gpu_alive_nodes()
        if not nodes:
            raise RuntimeError("No alive Ray nodes with GPUs.")
        from .ray_gpu_sampler import RayGPUSampler

        self._samplers = []
        for node in nodes:
            actor_cls = ray.remote(num_gpus=0, num_cpus=1, scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node["NodeID"], soft=False))(RayGPUSampler)
            self._samplers.append(actor_cls.remote())
        ray.get([actor.start.remote() for actor in self._samplers])
        self._active = True

    def stop(self) -> None:
        if not self._active:
            return
        self._active = False
        if not self._samplers:
            return
        traces = ray.get([actor.stop_and_collect.remote() for actor in self._samplers])
        for actor in self._samplers:
            try:
                ray.kill(actor)
            except Exception:
                pass
        self._samplers = []
        self._merge_traces(traces)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self.stop()
        finally:
            self._closed = True

    def _merge_traces(self, traces: list[dict]) -> None:
        usable = [trace for trace in traces if trace.get("power")]
        if not usable:
            return
        count = min(len(trace["power"]) for trace in usable)
        merged = []
        for index in range(count):
            power, memory, util, sm_mhz, mem_mhz = [], [], [], [], []
            for trace in usable:
                power.extend(trace["power"][index])
                memory.extend(trace["memory"][index] if index < len(trace["memory"]) else [])
                util.extend(trace["utilization"][index] if index < len(trace["utilization"]) else [])
                sm_mhz.extend(trace.get("sm_mhz", [])[index] if index < len(trace.get("sm_mhz", [])) else [])
                mem_mhz.extend(trace.get("mem_mhz", [])[index] if index < len(trace.get("mem_mhz", [])) else [])
            merged.append({
                "timestamp_s": float(usable[0]["timestamps"][index]),
                "power_w": power,
                "memory_mb": memory,
                "utilization_pct": util,
                "sm_mhz": sm_mhz,
                "mem_mhz": mem_mhz,
            })
        self._samples["gpu"] = merged
        self._cluster_gpus = sum(trace["n_gpus"] for trace in usable)

    def _selected_columns(self, width: int) -> list[int]:
        if self._explicit_indices is not None:
            return [index for index in self._explicit_indices if index < width]
        count = min(self._needed_gpus, width)
        return list(range(count))

    def compute_metrics(self, duration, total_output_tokens, num_responses,
                        start_time=None, end_time=None) -> EnergyMetrics:
        samples = self._samples["gpu"]
        metrics = EnergyMetrics(duration=duration, total_output_tokens=total_output_tokens,
                                 num_responses=num_responses, capabilities=self.capabilities,
                                 energy_scope="ray_cluster_gpus")
        if len(samples) < 2:
            metrics.warnings.append("Ray cluster monitor collected fewer than two NVML samples")
            return metrics
        first, last = samples[0]["timestamp_s"], samples[-1]["timestamp_s"]
        if (start_time is None or end_time is None or not (first <= start_time < end_time <= last)):
            start_time, end_time = first, last
        metrics.start_time = start_time
        metrics.end_time = end_time
        metrics.duration = end_time - start_time
        width = len(samples[0]["power_w"])
        columns = self._selected_columns(width)
        metrics.gpus_included_for_energy = columns
        energies = []
        for column in columns:
            series = [(sample["timestamp_s"], sample["power_w"][column]
                       if column < len(sample["power_w"]) else None) for sample in samples]
            integrated = integrate_power(series, start_time, end_time)
            metrics.per_gpu_power_w[column] = integrated["average_power_w"]
            metrics.per_gpu_energy_j[column] = integrated["energy_j"]
            if not integrated["valid"]:
                metrics.warnings.append(f"GPU {column}: {integrated['reason']}")
            energies.append(integrated["energy_j"])
        if energies and all(value is not None for value in energies):
            metrics.gpu_energy_j = sum(energies)
            metrics.gpu_avg_power_w = metrics.gpu_energy_j / metrics.duration
        return metrics

    def save_gpu_usage(self, path_stem):
        from pathlib import Path
        import json

        stem = Path(path_stem)
        name = stem.name
        for suffix in ("_gpu_usage.json", "_gpu_usage", ".json"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        stem = stem.with_name(name)
        stem.parent.mkdir(parents=True, exist_ok=True)
        samples = self._samples["gpu"]
        included = self._selected_columns(len(samples[0]["power_w"]) if samples else 0)
        paths = {}
        for key, field, unit, filename in (
            ("power", "power_w", "W", "gpu_power"),
            ("memory", "memory_mb", "MB", "gpu_memory"),
            ("utilization", "utilization_pct", "%", "gpu_utilization"),
            ("sm_clock", "sm_mhz", "MHz", "gpu_sm_clock"),
            ("mem_clock", "mem_mhz", "MHz", "gpu_mem_clock"),
        ):
            rows = []
            for sample in samples:
                values = sample.get(field) or []
                gpus = {}
                for index in included:
                    value = values[index] if index < len(values) else None
                    gpus[str(index)] = None if value is None else float(value)
                rows.append({"t_s": float(sample["timestamp_s"]), "gpus": gpus})
            payload = {"metric": key, "unit": unit, "gpus_included": included,
                       "num_samples": len(rows), "samples": rows}
            out = stem.parent / f"{stem.name}_{filename}.json"
            out.write_text(json.dumps(payload, indent=2))
            paths[key] = str(out)
        return paths
