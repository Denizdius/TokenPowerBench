"""
tokenpowerbench.energy
======================

Energy monitoring for LLM inference benchmarking.

Two monitor implementations:

    GPUEnergyMonitor     -- GPU power only via NVIDIA NVML.
    FullNodeEnergyMonitor -- GPU + CPU (Intel RAPL) + total node (IPMI).
"""

import os

from .base import EnergyMetrics, EnergyMonitor
from .gpu_monitor import GPUEnergyMonitor
from .full_node_monitor import FullNodeEnergyMonitor

__all__ = [
    "EnergyMetrics",
    "EnergyMonitor",
    "GPUEnergyMonitor",
    "FullNodeEnergyMonitor",
    "create_monitor",
]


def create_monitor(
    mode: str = "auto",
    *,
    tensor_parallel_size: int = 1,
    pipeline_parallel_size: int = 1,
    data_parallel_size: int = 1,
) -> EnergyMonitor:
    """
    Factory for energy monitors.

    Parameters
    ----------
    mode : str
        "auto"       FullNodeEnergyMonitor if RAPL is readable, else GPUEnergyMonitor.
        "gpu_only"   GPU power via NVML only.
        "full_node"  GPU + CPU (RAPL) + node total (IPMI).
    tensor_parallel_size, pipeline_parallel_size, data_parallel_size : int
        vLLM parallelism degrees. GPU energy totals include only the first
        ``TP × PP × DP`` devices (typically GPU 0 .. N-1).
    """
    gpu_kw = dict(
        tensor_parallel_size=tensor_parallel_size,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
    )
    if mode == "gpu_only":
        return GPUEnergyMonitor(**gpu_kw)
    if mode == "full_node":
        return FullNodeEnergyMonitor(**gpu_kw)
    if mode == "auto":
        if _rapl_accessible():
            return FullNodeEnergyMonitor(**gpu_kw)
        return GPUEnergyMonitor(**gpu_kw)
    raise ValueError(f"Unknown monitor mode: {mode!r}. Choose 'auto', 'gpu_only', or 'full_node'.")


def _rapl_accessible() -> bool:
    rapl_root = "/sys/class/powercap/intel-rapl"
    if not os.path.exists(rapl_root):
        return False
    try:
        for entry in os.listdir(rapl_root):
            energy_path = os.path.join(rapl_root, entry, "energy_uj")
            if os.access(energy_path, os.R_OK):
                return True
    except OSError:
        pass
    return False
