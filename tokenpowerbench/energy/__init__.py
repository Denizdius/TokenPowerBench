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


def create_monitor(mode: str = "auto") -> EnergyMonitor:
    """
    Factory for energy monitors.

    Parameters
    ----------
    mode : str
        "auto"       FullNodeEnergyMonitor if RAPL is readable, else GPUEnergyMonitor.
        "gpu_only"   GPU power via NVML only.
        "full_node"  GPU + CPU (RAPL) + node total (IPMI).
    """
    if mode == "gpu_only":
        return GPUEnergyMonitor()
    if mode == "full_node":
        return FullNodeEnergyMonitor()
    if mode == "auto":
        if _rapl_accessible():
            return FullNodeEnergyMonitor()
        return GPUEnergyMonitor()
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
