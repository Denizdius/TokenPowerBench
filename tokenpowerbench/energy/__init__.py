"""Energy monitors with independent sensor capability detection.

``gpu_only`` never probes RAPL or IPMI. ``auto`` enables each readable sensor;
``full_node`` requires an actual IPMI power reading; CPU/DRAM RAPL is optional.
Missing measurements are None and whole-node totals are exclusively IPMI.
"""

from .base import EnergyMetrics, EnergyMonitor
from .gpu_monitor import GPUEnergyMonitor, integrate_power
from .full_node_monitor import FullNodeEnergyMonitor, _RaplReader

__all__ = ["EnergyMetrics", "EnergyMonitor", "GPUEnergyMonitor", "FullNodeEnergyMonitor",
           "create_monitor", "integrate_power"]


def create_monitor(mode="auto", device_indices=None, device_uuids=None, *,
                   tensor_parallel_size=None, pipeline_parallel_size=1,
                   data_parallel_size=1, use_ray_cluster=False):
    """Create a monitor for the selected physical GPUs.

    Explicit device_indices are NVML physical indices. Otherwise device_uuids
    are used, or resolved from torch-visible CUDA devices. No host-wide default
    is used. Passing tensor/pipeline/data parallel sizes without device_indices
    selects physical GPUs ``0 .. TP×PP×DP-1``. ``use_ray_cluster`` samples NVML
    on every Ray GPU node instead of the driver host. Call close() in a finally
    block, even after a successful stop().
    """
    if use_ray_cluster:
        from .ray_gpu_monitor import RayClusterGPUEnergyMonitor

        print("[create_monitor] Using RayClusterGPUEnergyMonitor (NVML on each GPU node).")
        return RayClusterGPUEnergyMonitor(
            tensor_parallel_size=1 if tensor_parallel_size is None else tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            data_parallel_size=data_parallel_size,
            device_indices=device_indices,
        )
    if device_indices is None and device_uuids is None and tensor_parallel_size is not None:
        width = tensor_parallel_size * pipeline_parallel_size * data_parallel_size
        if width < 1:
            raise ValueError("TP × PP × DP must be at least 1")
        device_indices = list(range(width))
    if mode == "gpu_only":
        monitor = GPUEnergyMonitor(device_indices=device_indices, device_uuids=device_uuids)
    elif mode in ("auto", "full_node"):
        monitor = FullNodeEnergyMonitor(device_indices=device_indices, device_uuids=device_uuids,
                                        strict=mode == "full_node")
    else:
        raise ValueError(f"Unknown monitor mode: {mode!r}. Choose 'auto', 'gpu_only', or 'full_node'.")
    for name, capability in monitor.capabilities.items():
        state = "available" if capability["available"] else "unavailable"
        print(f"[Energy] {name}: {state}; {capability['reason']}")
    return monitor


def _rapl_accessible():
    """Compatibility probe based on actual counter reads, rather than os.access."""
    return _RaplReader().capability("cpu")["available"]
