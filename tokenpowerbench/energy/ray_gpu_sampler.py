"""
Per-node NVML sampler for multi-node GPU energy.

Ray must not request GPUs (num_gpus=0). NodeAffinitySchedulingStrategy pins
one actor per node so NVML sees that node's devices 0..N-1.
"""

from __future__ import annotations

import os
import socket
import threading
import time
from typing import Any, Dict, List

from .gpu_monitor import _GPU_SAMPLE_INTERVAL


def _clock_mhz(pynvml, handle, clock_type):
    try:
        value = float(pynvml.nvmlDeviceGetClockInfo(handle, clock_type))
    except Exception:
        return None
    if value < 0:
        return None
    return value


class RayGPUSampler:
    """Local NVML loop. Instantiated as a Ray actor on one GPU node."""

    def __init__(self) -> None:
        import pynvml

        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        self._handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(n)]
        self._hostname = socket.gethostname()
        self._n_gpus = n
        self._lock = threading.Lock()
        self._active = False
        self._thread: threading.Thread | None = None
        self._power: List[List[float]] = []
        self._memory: List[List[float]] = []
        self._util: List[List[float]] = []
        self._sm_mhz: List[List[float]] = []
        self._mem_mhz: List[List[float]] = []
        self._timestamps: List[float] = []
        self._t0: float | None = None
        print(
            f"[RayGPUSampler] {self._hostname}: {n} GPU(s)  "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}"
        )

    def start(self) -> str:
        self._power = []
        self._memory = []
        self._util = []
        self._sm_mhz = []
        self._mem_mhz = []
        self._timestamps = []
        self._t0 = None
        self._active = True
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        return self._hostname

    def stop_and_collect(self) -> Dict[str, Any]:
        self._active = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            import pynvml

            pynvml.nvmlShutdown()
        except Exception:
            pass
        with self._lock:
            return {
                "hostname": self._hostname,
                "n_gpus": self._n_gpus,
                "power": list(self._power),
                "memory": list(self._memory),
                "utilization": list(self._util),
                "sm_mhz": list(self._sm_mhz),
                "mem_mhz": list(self._mem_mhz),
                "timestamps": list(self._timestamps),
            }

    def _sample_loop(self) -> None:
        import pynvml

        while self._active:
            now = time.time()
            if self._t0 is None:
                self._t0 = now
            power_sample: List[float] = []
            memory_sample: List[float] = []
            util_sample: List[float] = []
            sm_sample: List[float] = []
            mem_clock_sample: List[float] = []
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
                sm_sample.append(_clock_mhz(pynvml, handle, 1))
                mem_clock_sample.append(_clock_mhz(pynvml, handle, 2))
            with self._lock:
                self._power.append(power_sample)
                self._memory.append(memory_sample)
                self._util.append(util_sample)
                self._sm_mhz.append(sm_sample)
                self._mem_mhz.append(mem_clock_sample)
                self._timestamps.append(now)
            time.sleep(_GPU_SAMPLE_INTERVAL)
