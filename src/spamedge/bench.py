"""Measurement helpers shared by the PC and Raspberry Pi benchmarks.

Latency is measured end to end: raw text in, spam probability out. That
includes the TF-IDF transform or the tokeniser, because a deployed filter pays
for them too.
"""

from __future__ import annotations

import os
import platform
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

WARMUP_CALLS = 50
TIMED_CALLS = 1000
REPEATS = 3
# My first PC measurements ran while another job was using the CPU, and p50 drifted
# by 50 % from one run to the next. Since then the benchmark checks the machine is
# idle before each run and refuses to measure above this system-wide CPU load.
MAX_IDLE_CPU_PERCENT = 15.0
IDLE_CHECK_SECONDS = 3.0


def system_cpu_percent(seconds: float = IDLE_CHECK_SECONDS) -> float:
    """System-wide CPU load averaged over `seconds`, in percent of all cores."""
    import psutil
    return float(psutil.cpu_percent(interval=seconds))


def summarize_runs(runs: list[dict[str, float]], keys=("p50_ms", "p95_ms", "p99_ms")) -> dict[str, dict]:
    """Mean and sample standard deviation of each statistic across repeated runs."""
    if not runs:
        raise ValueError("no run to summarize")
    out = {}
    for key in keys:
        values = np.array([r[key] for r in runs], dtype=float)
        out[key] = {"mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0}
    return out


def latency_stats(ns: np.ndarray) -> dict[str, float]:
    """Summary of per-call latencies given in nanoseconds, returned in milliseconds."""
    ms = np.asarray(ns, dtype=float) / 1e6
    if ms.size == 0:
        raise ValueError("no latency sample")
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    return {
        "n": int(ms.size), "mean_ms": float(ms.mean()), "std_ms": float(ms.std(ddof=1)) if ms.size > 1 else 0.0,
        "p50_ms": float(p50), "p95_ms": float(p95), "p99_ms": float(p99),
        "min_ms": float(ms.min()), "max_ms": float(ms.max()),
    }


def time_single_calls(predict_one: Callable[[str], float], texts: Sequence[str],
                      warmup: int = WARMUP_CALLS, timed: int = TIMED_CALLS) -> np.ndarray:
    """Call predict_one on one text at a time and return the duration of each call in ns.

    The texts are cycled in the given order, so every model sees exactly the same
    sequence of messages (and of message lengths). Warm-up calls fill caches and
    let lazy initialisation happen; they are not recorded.
    """
    if not texts:
        raise ValueError("no text to time")
    for i in range(warmup):
        predict_one(texts[i % len(texts)])
    out = np.empty(timed, dtype=np.int64)
    for i in range(timed):
        text = texts[i % len(texts)]
        t0 = time.perf_counter_ns()
        predict_one(text)
        out[i] = time.perf_counter_ns() - t0
    return out


def size_on_disk(path: Path) -> int:
    """Bytes of a file, or of every file below a directory."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        raise FileNotFoundError(path)
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def cpu_name() -> str:
    """A readable CPU name on Windows, Linux x86 and the Raspberry Pi."""
    if platform.system() == "Windows":
        try:
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        except OSError:
            return platform.processor()
    model_file = Path("/proc/device-tree/model")  # "Raspberry Pi 5 Model B Rev 1.0"
    if model_file.exists():
        return model_file.read_text(errors="ignore").strip("\x00\n ")
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def machine_info(torch_threads: int | None = None) -> dict:
    info = {
        "cpu": cpu_name(), "logical_cpus": os.cpu_count(), "machine": platform.machine(),
        "os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
    }
    if torch_threads is not None:
        info["torch_threads"] = torch_threads
    return info
