"""Hardware detection: GPU name, VRAM, and system RAM.

Design goals:
- Pure stdlib by default; `psutil` only used when installed (better RAM read).
- Never raise on detection failure — return a value or None and let the caller
  decide how to surface it (the CLI prints a warning instead of crashing).
- Cross-platform: NVIDIA (Linux/Windows), AMD (Linux), Intel, Apple Metal, CPU-only.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import List, Optional

# VRAM reserved (GB) for context + KV cache + CUDA/Metal overhead on top of the
# weights. Conservative default so a "fits" verdict is trustworthy.
DEFAULT_VRAM_HEADROOM_GB = 2.0
# CPU offload / partial GPU: weights that don't fit in VRAM spill to system RAM.
# This is how much free RAM we require for the overflow (GB).
CPU_OFFLOAD_MIN_FREE_RAM_GB = 4.0


@dataclass
class GpuInfo:
    name: str
    vram_gb: Optional[float] = None  # None = unknown (e.g. shared-memory iGPU)
    kind: str = "unknown"  # "nvidia" | "amd" | "apple" | "intel" | "cpu"
    details: List[str] = field(default_factory=list)

    @property
    def is_usable(self) -> bool:
        return self.kind != "cpu"


@dataclass
class Hardware:
    gpus: List[GpuInfo] = field(default_factory=list)
    total_ram_gb: Optional[float] = None
    free_ram_gb: Optional[float] = None
    os: str = field(default_factory=lambda: platform.platform())
    arch: str = field(default_factory=lambda: platform.machine())

    @property
    def best_gpu(self) -> Optional[GpuInfo]:
        """The GPU with the most known VRAM (ties broken by order)."""
        candidates = [
            g
            for g in self.gpus
            if g.is_usable and (g.vram_gb or 0) > 0
        ]
        if not candidates:
            return self.gpus[0] if self.gpus else None
        return max(candidates, key=lambda g: g.vram_gb or 0)

    @property
    def primary_vram_gb(self) -> Optional[float]:
        g = self.best_gpu
        return g.vram_gb if g else None

    def summary(self) -> str:
        lines = []
        if self.gpus:
            names = ", ".join(
                f"{g.name} ({g.vram_gb:.1f} GB VRAM)" if g.vram_gb else g.name
                for g in self.gpus
            )
            lines.append(f"GPU: {names}")
        else:
            lines.append("GPU: none detected (CPU-only inference)")
        if self.total_ram_gb is not None:
            free = (
                f", {self.free_ram_gb:.1f} GB free"
                if self.free_ram_gb is not None
                else ""
            )
            lines.append(f"RAM: {self.total_ram_gb:.1f} GB total{free}")
        lines.append(f"OS: {self.os} ({self.arch})")
        return "\n".join(lines)


def _run(cmd: List[str], timeout: int = 5) -> Optional[str]:
    """Run a command, return stdout (or None). Swallows all errors."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            check=False,
        )
        return out.stdout
    except Exception:
        return None


def _detect_nvidia() -> List[GpuInfo]:
    gpus: List[GpuInfo] = []
    # Preferred: structured query.
    out = _run([
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if out:
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if not parts or not parts[0]:
                continue
            name = parts[0]
            vram = None
            if len(parts) >= 2 and parts[1]:
                try:
                    vram = int(parts[1]) / 1024.0
                except ValueError:
                    vram = None
            details = [parts[2]] if len(parts) >= 3 and parts[2] else []
            gpus.append(GpuInfo(name=name, vram_gb=vram, kind="nvidia", details=details))
        return gpus

    # Fallback: parse the human banner (works even if --query flags are old).
    banner = _run(["nvidia-smi"])
    if banner:
        for m in re.finditer(
            r"(?:Tesla|GeForce|NVIDIA)[^ ]*(?: [^ ]+)*", banner
        ):
            pass
        m = re.search(r"(NVIDIA [A-Za-z0-9 \-]+?)(?:\s*\||\s*\d+ MiB|\s*$)", banner)
        if m:
            gpus.append(GpuInfo(name=m.group(1).strip(), vram_gb=None, kind="nvidia"))
        return gpus
    return []


def _detect_amd_linux() -> List[GpuInfo]:
    gpus: List[GpuInfo] = []
    # /sys/class/drm/card*/mem_info_vram_total
    base = "/sys/class/drm"
    try:
        import glob
        for path in sorted(glob.glob(f"{base}/card*/mem_info_vram_total")):
            with open(path) as fh:
                vram = int(fh.read().strip()) / (1024 ** 3)
            card = os.path.basename(os.path.dirname(path))
            name = f"AMD GPU ({card})"
            gpus.append(GpuInfo(name=name, vram_gb=round(vram, 1), kind="amd"))
    except Exception:
        pass
    if not gpus:
        out = _run(["rocminfo"])
        if out:
            for m in re.finditer(r"Card\[(\d+)\]\s+Name:\s+(.+)", out):
                gpus.append(GpuInfo(name=m.group(2).strip(), vram_gb=None, kind="amd"))
    return gpus


def _detect_apple() -> List[GpuInfo]:
    if sys.platform != "darwin":
        return []
    name = "Apple Silicon (unified memory)"
    # sysctl hw.memsize -> total unified memory, shared with GPU.
    out = _run(["sysctl", "-n", "hw.memsize"])
    vram = None
    if out:
        try:
            vram = int(out.strip()) / (1024 ** 3)
        except ValueError:
            vram = None
    model = os.uname().machine if hasattr(os, "uname") else "Apple"
    return [GpuInfo(name=f"{name} [{model}]", vram_gb=vram, kind="apple")]


def _detect_intel_linux() -> List[GpuInfo]:
    if not os.path.isdir("/dev/dri"):
        return []
    out = _run(["ls", "/dev/dri"])
    if out and re.search(r"renderD12[8-9]|i915", out):
        return [GpuInfo(name="Intel GPU (i915, shared memory)", vram_gb=None, kind="intel")]
    return []


def _total_and_free_ram_gb() -> tuple[Optional[float], Optional[float]]:
    total = free = None
    # psutil path (best when installed).
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        total = vm.total / (1024 ** 3)
        free = vm.available / (1024 ** 3)
    except Exception:
        pass
    if total is None:
        total = _os_total_ram_gb()
    if free is None:
        free = _os_free_ram_gb()
    return total, free


def _os_total_ram_gb() -> Optional[float]:
    # Linux: /proc/meminfo MemTotal.
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    kb = int(re.search(r"(\d+)", line).group(1))
                    return kb / (1024 ** 2)
    except Exception:
        pass
    # macOS / FreeBSD: sysctl hw.memsize.
    if sys.platform in ("darwin", "freebsd"):
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                return int(out.strip()) / (1024 ** 3)
            except ValueError:
                pass
    # Windows: GlobalMemoryStatusEx via ctypes.
    if os.name == "nt":
        return _windows_total_ram_gb()
    return None


def _windows_total_ram_gb() -> Optional[float]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return stat.ullTotalPhys / (1024 ** 3)
    except Exception:
        return None


def _os_free_ram_gb() -> Optional[float]:
    try:
        with open("/proc/meminfo") as fh:
            text = fh.read()
    except Exception:
        text = ""
    if "MemAvailable:" in text:
        m = re.search(r"MemAvailable:\s+(\d+)", text)
        if m:
            return int(m.group(1)) / (1024 ** 2)
    # macOS: vm_stat free + inactive pages (rough "available").
    if sys.platform == "darwin":
        out = _run(["vm_stat"])
        if out:
            page = 16384
            m = re.search(r"Pages free:\s+(\d+)", out)
            mi = re.search(r"Pages inactive:\s+(\d+)", out)
            if m:
                val = int(m.group(1)) * page
                if mi:
                    val += int(mi.group(1)) * page
                return val / (1024 ** 3)
    if os.name == "nt":
        return _windows_free_ram_gb()
    return None


def _windows_free_ram_gb() -> Optional[float]:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
        return stat.ullAvailPhys / (1024 ** 3)
    except Exception:
        return None


def detect() -> Hardware:
    """Detect GPUs + RAM. Never raises; returns best-effort Hardware."""
    gpus: List[GpuInfo] = []
    if os.name == "nt" or sys.platform.startswith("linux"):
        gpus.extend(_detect_nvidia())
        if sys.platform.startswith("linux"):
            gpus.extend(_detect_amd_linux())
            gpus.extend(_detect_intel_linux())
    elif sys.platform == "darwin":
        gpus.extend(_detect_apple())
    total_ram, free_ram = _total_and_free_ram_gb()
    return Hardware(
        gpus=gpus, total_ram_gb=total_ram, free_ram_gb=free_ram
    )
