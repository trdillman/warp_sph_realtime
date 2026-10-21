"""One-click diagnostics report and package logging.

Pure Python: importing this module never imports ``bpy`` or NVIDIA Warp. Every
probe is wrapped, so building a report can never raise.
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import platform
import re
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .warp_runtime import CERTIFIED_CUDA_MAJOR

LOG_FILE_NAME = "warpsph.log"
LOG_MAX_BYTES = 256 * 1024
LOG_BACKUPS = 1  # current file + one rotated copy = 2 x 256 KB
LOG_TAIL_LINES = 40
_HANDLER_TAG = "_warpsph_diagnostics_handler"

# (title, [(key, label)]) in report order. Values are plain values or zero-arg callables.
SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("Add-on", (("addon_version", "Add-on version"),)),
    ("Blender", (
        ("blender_version", "Blender version"),
        ("blender_build_hash", "Build hash"),
        ("gpu_backend", "GPU backend"),
    )),
    ("System", (("os", "OS"), ("python", "Python"))),
    ("Warp", (
        ("warp_version", "Warp version"),
        ("warp_location", "Warp location"),
        ("warp_source", "Warp source"),  # P1.11 hook: download / local wheel / existing folder
        ("cuda_driver", "CUDA driver version"),
        ("cudart", "CUDA runtime (cudart)"),
    )),
    ("GPU", (
        ("gpu_name", "GPU"),
        ("gpu_compute", "Compute capability"),
        ("gpu_memory", "GPU memory"),
    )),
    ("Runtime", (
        ("transport", "Active transport"),
        ("last_error", "Last error"),
        ("last_close", "Last close result"),
        ("interop_unavailable", "Interop unavailable"),
    )),
)


def unavailable(exc: BaseException | str) -> str:
    text = str(exc) if not isinstance(exc, BaseException) else (str(exc) or type(exc).__name__)
    return f"unavailable: {text}"


def redact(text: str, homes: tuple[str, ...] | None = None) -> str:
    """Replace the user's profile path with ``~`` (both slash styles, case-insensitive)."""
    if homes is None:
        homes = tuple(h for h in (os.path.expanduser("~"), os.environ.get("USERPROFILE", "")) if h)
    for home in sorted({h for h in homes if h and h not in ("~", "/", "\\")}, key=len, reverse=True):
        variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
        for variant in variants:
            text = re.sub(re.escape(variant), "~", text, flags=re.IGNORECASE)
    return text


def _resolve(value: Any) -> str:
    try:
        if callable(value):
            value = value()
        if value is None or value == "":
            return "none"
        return str(value)
    except Exception as exc:
        return unavailable(exc)


def read_log_tail(log_path: str | os.PathLike | None, lines: int = LOG_TAIL_LINES) -> list[str]:
    if not log_path:
        return []
    try:
        with open(log_path, encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for line in handle.readlines()[-lines:]]
    except Exception as exc:
        return [unavailable(exc)]


def build_report(data: Mapping[str, Any], recent_log: Any = None, homes: tuple[str, ...] | None = None) -> str:
    """Plain-text report. ``data`` maps SECTIONS keys to values or zero-arg callables."""
    try:
        out = ["WarpSPH diagnostics"]
        for title, fields in SECTIONS:
            out.append("")
            out.append(f"[{title}]")
            for key, label in fields:
                out.append(f"{label}: {_resolve(data[key]) if key in data else unavailable('not collected')}")
        out.append("")
        out.append("[Recent log]")
        try:
            tail = recent_log() if callable(recent_log) else recent_log
            tail = list(tail or [])
        except Exception as exc:
            tail = [unavailable(exc)]
        out.extend(str(line) for line in tail) if tail else out.append("none")
        return redact("\n".join(out), homes)
    except Exception as exc:  # last resort: the button must always produce something
        return f"WarpSPH diagnostics\nreport failed: {type(exc).__name__}"


def warp_probes(wp: Any = None) -> dict[str, Callable[[], Any]]:
    """Probes that read Warp only when the caller passes an already-activated module."""
    def need() -> Any:
        if wp is None:
            raise RuntimeError("Warp not activated this session (press Start once)")
        return wp

    def version() -> Any:
        if wp is None:
            return "not loaded"
        return getattr(wp, "__version__", None) or wp.config.version

    def driver() -> str:
        major, minor = need().get_cuda_driver_version()
        return f"{major}.{minor}"

    def cudart() -> str:
        runtime = need().context.runtime
        value = getattr(runtime, "toolkit_version", None) or runtime.core.cuda_toolkit_version()
        if isinstance(value, int):
            return f"{value // 1000}.{(value % 1000) // 10}"
        return ".".join(str(v) for v in value)

    def device() -> Any:
        return need().get_device("cuda:0")

    def compute() -> str:
        arch = int(device().arch)
        return f"{arch // 10}.{arch % 10} (sm_{arch})"

    def memory() -> str:
        return f"{device().total_memory / 2**20:.0f} MB"

    return {
        "warp_version": version,
        "warp_location": lambda: getattr(need(), "__file__", None) or "unknown",
        "cuda_driver": driver,
        "cudart": cudart,
        "gpu_name": lambda: device().name,
        "gpu_compute": compute,
        "gpu_memory": memory,
    }


def system_probes() -> dict[str, Callable[[], Any]]:
    return {
        "os": lambda: f"{platform.system()} {platform.release()} ({platform.version()})",
        "python": lambda: f"{sys.version.split()[0]} {platform.machine()}",
    }


# ---- first-run hardware check (works before Warp is installed) -----------------------------

_SMI_QUERY = "name,driver_version,memory.total,compute_cap"
_SMI_TIMEOUT = 8.0


def _smi_executable() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    legacy = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA Corporation" / "NVSMI" / "nvidia-smi.exe"
    return str(legacy) if legacy.is_file() else None


def nvidia_smi_info(run: Callable[..., Any] = subprocess.run) -> dict[str, str]:
    """GPU facts from the NVIDIA driver's ``nvidia-smi``: no Warp, no CUDA, no Blender needed.

    Returns the SECTIONS keys it can fill (``gpu_name``, ``gpu_compute``,
    ``gpu_memory``, ``cuda_driver``) plus ``driver_version`` and ``error``.
    Blocking (about 100 ms); call it from a worker thread, never from a draw.
    Never raises.
    """
    exe = _smi_executable()
    if exe is None:
        return {"error": "nvidia-smi not found: no NVIDIA driver installed?"}
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        rows = run([exe, f"--query-gpu={_SMI_QUERY}", "--format=csv,noheader,nounits"], capture_output=True,
                   text=True, timeout=_SMI_TIMEOUT, creationflags=flags, stdin=subprocess.DEVNULL)
        if rows.returncode != 0 or not rows.stdout.strip():
            return {"error": f"nvidia-smi failed: {(rows.stderr or rows.stdout).strip()[:200] or rows.returncode}"}
        gpus = [[cell.strip() for cell in line.split(",")] for line in rows.stdout.strip().splitlines()]
        name, driver, memory, compute = (gpus[0] + ["", "", "", ""])[:4]
        info = {"gpu_name": name + (f" (+{len(gpus) - 1} more)" if len(gpus) > 1 else ""),
                "driver_version": driver}
        if memory.isdigit():
            info["gpu_memory"] = f"{int(memory)} MB"
        if compute:
            info["gpu_compute"] = compute
        header = run([exe], capture_output=True, text=True, timeout=_SMI_TIMEOUT, creationflags=flags,
                     stdin=subprocess.DEVNULL)
        found = re.search(r"CUDA (?:UMD )?Version:\s*(\d+\.\d+)", header.stdout or "")
        if found:
            info["cuda_driver"] = found.group(1)
        return info
    except Exception as exc:
        return {"error": unavailable(exc)}


@dataclass(frozen=True)
class HardwareSummary:
    rows: tuple[tuple[str, str], ...]   # (label, value) in display order
    verdict: str
    ready: bool | None                  # None: not checked yet


def hardware_summary(smi: Mapping[str, str] | None, backend: str, wp: Any = None) -> HardwareSummary:
    """GPU, driver, backend and VRAM for the first-run panel.

    Reads ``warp_probes(wp)`` when Warp is already active (its values are the
    ones the solver will use); otherwise falls back to the ``nvidia-smi`` facts.
    """
    if smi is None:
        return HardwareSummary((("Graphics backend", backend),), "Checking the GPU...", None)
    probes = warp_probes(wp) if wp is not None else {}

    def pick(key: str) -> str | None:
        if key in probes:
            value = _resolve(probes[key])
            if not value.startswith("unavailable"):
                return value
        return smi.get(key)

    cuda = pick("cuda_driver")
    rows = (
        ("GPU", pick("gpu_name") or "not found"),
        ("NVIDIA driver", smi.get("driver_version") or "unknown"),
        ("Driver supports CUDA", cuda or "unknown"),
        ("GPU memory (VRAM)", pick("gpu_memory") or "unknown"),
        ("Graphics backend", backend),
    )
    needed = CERTIFIED_CUDA_MAJOR
    if not pick("gpu_name"):
        return HardwareSummary(rows, smi.get("error") or "No NVIDIA GPU found: WarpSPH needs one", False)
    try:
        too_old = cuda is not None and int(cuda.split(".")[0]) < needed
    except ValueError:
        too_old = False
    if too_old:
        return HardwareSummary(rows, f"Driver only supports CUDA {cuda}; Warp needs {needed}.x: update the NVIDIA driver", False)
    if cuda is None:
        return HardwareSummary(rows, "GPU found; the driver did not report its CUDA version", True)
    return HardwareSummary(rows, f"Ready for Warp: the driver supports CUDA {cuda} (needs {needed}.x or newer)", True)


# ---- logging -------------------------------------------------------------------------------

def _package_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(name or __package__ or "warp_sph_realtime")


def teardown_logging(logger: logging.Logger | None = None) -> None:
    logger = logger or _package_logger()
    for handler in list(logger.handlers):
        if getattr(handler, _HANDLER_TAG, False):
            logger.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass


def setup_logging(log_dir: str | os.PathLike, logger: logging.Logger | None = None) -> Path | None:
    """Attach one rotating file handler; safe to call repeatedly. Returns the log path."""
    logger = logger or _package_logger()
    teardown_logging(logger)
    try:
        directory = Path(log_dir)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / LOG_FILE_NAME
        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUPS, encoding="utf-8", delay=True)
    except Exception:
        return None  # logging must never break registration
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    setattr(handler, _HANDLER_TAG, True)
    logger.addHandler(handler)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    return path


def log_path(logger: logging.Logger | None = None) -> Path | None:
    for handler in (logger or _package_logger()).handlers:
        if getattr(handler, _HANDLER_TAG, False):
            return Path(handler.baseFilename)
    return None
