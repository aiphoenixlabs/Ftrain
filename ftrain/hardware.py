"""
FTRAIN Hardware Intelligence
============================

One place that knows what machine FTRAIN is running on and what it should
do about it. Supports:

- NVIDIA CUDA (including multi-GPU DDP rank pinning by the caller)
- AMD ROCm (torch.version.hip builds)
- Intel XPU / Arc (torch.xpu, torch >= 2.4)
- Apple Silicon MPS (macOS)
- DirectML (Windows AMD/Intel via torch-directml)
- CPU (always available; thread-aware recommendations)

Everything here is probe-based and defensive: every accelerator API is
checked with hasattr/try so the same code runs everywhere. torch itself is
the only dependency.
"""

from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

__all__ = [
    "HardwareProfile",
    "detect",
    "best_device",
    "preferred_dtype",
    "autocast_spec",
    "scaler_device",
    "offload_plan",
    "recommend_training",
]


@dataclass
class HardwareProfile:
    """What FTRAIN knows about the current machine."""

    backend: str                       # cuda | rocm | xpu | mps | dml | cpu
    device_name: str = "cpu"
    device_count: int = 1
    memory_total_bytes: Optional[int] = None
    memory_available_bytes: Optional[int] = None
    compute_capability: Optional[Tuple[int, int]] = None
    bf16: bool = False
    fp16: bool = False
    torch_version: str = ""
    os_name: str = ""
    cpu_threads: int = 1
    notes: List[str] = field(default_factory=list)

    @property
    def is_accelerator(self) -> bool:
        return self.backend != "cpu"

    def summary(self) -> str:
        mem = ""
        if self.memory_total_bytes:
            mem = f", {self.memory_total_bytes / (1024 ** 3):.1f} GB"
        return (
            f"{self.backend}: {self.device_name}{mem} "
            f"(bf16={self.bf16}, fp16={self.fp16})"
        )


# =============================================================================
# Probes
# =============================================================================

def _cuda_profile() -> Optional[HardwareProfile]:
    if not torch.cuda.is_available():
        return None

    is_rocm = bool(getattr(torch.version, "hip", None))
    index = 0

    try:
        props = torch.cuda.get_device_properties(index)
        name = props.name
        total = int(props.total_memory)
        count = torch.cuda.device_count()
    except Exception:
        name, total, count = "unknown GPU", None, 1

    capability = None
    try:
        capability = tuple(torch.cuda.get_device_capability(index))
    except Exception:
        pass

    bf16 = False
    try:
        bf16 = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16 = capability is not None and capability >= (8, 0)

    free = None
    try:
        free_total = torch.cuda.mem_get_info(index)
        free = int(free_total[0])
    except Exception:
        pass

    return HardwareProfile(
        backend="rocm" if is_rocm else "cuda",
        device_name=name,
        device_count=count,
        memory_total_bytes=total,
        memory_available_bytes=free,
        compute_capability=capability,
        bf16=bf16,
        fp16=True,
        torch_version=torch.__version__,
        os_name=platform.system(),
        cpu_threads=os.cpu_count() or 1,
    )


def _xpu_profile() -> Optional[HardwareProfile]:
    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        return None
    try:
        if not xpu.is_available():
            return None
        count = xpu.device_count()
        name = "Intel XPU"
        try:
            name = xpu.get_device_name(0)
        except Exception:
            pass
        total = None
        try:
            props = xpu.get_device_properties(0)
            total = int(getattr(props, "total_memory", 0)) or None
        except Exception:
            pass
        bf16 = True  # Intel PVC/Arc support bf16 in torch 2.4+
        return HardwareProfile(
            backend="xpu",
            device_name=name,
            device_count=count,
            memory_total_bytes=total,
            bf16=bf16,
            fp16=True,
            torch_version=torch.__version__,
            os_name=platform.system(),
            cpu_threads=os.cpu_count() or 1,
        )
    except Exception:
        return None


def _mps_profile() -> Optional[HardwareProfile]:
    try:
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return None
    except Exception:
        return None

    total = None
    try:
        import subprocess

        output = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if output.returncode == 0:
            total = int(output.stdout.strip())
    except Exception:
        pass

    return HardwareProfile(
        backend="mps",
        device_name=f"Apple Silicon ({platform.machine()})",
        device_count=1,
        memory_total_bytes=total,
        # Unified memory: torch reports ~70% of RAM as usable headroom.
        memory_available_bytes=int(total * 0.7) if total else None,
        bf16=True,   # MPS bf16 supported on macOS 14+ / M2+; autocast fp16 default
        fp16=True,
        torch_version=torch.__version__,
        os_name=platform.system(),
        cpu_threads=os.cpu_count() or 1,
        notes=["unified memory: leave ~30% headroom for the OS"],
    )


def _dml_profile() -> Optional[HardwareProfile]:
    try:
        import torch_directml  # type: ignore

        count = torch_directml.device_count()
        if count <= 0:
            return None
        return HardwareProfile(
            backend="dml",
            device_name="DirectML device",
            device_count=count,
            bf16=False,
            fp16=True,
            torch_version=torch.__version__,
            os_name=platform.system(),
            cpu_threads=os.cpu_count() or 1,
        )
    except Exception:
        return None


def _cpu_profile() -> HardwareProfile:
    profile = HardwareProfile(
        backend="cpu",
        device_name=platform.processor() or "cpu",
        device_count=1,
        bf16=True,
        fp16=False,
        torch_version=torch.__version__,
        os_name=platform.system(),
        cpu_threads=os.cpu_count() or 1,
    )
    try:
        import multiprocessing

        profile.cpu_threads = multiprocessing.cpu_count()
    except Exception:
        pass
    return profile


def detect() -> HardwareProfile:
    """Probe the machine in accelerator-priority order."""
    for probe in (_cuda_profile, _xpu_profile, _mps_profile, _dml_profile):
        try:
            profile = probe()
        except Exception:
            logger.debug("Hardware probe failed: %s", probe.__name__, exc_info=True)
            profile = None
        if profile is not None:
            return profile
    return _cpu_profile()


# =============================================================================
# Device / dtype selection
# =============================================================================

def best_device(
    *,
    local_rank: int = 0,
    distributed: bool = False,
) -> torch.device:
    """The single device this process should own.

    Under real DDP the caller pins each rank to its own CUDA device; that
    logic stays with the caller. On every other backend FTRAIN owns exactly
    one device.
    """
    profile = detect()

    if profile.backend in ("cuda", "rocm"):
        count = max(1, profile.device_count)
        if distributed and count > 1:
            index = min(max(0, int(local_rank)), count - 1)
            return torch.device(f"cuda:{index}")
        return torch.device("cuda:0")

    if profile.backend == "xpu":
        return torch.device("xpu", 0)

    if profile.backend == "mps":
        return torch.device("mps")

    if profile.backend == "dml":
        try:
            import torch_directml  # type: ignore

            return torch_directml.device(min(local_rank, max(0, profile.device_count - 1)))
        except Exception:
            pass

    return torch.device("cpu")


def preferred_dtype(profile: HardwareProfile) -> torch.dtype:
    """Best training dtype for this machine."""
    if profile.backend in ("cuda", "rocm"):
        return torch.bfloat16 if profile.bf16 else torch.float16
    if profile.backend == "xpu":
        return torch.bfloat16 if profile.bf16 else torch.float16
    if profile.backend == "mps":
        return torch.float16
    return torch.float32


def autocast_spec(device: torch.device, bf16_supported: bool = True) -> Tuple[bool, str, torch.dtype]:
    """(enabled, device_type, dtype) for torch.autocast on this backend."""
    kind = device.type

    if kind in ("cuda",):
        dtype = torch.bfloat16 if bf16_supported else torch.float16
        return True, "cuda", dtype

    if kind == "xpu":
        return True, "xpu", torch.bfloat16

    if kind == "mps":
        return True, "mps", torch.float16

    return False, "cpu", torch.float32


def scaler_device(device: torch.device) -> Optional[str]:
    """GradScaler backend for fp16 runs; None where scaling is unsupported."""
    if device.type == "cuda":
        return "cuda"
    if device.type == "mps":
        return "mps"
    if device.type == "xpu":
        return "xpu"
    return None


# =============================================================================
# Memory planning / intelligent offload
# =============================================================================

_GB = 1024 ** 3


def offload_plan(
    profile: HardwareProfile,
    model_bytes: int,
    *,
    training: bool = True,
) -> Dict[str, Any]:
    """Decide where a model of ``model_bytes`` should live.

    Training keeps the single-device invariant (no sharding); inference
    (merge/captain/eval) may use accelerate ``device_map="auto"`` to spill
    into unified/system memory instead of failing.
    """
    plan: Dict[str, Any] = {
        "strategy": "single_device",
        "device_map": None,
        "load_in_4bit": False,
        "gradient_checkpointing": False,
        "warning": None,
    }

    available = profile.memory_available_bytes or profile.memory_total_bytes

    if available is None:
        return plan

    headroom = int(available * 0.85)

    if model_bytes <= headroom:
        return plan

    four_bit_bytes = int(model_bytes * 0.30)

    if training:
        if four_bit_bytes <= headroom:
            plan["load_in_4bit"] = True
            plan["gradient_checkpointing"] = True
            plan["warning"] = (
                f"model needs ~{model_bytes / _GB:.1f} GB but only "
                f"{available / _GB:.1f} GB is usable; FTRAIN recommends "
                "4-bit quantization + gradient checkpointing"
            )
        else:
            plan["strategy"] = "cpu_or_disk"
            plan["warning"] = (
                f"model needs ~{model_bytes / _GB:.1f} GB even in 4-bit "
                f"({four_bit_bytes / _GB:.1f} GB) — train on a larger "
                "accelerator, or on CPU with a small model"
            )
        return plan

    # Inference-only: allow sharded placement across devices.
    plan["strategy"] = "auto_shard"
    plan["device_map"] = "auto"
    plan["warning"] = (
        f"model needs ~{model_bytes / _GB:.1f} GB; using accelerate "
        "auto-sharding (inference only)"
    )
    return plan


def recommend_training(profile: HardwareProfile, *, load_in_4bit: bool = False) -> Dict[str, Any]:
    """Actionable per-backend training recommendations."""
    rec: Dict[str, Any] = {
        "backend": profile.backend,
        "dtype": str(preferred_dtype(profile)).replace("torch.", ""),
        "attention": "sdpa",
        "gradient_checkpointing": False,
        "batch_size_hint": "keep as configured",
        "pin_memory": profile.backend == "cuda",
        "notes": list(profile.notes),
    }

    if profile.backend == "cpu":
        rec["attention"] = "eager"
        rec["gradient_checkpointing"] = False
        rec["batch_size_hint"] = "1-2; CPU training is for tiny models only"
        rec["notes"].append(f"{profile.cpu_threads} threads detected")
    elif profile.backend == "mps":
        rec["gradient_checkpointing"] = True
        rec["batch_size_hint"] = "1-4; MPS fp16 + checkpointing is the stable combo"
        rec["notes"].append("avoid float64 ops; some kernels fall back to CPU")
    elif profile.backend == "rocm":
        rec["notes"].append("ROCm: ensure torch was built with HIP support")
    elif profile.backend == "dml":
        rec["gradient_checkpointing"] = True
        rec["batch_size_hint"] = "1-2; DirectML does not support all training ops"
        rec["notes"].append("DirectML is inference-leaning; training support is partial")

    if load_in_4bit and profile.backend not in ("cuda", "rocm"):
        rec["notes"].append(
            "load_in_4bit requires bitsandbytes (NVIDIA); it is ignored on "
            "this backend"
        )

    return rec


# =============================================================================
# Maximum power mode
# =============================================================================

def maximum_power_plan(
    profile: HardwareProfile,
    *,
    training: bool,
    model_bytes: Optional[int] = None,
    current_batch: int = 1,
    seq_length: int = 512,
) -> Dict[str, Any]:
    """
    Plan how to extract maximum throughput from this machine.

    ``maximum_power=True`` is an explicit user consent to trade memory
    headroom and thermal comfort for speed. The plan never touches the
    learning rate or anything that affects correctness — only throughput.
    """
    plan: Dict[str, Any] = {
        "backend": profile.backend,
        "dtype": str(preferred_dtype(profile)).replace("torch.", ""),
        "tf32": profile.backend == "cuda",
        "cudnn_benchmark": profile.backend in ("cuda", "rocm"),
        "attention": "sdpa",
        "pin_memory": profile.backend == "cuda",
        "gradient_checkpointing": None,      # None = leave user's setting
        "load_in_4bit": None,                # None = leave user's setting
        "suggested_batch": current_batch,
        "suggested_workers": None,
        "torch_compile": False,
        "warnings": [],
        "applied": [],
    }

    if profile.backend == "cpu":
        plan["warnings"].append(
            "CPU has no hidden performance to unlock; maximum_power applies "
            "no throughput changes"
        )
        return plan

    # 1. Memory-driven decisions (CUDA-family only; MPS unified memory is
    #    handled conservatively because freeing is lazy on macOS).
    usable = profile.memory_available_bytes or profile.memory_total_bytes
    if usable is not None and model_bytes is not None and profile.backend in ("cuda", "rocm"):
        headroom = usable - model_bytes

        if headroom > model_bytes * 0.8:
            # Plenty of room: checkpointing costs ~20-30% speed; turn it off.
            plan["gradient_checkpointing"] = False
            plan["applied"].append("gradient_checkpointing disabled (VRAM allows)")
            plan["load_in_4bit"] = False
            plan["applied"].append("4-bit quantization disabled (full-precision is faster)")
        elif headroom < model_bytes * 0.25:
            plan["gradient_checkpointing"] = True
            plan["applied"].append("gradient_checkpointing enabled (tight VRAM)")
            plan["warnings"].append(
                "VRAM is tight; batch was NOT increased automatically"
            )
        else:
            plan["gradient_checkpointing"] = False

        if headroom > model_bytes * 0.5 and training:
            boost = max(1, int(headroom // max(1, model_bytes // 4)))
            suggested = max(current_batch, min(boost, 32))
            if suggested > current_batch:
                plan["suggested_batch"] = suggested
                plan["applied"].append(
                    f"per_device_batch_size {current_batch} → {suggested}"
                )

    # 2. Dataloader workers: parallel feeding matters once the GPU is fast.
    if profile.backend in ("cuda", "rocm", "xpu"):
        workers = max(2, min(4, (profile.cpu_threads or 4) - 1))
        if os.name == "nt":
            workers = min(workers, 2)  # Windows spawn overhead
        plan["suggested_workers"] = workers

    # 3. torch.compile: NVIDIA-only in practice; opt-in inside the plan.
    if profile.backend == "cuda" and profile.compute_capability and profile.compute_capability >= (8, 0):
        plan["torch_compile"] = True

    if profile.backend == "mps":
        plan["warnings"].append(
            "MPS: maximum power uses fp16 + larger batches; torch.compile "
            "and TF32 do not apply"
        )
    if profile.backend == "dml":
        plan["warnings"].append(
            "DirectML: limited operator coverage; throughput gains are modest"
        )

    return plan


def apply_maximum_power(plan: Dict[str, Any]) -> Dict[str, bool]:
    """Execute the global (non-config) parts of a maximum-power plan."""
    applied: Dict[str, bool] = {}
    try:
        from .speed import flash_mode

        status = flash_mode(
            enabled=bool(plan.get("cudnn_benchmark", False)),
            tf32=bool(plan.get("tf32", False)),
        )
        applied["flash_mode"] = any(
            v for k, v in status.items() if isinstance(v, bool)
        )
        applied["backend"] = status.get("backend") == "cpu"
    except Exception:
        logger.debug("maximum_power: flash_mode unavailable.", exc_info=True)
        applied["flash_mode"] = False
    return applied
