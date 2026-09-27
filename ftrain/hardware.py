"""
FTRAIN Hardware Intelligence
=============================

Hardware detection + training capability + Unsloth intelligence.

Goals:
    1. Detect the real compute backend.
    2. Inspect every visible accelerator.
    3. Measure VRAM / unified memory where possible.
    4. Detect BF16 / FP16 capability.
    5. Detect multi-GPU availability.
    6. Detect the installed Unsloth stack.
    7. Decide whether Unsloth should actually be used.
    8. Separate:
         - inference capability
         - training capability
         - Unsloth acceleration capability
         - quantized/QLoRA capability
    9. Produce a safe recommendation for FTRAIN Core.
   10. Never enable model sharding during normal training merely because
       multiple GPUs exist.

Supported backend families:
    - NVIDIA CUDA
    - AMD ROCm
    - Intel XPU
    - Apple MPS
    - DirectML
    - CPU

The module is intentionally defensive. Optional dependencies such as
Unsloth and torch-directml are never required for import.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import os
import platform
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Dict, List, Optional, Tuple

import torch


logger = logging.getLogger(__name__)


__all__ = [
    # Profiles
    "GPUInfo",
    "UnslothStatus",
    "HardwareProfile",

    # Detection
    "detect",
    "detect_all_gpus",
    "detect_unsloth",

    # Device selection
    "best_device",
    "best_devices",

    # Precision
    "preferred_dtype",
    "autocast_spec",
    "scaler_device",

    # Memory / planning
    "offload_plan",
    "estimate_model_memory",
    "recommend_training",

    # High level
    "training_plan",
    "hardware_report",
]


# =============================================================================
# Constants
# =============================================================================

_GB = 1024 ** 3
_MB = 1024 ** 2


# =============================================================================
# GPU INFORMATION
# =============================================================================

@dataclass
class GPUInfo:
    """
    Information about one visible accelerator.
    """

    index: int

    name: str = "unknown"

    backend: str = "unknown"

    total_memory_bytes: Optional[int] = None
    free_memory_bytes: Optional[int] = None

    compute_capability: Optional[Tuple[int, int]] = None

    bf16: bool = False
    fp16: bool = False

    multi_processor_count: Optional[int] = None

    capability_label: str = ""

    @property
    def total_memory_gb(self) -> Optional[float]:
        if self.total_memory_bytes is None:
            return None

        return self.total_memory_bytes / _GB

    @property
    def free_memory_gb(self) -> Optional[float]:
        if self.free_memory_bytes is None:
            return None

        return self.free_memory_bytes / _GB

    @property
    def compute_capability_value(self) -> float:
        if self.compute_capability is None:
            return 0.0

        major, minor = self.compute_capability

        return major + minor / 10.0


# =============================================================================
# UNSLOTH STATUS
# =============================================================================

@dataclass
class UnslothStatus:
    """
    What FTRAIN knows about the installed Unsloth stack.
    """

    installed: bool = False

    version: str = ""

    importable: bool = False

    fast_language_model: bool = False

    backend_supported: bool = False

    training_supported: bool = False

    quantized_training_supported: bool = False

    multi_gpu_supported: bool = False

    recommended: bool = False

    reason: str = ""

    warnings: List[str] = field(
        default_factory=list
    )

    features: List[str] = field(
        default_factory=list
    )


# =============================================================================
# HARDWARE PROFILE
# =============================================================================

@dataclass
class HardwareProfile:
    """
    Complete hardware intelligence profile.
    """

    # ------------------------------------------------------------------
    # General platform
    # ------------------------------------------------------------------

    backend: str

    device_name: str = "cpu"

    device_count: int = 1

    gpus: List[GPUInfo] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Primary device memory
    # ------------------------------------------------------------------

    memory_total_bytes: Optional[int] = None

    memory_available_bytes: Optional[int] = None

    # ------------------------------------------------------------------
    # Compute
    # ------------------------------------------------------------------

    compute_capability: Optional[
        Tuple[int, int]
    ] = None

    bf16: bool = False

    fp16: bool = False

    # ------------------------------------------------------------------
    # Software
    # ------------------------------------------------------------------

    torch_version: str = ""

    cuda_version: Optional[str] = None

    rocm_version: Optional[str] = None

    os_name: str = ""

    architecture: str = ""

    cpu_threads: int = 1

    # ------------------------------------------------------------------
    # Optional stacks
    # ------------------------------------------------------------------

    unsloth: UnslothStatus = field(
        default_factory=UnslothStatus
    )

    bitsandbytes: bool = False

    triton: bool = False

    flash_attn: bool = False

    accelerate: bool = False

    transformers: bool = False

    trl: bool = False

    # ------------------------------------------------------------------
    # Notes
    # ------------------------------------------------------------------

    notes: List[str] = field(
        default_factory=list
    )

    warnings: List[str] = field(
        default_factory=list
    )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_accelerator(self) -> bool:

        return self.backend != "cpu"

    @property
    def total_memory_gb(self) -> Optional[float]:

        if self.memory_total_bytes is None:
            return None

        return self.memory_total_bytes / _GB

    @property
    def available_memory_gb(self) -> Optional[float]:

        if self.memory_available_bytes is None:
            return None

        return self.memory_available_bytes / _GB

    @property
    def supports_multi_gpu(self) -> bool:

        return self.device_count > 1

    @property
    def supports_bf16(self) -> bool:

        return self.bf16

    @property
    def supports_fp16(self) -> bool:

        return self.fp16

    @property
    def can_use_unsloth(self) -> bool:

        return self.unsloth.recommended

    def summary(self) -> str:

        memory = ""

        if self.memory_total_bytes:

            memory = (
                f", "
                f"{self.memory_total_bytes / _GB:.1f} GB"
            )

        return (
            f"{self.backend}: "
            f"{self.device_name}"
            f"{memory} "
            f"(bf16={self.bf16}, "
            f"fp16={self.fp16}, "
            f"GPUs={self.device_count}, "
            f"Unsloth={self.can_use_unsloth})"
        )


# =============================================================================
# PACKAGE DETECTION
# =============================================================================

def _package_installed(
    package_name: str,
) -> bool:

    try:

        return (
            importlib.util.find_spec(
                package_name
            )
            is not None
        )

    except Exception:

        return False


def _package_version(
    package_name: str,
) -> str:

    try:

        return version(
            package_name
        )

    except PackageNotFoundError:

        return ""

    except Exception:

        return ""


# =============================================================================
# CUDA / ROCm
# =============================================================================

def _cuda_profile() -> Optional[HardwareProfile]:

    if not torch.cuda.is_available():

        return None

    is_rocm = bool(
        getattr(
            torch.version,
            "hip",
            None,
        )
    )

    backend = (
        "rocm"
        if is_rocm
        else "cuda"
    )

    count = max(
        1,
        torch.cuda.device_count(),
    )

    gpus: List[GPUInfo] = []

    for index in range(count):

        try:

            props = (
                torch.cuda.get_device_properties(
                    index
                )
            )

            name = str(
                props.name
            )

            total = int(
                props.total_memory
            )

            capability = None

            try:

                capability = tuple(
                    torch.cuda.get_device_capability(
                        index
                    )
                )

            except Exception:
                pass

            # ------------------------------------------------------
            # BF16
            # ------------------------------------------------------

            bf16 = False

            try:

                with torch.cuda.device(
                    index
                ):

                    bf16 = bool(
                        torch.cuda.is_bf16_supported()
                    )

            except Exception:

                if capability is not None:

                    bf16 = (
                        capability
                        >= (8, 0)
                    )

            # ------------------------------------------------------
            # FP16
            # ------------------------------------------------------

            fp16 = True

            # ------------------------------------------------------
            # Free memory
            # ------------------------------------------------------

            free = None

            try:

                free, _total = (
                    torch.cuda.mem_get_info(
                        index
                    )
                )

                free = int(
                    free
                )

            except Exception:
                pass

            gpus.append(
                GPUInfo(
                    index=index,
                    name=name,
                    backend=backend,
                    total_memory_bytes=total,
                    free_memory_bytes=free,
                    compute_capability=capability,
                    bf16=bf16,
                    fp16=fp16,
                    multi_processor_count=int(
                        getattr(
                            props,
                            "multi_processor_count",
                            0,
                        )
                    ),
                    capability_label=(
                        (
                            f"sm_{capability[0]}"
                            f"{capability[1]}"
                        )
                        if capability is not None
                        else ""
                    ),
                )
            )

        except Exception as exc:

            logger.debug(
                "GPU %s probe failed: %s",
                index,
                exc,
            )

            gpus.append(
                GPUInfo(
                    index=index,
                    backend=backend,
                    name="unknown GPU",
                )
            )

    primary = (
        gpus[0]
        if gpus
        else GPUInfo(
            index=0,
            backend=backend,
        )
    )

    total = primary.total_memory_bytes

    free = primary.free_memory_bytes

    return HardwareProfile(
        backend=backend,
        device_name=primary.name,
        device_count=count,
        gpus=gpus,
        memory_total_bytes=total,
        memory_available_bytes=free,
        compute_capability=primary.compute_capability,
        bf16=primary.bf16,
        fp16=primary.fp16,
        torch_version=torch.__version__,
        cuda_version=getattr(
            torch.version,
            "cuda",
            None,
        ),
        rocm_version=getattr(
            torch.version,
            "hip",
            None,
        ),
        os_name=platform.system(),
        architecture=platform.machine(),
        cpu_threads=os.cpu_count() or 1,
    )


# =============================================================================
# INTEL XPU
# =============================================================================

def _xpu_profile() -> Optional[HardwareProfile]:

    xpu = getattr(
        torch,
        "xpu",
        None,
    )

    if xpu is None:
        return None

    try:

        if not xpu.is_available():
            return None

        count = max(
            1,
            xpu.device_count(),
        )

        gpus: List[
            GPUInfo
        ] = []

        for index in range(count):

            name = "Intel XPU"

            try:

                name = str(
                    xpu.get_device_name(
                        index
                    )
                )

            except Exception:
                pass

            total = None

            try:

                props = (
                    xpu.get_device_properties(
                        index
                    )
                )

                total = int(
                    getattr(
                        props,
                        "total_memory",
                        0,
                    )
                ) or None

            except Exception:
                pass

            bf16 = False

            # Don't blindly claim BF16.
            # Ask PyTorch's device implementation first.
            try:

                if hasattr(
                    torch,
                    "xpu",
                ) and hasattr(
                    torch.xpu,
                    "is_bf16_supported",
                ):

                    bf16 = bool(
                        torch.xpu.is_bf16_supported()
                    )

            except Exception:
                pass

            gpus.append(
                GPUInfo(
                    index=index,
                    name=name,
                    backend="xpu",
                    total_memory_bytes=total,
                    bf16=bf16,
                    fp16=True,
                )
            )

        primary = gpus[0]

        return HardwareProfile(
            backend="xpu",
            device_name=primary.name,
            device_count=count,
            gpus=gpus,
            memory_total_bytes=primary.total_memory_bytes,
            bf16=primary.bf16,
            fp16=True,
            torch_version=torch.__version__,
            os_name=platform.system(),
            architecture=platform.machine(),
            cpu_threads=os.cpu_count() or 1,
        )

    except Exception as exc:

        logger.debug(
            "XPU probe failed: %s",
            exc,
        )

        return None


# =============================================================================
# APPLE MPS
# =============================================================================

def _mps_profile() -> Optional[HardwareProfile]:

    try:

        mps = getattr(
            torch.backends,
            "mps",
            None,
        )

        if (
            mps is None
            or not mps.is_available()
        ):

            return None

    except Exception:

        return None

    total = None

    try:

        # macOS unified memory.
        if platform.system() == "Darwin":

            import subprocess

            output = subprocess.run(
                [
                    "sysctl",
                    "-n",
                    "hw.memsize",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )

            if (
                output.returncode == 0
                and output.stdout.strip()
            ):

                total = int(
                    output.stdout.strip()
                )

    except Exception:
        pass

    # Don't claim BF16 universally.
    # MPS support depends on hardware/macOS/PyTorch.
    bf16 = False

    try:

        # Current PyTorch can report support indirectly through
        # tensor operations, but this is intentionally conservative.
        test = torch.ones(
            1,
            device="mps",
            dtype=torch.bfloat16,
        )

        del test

        bf16 = True

    except Exception:

        bf16 = False

    gpu = GPUInfo(
        index=0,
        name=(
            f"Apple Silicon "
            f"({platform.machine()})"
        ),
        backend="mps",
        total_memory_bytes=total,
        free_memory_bytes=None,
        bf16=bf16,
        fp16=True,
    )

    profile = HardwareProfile(
        backend="mps",
        device_name=gpu.name,
        device_count=1,
        gpus=[gpu],
        memory_total_bytes=total,
        # Unified memory is shared with OS.
        memory_available_bytes=(
            int(total * 0.70)
            if total
            else None
        ),
        bf16=bf16,
        fp16=True,
        torch_version=torch.__version__,
        os_name=platform.system(),
        architecture=platform.machine(),
        cpu_threads=os.cpu_count() or 1,
    )

    profile.notes.append(
        "MPS uses unified memory; reserve headroom for macOS."
    )

    return profile


# =============================================================================
# DIRECTML
# =============================================================================

def _dml_profile() -> Optional[HardwareProfile]:

    try:

        import torch_directml  # type: ignore

        count = int(
            torch_directml.device_count()
        )

        if count <= 0:
            return None

        return HardwareProfile(
            backend="dml",
            device_name="DirectML device",
            device_count=count,
            gpus=[
                GPUInfo(
                    index=i,
                    backend="dml",
                    name="DirectML device",
                    fp16=True,
                )
                for i in range(count)
            ],
            bf16=False,
            fp16=True,
            torch_version=torch.__version__,
            os_name=platform.system(),
            architecture=platform.machine(),
            cpu_threads=os.cpu_count() or 1,
        )

    except Exception:

        return None


# =============================================================================
# CPU
# =============================================================================

def _cpu_profile() -> HardwareProfile:

    threads = (
        os.cpu_count()
        or 1
    )

    try:

        import multiprocessing

        threads = (
            multiprocessing.cpu_count()
        )

    except Exception:
        pass

    return HardwareProfile(
        backend="cpu",
        device_name=(
            platform.processor()
            or "CPU"
        ),
        device_count=1,
        bf16=False,
        fp16=False,
        torch_version=torch.__version__,
        os_name=platform.system(),
        architecture=platform.machine(),
        cpu_threads=threads,
    )


# =============================================================================
# INSTALLED STACK
# =============================================================================

def _detect_optional_stack(
    profile: HardwareProfile,
) -> None:

    profile.bitsandbytes = (
        _package_installed(
            "bitsandbytes"
        )
    )

    profile.triton = (
        _package_installed(
            "triton"
        )
    )

    profile.flash_attn = (
        _package_installed(
            "flash_attn"
        )
    )

    profile.accelerate = (
        _package_installed(
            "accelerate"
        )
    )

    profile.transformers = (
        _package_installed(
            "transformers"
        )
    )

    profile.trl = (
        _package_installed(
            "trl"
        )
    )


# =============================================================================
# UNSLOTH DETECTION
# =============================================================================

def detect_unsloth(
    profile: Optional[
        HardwareProfile
    ] = None,
) -> UnslothStatus:
    """
    Determine whether FTRAIN can use Unsloth.

    Important:
        Installed != usable.
        Importable != recommended.

    We therefore evaluate the backend, package import, and
    critical FastLanguageModel entry point.
    """

    if profile is None:
        profile = detect(
            probe_unsloth=False
        )

    status = UnslothStatus()

    # ----------------------------------------------------------
    # Is package installed?
    # ----------------------------------------------------------

    status.installed = (
        _package_installed(
            "unsloth"
        )
    )

    if not status.installed:

        status.reason = (
            "Unsloth is not installed."
        )

        return status

    status.version = (
        _package_version(
            "unsloth"
        )
    )

    # ----------------------------------------------------------
    # Try importing the actual library.
    # ----------------------------------------------------------

    try:

        unsloth = importlib.import_module(
            "unsloth"
        )

        status.importable = True

    except Exception as exc:

        status.reason = (
            "Unsloth is installed but "
            f"cannot be imported: {exc}"
        )

        status.warnings.append(
            "Fix the Unsloth/PyTorch/backend installation before "
            "enabling Unsloth in FTRAIN."
        )

        return status

    # ----------------------------------------------------------
    # FastLanguageModel
    # ----------------------------------------------------------

    try:

        status.fast_language_model = (
            hasattr(
                unsloth,
                "FastLanguageModel",
            )
            or hasattr(
                importlib.import_module(
                    "unsloth.models"
                ),
                "FastLanguageModel",
            )
        )

    except Exception:

        status.fast_language_model = False

    # ----------------------------------------------------------
    # Backend policy
    # ----------------------------------------------------------

    backend = profile.backend

    # Current Unsloth publicly advertises NVIDIA, AMD,
    # Intel, CPU and multi-GPU support.
    #
    # Actual backend usability still depends on the installed
    # wheel/runtime, so we treat package import as the runtime
    # gate rather than inventing a hard minimum matrix.
    if backend in (
        "cuda",
        "rocm",
        "xpu",
        "cpu",
    ):

        status.backend_supported = True

    elif backend == "mps":

        # Unsloth's broad macOS support does not mean every
        # native MPS training path behaves like CUDA.
        status.backend_supported = False

        status.warnings.append(
            "MPS is detected, but FTRAIN does not automatically "
            "select Unsloth acceleration for native MPS training."
        )

    elif backend == "dml":

        status.backend_supported = False

        status.warnings.append(
            "DirectML is not selected as an Unsloth training backend "
            "by FTRAIN."
        )

    else:

        status.backend_supported = False

    # ----------------------------------------------------------
    # Training
    # ----------------------------------------------------------

    status.training_supported = (
        status.importable
        and status.fast_language_model
        and status.backend_supported
    )

    # ----------------------------------------------------------
    # 4-bit / QLoRA
    # ----------------------------------------------------------

    if backend in (
        "cuda",
        "rocm",
        "xpu",
    ):

        # Don't claim QLoRA simply because bitsandbytes exists.
        # It must have a usable package on the current backend.
        status.quantized_training_supported = (
            status.training_supported
            and profile.bitsandbytes
        )

    else:

        status.quantized_training_supported = False

    # ----------------------------------------------------------
    # Multi GPU
    # ----------------------------------------------------------

    status.multi_gpu_supported = (
        status.training_supported
        and profile.device_count > 1
    )

    # ----------------------------------------------------------
    # Final recommendation
    # ----------------------------------------------------------

    status.recommended = (
        status.training_supported
    )

    if status.recommended:

        status.features.extend(
            [
                "FastLanguageModel",
                "accelerated training",
            ]
        )

        if profile.backend in (
            "cuda",
            "rocm",
            "xpu",
        ):

            status.features.append(
                "GPU acceleration"
            )

        if status.quantized_training_supported:

            status.features.append(
                "4-bit / QLoRA candidate"
            )

        if status.multi_gpu_supported:

            status.features.append(
                "multi-GPU candidate"
            )

        status.reason = (
            "Unsloth is installed, importable, "
            "and FTRAIN considers the current backend "
            "eligible."
        )

    else:

        if not status.importable:

            status.reason = (
                "Unsloth import failed."
            )

        elif not status.fast_language_model:

            status.reason = (
                "Unsloth loaded, but "
                "FastLanguageModel is unavailable."
            )

        elif not status.backend_supported:

            status.reason = (
                f"Backend '{backend}' is not enabled "
                "for automatic Unsloth training."
            )

        else:

            status.reason = (
                "Unsloth is present but not considered "
                "safe for automatic acceleration."
            )

    return status


# =============================================================================
# MASTER DETECTOR
# =============================================================================

def detect(
    *,
    probe_unsloth: bool = True,
) -> HardwareProfile:
    """
    Probe the machine.

    Detection order:
        CUDA / ROCm
        XPU
        MPS
        DirectML
        CPU
    """

    profile = None

    probes = (
        _cuda_profile,
        _xpu_profile,
        _mps_profile,
        _dml_profile,
    )

    for probe in probes:

        try:

            profile = probe()

        except Exception:

            logger.debug(
                "Hardware probe failed: %s",
                probe.__name__,
                exc_info=True,
            )

            profile = None

        if profile is not None:
            break

    if profile is None:

        profile = _cpu_profile()

    _detect_optional_stack(
        profile
    )

    if probe_unsloth:

        profile.unsloth = detect_unsloth(
            profile
        )

    return profile


# =============================================================================
# GPU LIST
# =============================================================================

def detect_all_gpus() -> List[GPUInfo]:

    profile = detect(
        probe_unsloth=False
    )

    return list(
        profile.gpus
    )


# =============================================================================
# DEVICE SELECTION
# =============================================================================

def best_device(
    *,
    local_rank: int = 0,
    distributed: bool = False,
) -> torch.device:
    """
    Select the device owned by this process.

    IMPORTANT:
        This does NOT use device_map='auto' for training.

    DDP:
        rank 0 -> cuda:0
        rank 1 -> cuda:1
        ...
    """

    profile = detect(
        probe_unsloth=False
    )

    if profile.backend in (
        "cuda",
        "rocm",
    ):

        count = max(
            1,
            profile.device_count,
        )

        if distributed and count > 1:

            index = min(
                max(
                    0,
                    int(local_rank),
                ),
                count - 1,
            )

            return torch.device(
                f"cuda:{index}"
            )

        return torch.device(
            "cuda:0"
        )

    if profile.backend == "xpu":

        index = min(
            max(
                0,
                int(local_rank),
            ),
            max(
                0,
                profile.device_count - 1,
            ),
        )

        return torch.device(
            f"xpu:{index}"
        )

    if profile.backend == "mps":

        return torch.device(
            "mps"
        )

    if profile.backend == "dml":

        try:

            import torch_directml

            index = min(
                max(
                    0,
                    int(local_rank),
                ),
                max(
                    0,
                    profile.device_count - 1,
                ),
            )

            return (
                torch_directml.device(
                    index
                )
            )

        except Exception:
            pass

    return torch.device(
        "cpu"
    )


def best_devices(
    *,
    distributed: bool = False,
) -> List[torch.device]:
    """
    Return all devices visible to FTRAIN.

    DDP callers can use these for process setup.
    """

    profile = detect(
        probe_unsloth=False
    )

    devices = []

    if profile.backend in (
        "cuda",
        "rocm",
    ):

        for index in range(
            profile.device_count
        ):

            devices.append(
                torch.device(
                    f"cuda:{index}"
                )
            )

    elif profile.backend == "xpu":

        for index in range(
            profile.device_count
        ):

            devices.append(
                torch.device(
                    f"xpu:{index}"
                )
            )

    elif profile.backend == "mps":

        devices.append(
            torch.device(
                "mps"
            )
        )

    elif profile.backend == "dml":

        try:

            import torch_directml

            for index in range(
                profile.device_count
            ):

                devices.append(
                    torch_directml.device(
                        index
                    )
                )

        except Exception:
            pass

    else:

        devices.append(
            torch.device(
                "cpu"
            )
        )

    return devices


# =============================================================================
# DTYPE SELECTION
# =============================================================================

def preferred_dtype(
    profile: HardwareProfile,
) -> torch.dtype:
    """
    Select the preferred training dtype.

    Priority:
        BF16
        FP16
        FP32
    """

    if profile.bf16:

        return torch.bfloat16

    if profile.fp16:

        return torch.float16

    return torch.float32


# =============================================================================
# AUTOCAST
# =============================================================================

def autocast_spec(
    device: torch.device,
    bf16_supported: Optional[
        bool
    ] = None,
) -> Tuple[
    bool,
    str,
    torch.dtype,
]:
    """
    Returns:

        enabled
        device_type
        dtype
    """

    kind = (
        device.type
    )

    if kind == "cuda":

        if bf16_supported is None:

            try:

                bf16_supported = bool(
                    torch.cuda.is_bf16_supported(
                        device=device
                    )
                )

            except Exception:

                bf16_supported = False

        dtype = (
            torch.bfloat16
            if bf16_supported
            else torch.float16
        )

        return (
            True,
            "cuda",
            dtype,
        )

    if kind == "xpu":

        dtype = (
            torch.bfloat16
            if bf16_supported is not False
            else torch.float16
        )

        return (
            True,
            "xpu",
            dtype,
        )

    if kind == "mps":

        return (
            True,
            "mps",
            torch.float16,
        )

    return (
        False,
        "cpu",
        torch.float32,
    )


# =============================================================================
# GRAD SCALER
# =============================================================================

def scaler_device(
    device: torch.device,
) -> Optional[str]:

    if device.type in (
        "cuda",
        "xpu",
        "mps",
    ):

        return device.type

    return None


# =============================================================================
# MODEL MEMORY ESTIMATION
# =============================================================================

def estimate_model_memory(
    parameters: int,
    *,
    dtype: torch.dtype = torch.bfloat16,
    training: bool = True,
    optimizer: str = "adamw",
    lora: bool = False,
) -> Dict[str, float]:
    """
    Estimate approximate model memory.

    This is a planning heuristic, not a substitute for an actual
    memory measurement.

    Examples:
        BF16 inference
        BF16 full fine-tuning
        LoRA
        QLoRA
    """

    bytes_per_param = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
        torch.int8: 1,
        torch.uint8: 1,
    }.get(
        dtype,
        2,
    )

    weight_bytes = (
        parameters
        * bytes_per_param
    )

    # ----------------------------------------------------------
    # Inference
    # ----------------------------------------------------------

    if not training:

        return {
            "weights_gb":
                weight_bytes / _GB,

            "gradients_gb":
                0.0,

            "optimizer_gb":
                0.0,

            "estimated_total_gb":
                weight_bytes / _GB,
        }

    # ----------------------------------------------------------
    # LoRA / QLoRA
    # ----------------------------------------------------------

    if lora:

        # LoRA does not maintain full gradients/optimizer states
        # for all base-model parameters.
        #
        # This intentionally remains conservative because activations
        # can dominate training memory.
        gradient_bytes = (
            parameters
            * 0.03
            * 2
        )

        optimizer_bytes = (
            parameters
            * 0.03
            * 8
        )

    else:

        gradient_bytes = (
            weight_bytes
        )

        if optimizer.lower() in (
            "adam",
            "adamw",
        ):

            # 2 FP32 moment states.
            optimizer_bytes = (
                parameters
                * 8
            )

        elif optimizer.lower() in (
            "sgd",
        ):

            optimizer_bytes = (
                parameters
                * 4
            )

        else:

            optimizer_bytes = (
                parameters
                * 8
            )

    # ----------------------------------------------------------
    # Training overhead
    # ----------------------------------------------------------

    overhead = (
        weight_bytes
        * 0.15
    )

    total = (
        weight_bytes
        + gradient_bytes
        + optimizer_bytes
        + overhead
    )

    return {
        "weights_gb":
            weight_bytes / _GB,

        "gradients_gb":
            gradient_bytes / _GB,

        "optimizer_gb":
            optimizer_bytes / _GB,

        "overhead_gb":
            overhead / _GB,

        "estimated_total_gb":
            total / _GB,
    }


# =============================================================================
# OFFLOAD PLAN
# =============================================================================

def offload_plan(
    profile: HardwareProfile,
    model_bytes: int,
    *,
    training: bool = True,
    use_unsloth: Optional[
        bool
    ] = None,
) -> Dict[str, Any]:
    """
    Create a safe memory plan.

    Critical rule:
        Training does NOT automatically become device_map='auto'.

    Auto-sharding is reserved for inference / evaluation / model
    inspection unless the caller explicitly implements a distributed
    training strategy.
    """

    if use_unsloth is None:

        use_unsloth = (
            profile.can_use_unsloth
        )

    available = (
        profile.memory_available_bytes
        or profile.memory_total_bytes
    )

    plan = {
        "strategy":
            "single_device",

        "device_map":
            None,

        "load_in_4bit":
            False,

        "gradient_checkpointing":
            False,

        "use_unsloth":
            bool(use_unsloth),

        "warning":
            None,
    }

    if available is None:

        if training:

            plan["warning"] = (
                "VRAM/RAM could not be measured; "
                "use conservative settings."
            )

        return plan

    # Keep safety headroom.
    usable = int(
        available
        * 0.85
    )

    # ----------------------------------------------------------
    # Fits directly
    # ----------------------------------------------------------

    if model_bytes <= usable:

        return plan

    # ----------------------------------------------------------
    # 4-bit candidate
    # ----------------------------------------------------------

    four_bit_bytes = int(
        model_bytes
        * 0.30
    )

    if training:

        if (
            four_bit_bytes
            <= usable
        ):

            plan["load_in_4bit"] = (
                profile.backend
                in (
                    "cuda",
                    "rocm",
                    "xpu",
                )
                and profile.bitsandbytes
            )

            plan[
                "gradient_checkpointing"
            ] = True

            plan[
                "warning"
            ] = (
                f"~{model_bytes / _GB:.1f} GB "
                "does not comfortably fit; "
                "using memory-saving training."
            )

            if (
                not plan[
                    "load_in_4bit"
                ]
            ):

                plan[
                    "warning"
                ] += (
                    " 4-bit was not enabled because "
                    "the current backend/install lacks a "
                    "confirmed bitsandbytes path."
                )

        else:

            plan["strategy"] = (
                "larger_accelerator_required"
            )

            plan[
                "warning"
            ] = (
                f"Model requires approximately "
                f"{model_bytes / _GB:.1f} GB; even a "
                f"4-bit estimate is ~"
                f"{four_bit_bytes / _GB:.1f} GB."
            )

        return plan

    # ----------------------------------------------------------
    # Inference-only spilling
    # ----------------------------------------------------------

    plan["strategy"] = (
        "auto_shard_or_offload"
    )

    plan["device_map"] = "auto"

    plan[
        "warning"
    ] = (
        "Inference/evaluation only: "
        "device_map='auto' may be used to spill "
        "layers into available memory."
    )

    return plan


# =============================================================================
# TRAINING RECOMMENDATION
# =============================================================================

def recommend_training(
    profile: HardwareProfile,
    *,
    load_in_4bit: bool = False,
    prefer_unsloth: bool = True,
) -> Dict[str, Any]:
    """
    Produce an actionable FTRAIN training recommendation.
    """

    dtype = preferred_dtype(
        profile
    )

    use_unsloth = (
        prefer_unsloth
        and profile.can_use_unsloth
    )

    recommendation: Dict[
        str,
        Any
    ] = {
        "backend":
            profile.backend,

        "device":
            str(
                best_device()
            ),

        "dtype":
            str(dtype).replace(
                "torch.",
                ""
            ),

        "use_unsloth":
            use_unsloth,

        "unsloth_reason":
            profile.unsloth.reason,

        "quantized_training":
            (
                load_in_4bit
                and
                profile.unsloth
                .quantized_training_supported
            ),

        "gradient_checkpointing":
            False,

        "attention":
            "sdpa",

        "pin_memory":
            profile.backend == "cuda",

        "multi_gpu":
            profile.device_count > 1,

        "distributed_training":
            (
                profile.device_count > 1
                and profile.backend
                in (
                    "cuda",
                    "rocm",
                    "xpu",
                )
            ),

        "notes":
            list(profile.notes),

        "warnings":
            list(profile.warnings),
    }

    # ----------------------------------------------------------
    # Unsloth
    # ----------------------------------------------------------

    if use_unsloth:

        recommendation[
            "notes"
        ].append(
            "FTRAIN should prefer Unsloth's accelerated model/training path."
        )

        if profile.device_count > 1:

            recommendation[
                "notes"
            ].append(
                "Multiple GPUs detected: use distributed/multi-GPU "
                "training, not device_map='auto'."
            )

    # ----------------------------------------------------------
    # CUDA
    # ----------------------------------------------------------

    if profile.backend == "cuda":

        recommendation[
            "gradient_checkpointing"
        ] = False

        recommendation[
            "attention"
        ] = (
            "flash-attention / SDPA"
        )

        if profile.bf16:

            recommendation[
                "notes"
            ].append(
                "BF16 is available and should normally be preferred."
            )

        elif profile.fp16:

            recommendation[
                "notes"
            ].append(
                "BF16 unavailable; use FP16 with gradient scaling."
            )

    # ----------------------------------------------------------
    # ROCm
    # ----------------------------------------------------------

    elif profile.backend == "rocm":

        recommendation[
            "attention"
        ] = "SDPA"

        recommendation[
            "notes"
        ].append(
            "ROCm detected; prefer the ROCm-compatible Unsloth stack."
        )

    # ----------------------------------------------------------
    # XPU
    # ----------------------------------------------------------

    elif profile.backend == "xpu":

        recommendation[
            "attention"
        ] = "SDPA"

        recommendation[
            "gradient_checkpointing"
        ] = True

        recommendation[
            "notes"
        ].append(
            "Intel XPU detected; prefer BF16 when confirmed by PyTorch."
        )

    # ----------------------------------------------------------
    # MPS
    # ----------------------------------------------------------

    elif profile.backend == "mps":

        recommendation[
            "attention"
        ] = "SDPA"

        recommendation[
            "gradient_checkpointing"
        ] = True

        recommendation[
            "notes"
        ].append(
            "MPS training is memory-sensitive; keep batch size conservative."
        )

        recommendation[
            "warnings"
        ].append(
            "FTRAIN will not automatically force Unsloth acceleration on MPS."
        )

    # ----------------------------------------------------------
    # DirectML
    # ----------------------------------------------------------

    elif profile.backend == "dml":

        recommendation[
            "attention"
        ] = "eager"

        recommendation[
            "gradient_checkpointing"
        ] = True

        recommendation[
            "warnings"
        ].append(
            "DirectML operator coverage varies; expect lower training compatibility."
        )

    # ----------------------------------------------------------
    # CPU
    # ----------------------------------------------------------

    elif profile.backend == "cpu":

        recommendation[
            "attention"
        ] = "eager"

        recommendation[
            "gradient_checkpointing"
        ] = False

        recommendation[
            "notes"
        ].append(
            f"{profile.cpu_threads} CPU threads detected."
        )

        recommendation[
            "warnings"
        ].append(
            "CPU training should generally be limited to small models."
        )

    # ----------------------------------------------------------
    # 4-bit request
    # ----------------------------------------------------------

    if load_in_4bit:

        if (
            profile.unsloth
            .quantized_training_supported
        ):

            recommendation[
                "notes"
            ].append(
                "4-bit training path is available."
            )

        else:

            recommendation[
                "warnings"
            ].append(
                "4-bit was requested but a confirmed "
                "quantized Unsloth/bitsandbytes path is unavailable."
            )

    return recommendation


# =============================================================================
# FTRAIN TRAINING PLAN
# =============================================================================

def training_plan(
    *,
    model_parameters: Optional[int] = None,
    profile: Optional[
        HardwareProfile
    ] = None,
    load_in_4bit: bool = False,
    prefer_unsloth: bool = True,
) -> Dict[str, Any]:
    """
    One-call hardware-aware planning API.

    This is the function Core can call before loading a model.
    """

    if profile is None:

        profile = detect()

    plan = recommend_training(
        profile,
        load_in_4bit=load_in_4bit,
        prefer_unsloth=prefer_unsloth,
    )

    # ----------------------------------------------------------
    # Optional model memory estimate
    # ----------------------------------------------------------

    if model_parameters is not None:

        dtype = preferred_dtype(
            profile
        )

        memory = estimate_model_memory(
            model_parameters,
            dtype=dtype,
            training=True,
            lora=(
                load_in_4bit
                or profile.can_use_unsloth
            ),
        )

        plan[
            "estimated_memory"
        ] = memory

        if (
            profile.memory_available_bytes
            and
            memory[
                "estimated_total_gb"
            ]
            >
            profile.memory_available_bytes
            / _GB
            * 0.85
        ):

            plan[
                "warnings"
            ].append(
                "Estimated training memory exceeds conservative "
                "available-memory headroom."
            )

    return plan


# =============================================================================
# HARDWARE REPORT
# =============================================================================

def hardware_report(
    profile: Optional[
        HardwareProfile
    ] = None,
) -> Dict[str, Any]:
    """
    Return a JSON-friendly hardware report.
    """

    if profile is None:

        profile = detect()

    return {
        "backend":
            profile.backend,

        "device_name":
            profile.device_name,

        "device_count":
            profile.device_count,

        "memory_total_gb":
            profile.total_memory_gb,

        "memory_available_gb":
            profile.available_memory_gb,

        "compute_capability":
            profile.compute_capability,

        "bf16":
            profile.bf16,

        "fp16":
            profile.fp16,

        "torch_version":
            profile.torch_version,

        "cuda_version":
            profile.cuda_version,

        "rocm_version":
            profile.rocm_version,

        "os":
            profile.os_name,

        "architecture":
            profile.architecture,

        "cpu_threads":
            profile.cpu_threads,

        "packages": {
            "unsloth":
                profile.unsloth.version,

            "bitsandbytes":
                profile.bitsandbytes,

            "triton":
                profile.triton,

            "flash_attn":
                profile.flash_attn,

            "accelerate":
                profile.accelerate,

            "transformers":
                profile.transformers,

            "trl":
                profile.trl,
        },

        "unsloth": {
            "installed":
                profile.unsloth.installed,

            "importable":
                profile.unsloth.importable,

            "fast_language_model":
                profile.unsloth.fast_language_model,

            "backend_supported":
                profile.unsloth.backend_supported,

            "training_supported":
                profile.unsloth.training_supported,

            "quantized_training_supported":
                profile.unsloth.quantized_training_supported,

            "multi_gpu_supported":
                profile.unsloth.multi_gpu_supported,

            "recommended":
                profile.unsloth.recommended,

            "reason":
                profile.unsloth.reason,

            "warnings":
                list(
                    profile.unsloth.warnings
                ),

            "features":
                list(
                    profile.unsloth.features
                ),
        },

        "gpus": [
            {
                "index":
                    gpu.index,

                "name":
                    gpu.name,

                "backend":
                    gpu.backend,

                "total_memory_gb":
                    gpu.total_memory_gb,

                "free_memory_gb":
                    gpu.free_memory_gb,

                "compute_capability":
                    gpu.compute_capability,

                "capability_label":
                    gpu.capability_label,

                "bf16":
                    gpu.bf16,

                "fp16":
                    gpu.fp16,

                "multi_processor_count":
                    gpu.multi_processor_count,
            }
            for gpu in profile.gpus
        ],

        "notes":
            list(
                profile.notes
            ),

        "warnings":
            list(
                profile.warnings
            ),

        "summary":
            profile.summary(),
    }
