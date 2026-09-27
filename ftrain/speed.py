# ============================================================
# FTRAIN - SPEED / PERFORMANCE ENGINE
# ============================================================
#
# Central runtime optimization utilities for:
#   - CUDA
#   - ROCm
#   - Intel XPU
#   - Apple MPS
#   - CPU
#
# Designed for:
#   - LLM training
#   - LoRA / DoRA
#   - inference
#   - evaluation
#   - Gemma / Qwen / DeepSeek class models
#   - FTRAIN custom trainers
#
# Important:
# This module is intentionally defensive. PyTorch changes APIs
# frequently, so unsupported features are skipped rather than
# crashing the entire trainer.
# ============================================================

import gc
import logging
import os
import time

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SpeedConfig:
    """
    Global FTRAIN performance configuration.
    """

    enabled: bool = True

    # --------------------------------------------------------
    # Math / precision
    # --------------------------------------------------------

    tf32: bool = True
    matmul_precision: str = "high"

    # --------------------------------------------------------
    # CUDA / cuDNN
    # --------------------------------------------------------

    cudnn_benchmark: bool = True

    # --------------------------------------------------------
    # SDPA / attention
    # --------------------------------------------------------

    enable_flash_attention: bool = True
    enable_mem_efficient_attention: bool = True
    enable_math_attention: bool = True

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    empty_cache: bool = True
    garbage_collect: bool = True

    # --------------------------------------------------------
    # Model layout
    # --------------------------------------------------------

    channels_last: bool = False

    # --------------------------------------------------------
    # Compilation
    # --------------------------------------------------------

    compile_model: bool = False
    compile_mode: str = "default"
    compile_fullgraph: bool = False
    compile_dynamic: bool = False

    # --------------------------------------------------------
    # CUDA allocator
    # --------------------------------------------------------

    expandable_segments: bool = True

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    cpu_threads: Optional[int] = None
    cpu_interop_threads: Optional[int] = None

    # --------------------------------------------------------
    # Profiling / warmup
    # --------------------------------------------------------

    synchronize_before_timing: bool = True


# ============================================================
# BACKEND DETECTION
# ============================================================

def detect_backend() -> str:
    """
    Return the active PyTorch compute backend.

    Possible values:

        cuda
        rocm
        xpu
        mps
        cpu
        unknown
    """

    try:
        import torch

        # ----------------------------------------------------
        # CUDA / ROCm
        # ----------------------------------------------------

        if torch.cuda.is_available():

            if getattr(
                torch.version,
                "hip",
                None,
            ):
                return "rocm"

            return "cuda"

        # ----------------------------------------------------
        # Intel XPU
        # ----------------------------------------------------

        if (
            hasattr(torch, "xpu")
            and torch.xpu.is_available()
        ):
            return "xpu"

        # ----------------------------------------------------
        # Apple Metal
        # ----------------------------------------------------

        mps = getattr(
            torch.backends,
            "mps",
            None,
        )

        if (
            mps is not None
            and mps.is_available()
        ):
            return "mps"

        return "cpu"

    except Exception:
        return "unknown"


# ============================================================
# DEVICE CAPABILITY
# ============================================================

def get_device_info() -> Dict[str, Any]:
    """
    Collect useful hardware information without requiring CUDA.
    """

    info: Dict[str, Any] = {
        "backend": detect_backend(),
        "device_count": 0,
        "devices": [],
        "torch_version": None,
        "cuda_version": None,
        "rocm_version": None,
    }

    try:
        import torch

        info["torch_version"] = torch.__version__

        info["cuda_version"] = getattr(
            torch.version,
            "cuda",
            None,
        )

        info["rocm_version"] = getattr(
            torch.version,
            "hip",
            None,
        )

        # ----------------------------------------------------
        # CUDA / ROCm
        # ----------------------------------------------------

        if torch.cuda.is_available():

            count = torch.cuda.device_count()

            info["device_count"] = count

            for index in range(count):

                try:

                    props = (
                        torch.cuda.get_device_properties(
                            index
                        )
                    )

                    info["devices"].append(
                        {
                            "index": index,
                            "name": props.name,
                            "major": props.major,
                            "minor": props.minor,
                            "total_memory_bytes":
                                int(
                                    props.total_memory
                                ),
                            "total_memory_gb":
                                round(
                                    props.total_memory
                                    / 1024**3,
                                    2,
                                ),
                            "multi_processor_count":
                                int(
                                    props.multi_processor_count
                                ),
                        }
                    )

                except Exception as exc:

                    info["devices"].append(
                        {
                            "index": index,
                            "error": str(exc),
                        }
                    )

        # ----------------------------------------------------
        # XPU
        # ----------------------------------------------------

        elif (
            hasattr(torch, "xpu")
            and torch.xpu.is_available()
        ):

            count = torch.xpu.device_count()

            info["device_count"] = count

            for index in range(count):

                info["devices"].append(
                    {
                        "index": index,
                        "name": torch.xpu.get_device_name(
                            index
                        ),
                    }
                )

        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        else:

            info["devices"].append(
                {
                    "index": 0,
                    "name": "CPU",
                    "logical_cpu_count":
                        os.cpu_count(),
                }
            )

            info["device_count"] = 1

    except Exception as exc:

        info["error"] = str(exc)

    return info


# ============================================================
# CUDA MEMORY
# ============================================================

def cuda_memory_stats(
    device: Optional[Any] = None,
) -> Dict[str, float]:
    """
    Return current CUDA memory information in GB.
    """

    result = {
        "allocated_gb": 0.0,
        "reserved_gb": 0.0,
        "free_gb": 0.0,
        "total_gb": 0.0,
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return result

        if device is None:
            device = torch.cuda.current_device()

        allocated = torch.cuda.memory_allocated(
            device
        )

        reserved = torch.cuda.memory_reserved(
            device
        )

        free, total = torch.cuda.mem_get_info(
            device
        )

        result["allocated_gb"] = (
            allocated / 1024**3
        )

        result["reserved_gb"] = (
            reserved / 1024**3
        )

        result["free_gb"] = (
            free / 1024**3
        )

        result["total_gb"] = (
            total / 1024**3
        )

    except Exception as exc:

        logger.debug(
            "CUDA memory stats failed: %s",
            exc,
        )

    return result


# ============================================================
# CACHE / MEMORY CLEANUP
# ============================================================

def clear_memory(
    cuda: bool = True,
    collect_gc: bool = True,
) -> Dict[str, bool]:
    """
    Aggressively clean Python/CUDA caches.

    Safe to call between:
        experiments
        benchmarks
        checkpoints
        merge stages
    """

    status = {
        "gc": False,
        "cuda_cache": False,
    }

    # --------------------------------------------------------
    # Python garbage collection
    # --------------------------------------------------------

    if collect_gc:

        try:

            gc.collect()

            status["gc"] = True

        except Exception:
            pass

    # --------------------------------------------------------
    # CUDA allocator
    # --------------------------------------------------------

    if cuda:

        try:

            import torch

            if torch.cuda.is_available():

                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

                status["cuda_cache"] = True

        except Exception:
            pass

    return status


# ============================================================
# TF32 / MATMUL
# ============================================================

def configure_matmul(
    tf32: bool = True,
    precision: str = "high",
) -> Dict[str, bool]:
    """
    Configure float32 matrix multiplication.

    TF32 is useful on NVIDIA Ampere+ and newer GPUs.
    """

    status = {
        "tf32": False,
        "matmul_precision": False,
    }

    try:

        import torch

        # ----------------------------------------------------
        # Generic PyTorch matmul precision
        # ----------------------------------------------------

        if hasattr(
            torch,
            "set_float32_matmul_precision",
        ):

            try:

                torch.set_float32_matmul_precision(
                    precision
                )

                status[
                    "matmul_precision"
                ] = True

            except Exception:
                pass

        # ----------------------------------------------------
        # CUDA-specific TF32
        # ----------------------------------------------------

        if (
            tf32
            and torch.cuda.is_available()
        ):

            try:

                major, _ = (
                    torch.cuda.get_device_capability()
                )

                if major >= 8:

                    if hasattr(
                        torch.backends.cuda.matmul,
                        "allow_tf32",
                    ):

                        torch.backends.cuda.matmul.allow_tf32 = True

                        status["tf32"] = True

                    if hasattr(
                        torch.backends.cudnn,
                        "allow_tf32",
                    ):

                        torch.backends.cudnn.allow_tf32 = True

            except Exception:
                pass

    except Exception as exc:

        logger.debug(
            "Matmul configuration failed: %s",
            exc,
        )

    return status


# ============================================================
# ATTENTION BACKENDS
# ============================================================

def configure_attention(
    flash: bool = True,
    mem_efficient: bool = True,
    math: bool = True,
) -> Dict[str, bool]:
    """
    Configure PyTorch SDPA/attention backends when the installed
    PyTorch version exposes the necessary APIs.

    This does NOT install FlashAttention.
    It only enables compatible PyTorch attention kernels.
    """

    status = {
        "flash": False,
        "mem_efficient": False,
        "math": False,
    }

    try:

        import torch

        # ----------------------------------------------------
        # Legacy CUDA SDPA backend flags
        # ----------------------------------------------------

        backends = getattr(
            torch.backends,
            "cuda",
            None,
        )

        if backends is not None:

            # flash_sdp
            try:

                if hasattr(
                    backends,
                    "enable_flash_sdp",
                ):

                    backends.enable_flash_sdp(
                        flash
                    )

                    status["flash"] = bool(
                        flash
                    )

            except Exception:
                pass

            # memory efficient
            try:

                if hasattr(
                    backends,
                    "enable_mem_efficient_sdp",
                ):

                    backends.enable_mem_efficient_sdp(
                        mem_efficient
                    )

                    status[
                        "mem_efficient"
                    ] = bool(
                        mem_efficient
                    )

            except Exception:
                pass

            # math
            try:

                if hasattr(
                    backends,
                    "enable_math_sdp",
                ):

                    backends.enable_math_sdp(
                        math
                    )

                    status["math"] = bool(
                        math
                    )

            except Exception:
                pass

    except Exception as exc:

        logger.debug(
            "Attention configuration failed: %s",
            exc,
        )

    return status


# ============================================================
# CU DNN
# ============================================================

def configure_cudnn(
    benchmark: bool = True,
) -> bool:

    try:

        import torch

        if torch.cuda.is_available():

            torch.backends.cudnn.benchmark = (
                benchmark
            )

            return True

    except Exception:
        pass

    return False


# ============================================================
# CPU THREAD CONFIGURATION
# ============================================================

def configure_cpu_threads(
    threads: Optional[int] = None,
    interop_threads: Optional[int] = None,
) -> Dict[str, bool]:

    status = {
        "threads": False,
        "interop_threads": False,
    }

    try:

        import torch

        if threads is not None:

            threads = max(
                1,
                int(threads),
            )

            torch.set_num_threads(
                threads
            )

            status["threads"] = True

        if interop_threads is not None:

            interop_threads = max(
                1,
                int(interop_threads),
            )

            # This can only safely be changed before
            # parallel work starts.
            try:

                torch.set_num_interop_threads(
                    interop_threads
                )

                status[
                    "interop_threads"
                ] = True

            except RuntimeError:
                # PyTorch may reject changes after
                # parallel work has already started.
                pass

    except Exception as exc:

        logger.debug(
            "CPU thread configuration failed: %s",
            exc,
        )

    return status


# ============================================================
# CUDA ALLOCATOR CONFIGURATION
# ============================================================

def configure_allocator(
    expandable_segments: bool = True,
) -> bool:
    """
    Configure PYTORCH_CUDA_ALLOC_CONF when possible.

    Must ideally be applied before the first CUDA allocation.
    """

    try:

        if not expandable_segments:
            return False

        current = os.environ.get(
            "PYTORCH_CUDA_ALLOC_CONF",
            "",
        )

        if (
            "expandable_segments" not in current
        ):

            if current:

                current += ","

            current += (
                "expandable_segments:True"
            )

            os.environ[
                "PYTORCH_CUDA_ALLOC_CONF"
            ] = current

        return True

    except Exception:
        return False


# ============================================================
# MAIN SPEED MODE
# ============================================================

def flash_mode(
    enabled: bool = True,
    tf32: bool = True,
    *,
    matmul_precision: str = "high",
    cudnn_benchmark: bool = True,
    attention: bool = True,
    compile_model: bool = False,
    cpu_threads: Optional[int] = None,
    cpu_interop_threads: Optional[int] = None,
) -> Dict[str, Any]:
    """
    FTRAIN global speed configuration.

    Backward compatible with:

        flash_mode()

    or:

        flash_mode(
            enabled=True,
            tf32=True,
        )

    Returns a detailed runtime report.
    """

    status: Dict[str, Any] = {
        "enabled": bool(enabled),
        "backend": "unknown",

        "cudnn_benchmark": False,
        "tf32": False,
        "matmul_precision": False,

        "attention_flash": False,
        "attention_mem_efficient": False,
        "attention_math": False,

        "cpu_threads": False,
        "cpu_interop_threads": False,

        "allocator": False,
        "compile_requested": bool(
            compile_model
        ),

        "device_count": 0,
    }

    if not enabled:

        status["backend"] = detect_backend()

        return status

    try:

        import torch

        backend = detect_backend()

        status["backend"] = backend

        # ----------------------------------------------------
        # Device count
        # ----------------------------------------------------

        if backend in (
            "cuda",
            "rocm",
        ):

            status["device_count"] = (
                torch.cuda.device_count()
            )

        elif backend == "xpu":

            status["device_count"] = (
                torch.xpu.device_count()
            )

        else:

            status["device_count"] = 1

        # ----------------------------------------------------
        # Matmul / TF32
        # ----------------------------------------------------

        matmul_status = configure_matmul(
            tf32=tf32,
            precision=matmul_precision,
        )

        status.update(
            matmul_status
        )

        # ----------------------------------------------------
        # cuDNN
        # ----------------------------------------------------

        if cudnn_benchmark:

            status[
                "cudnn_benchmark"
            ] = configure_cudnn(
                True
            )

        # ----------------------------------------------------
        # Attention
        # ----------------------------------------------------

        if attention:

            attention_status = (
                configure_attention(
                    flash=True,
                    mem_efficient=True,
                    math=True,
                )
            )

            status[
                "attention_flash"
            ] = attention_status[
                "flash"
            ]

            status[
                "attention_mem_efficient"
            ] = attention_status[
                "mem_efficient"
            ]

            status[
                "attention_math"
            ] = attention_status[
                "math"
            ]

        # ----------------------------------------------------
        # CPU
        # ----------------------------------------------------

        if backend == "cpu":

            cpu_status = configure_cpu_threads(
                cpu_threads,
                cpu_interop_threads,
            )

            status.update(
                {
                    "cpu_threads":
                        cpu_status["threads"],

                    "cpu_interop_threads":
                        cpu_status[
                            "interop_threads"
                        ],
                }
            )

        # ----------------------------------------------------
        # CUDA allocator
        # ----------------------------------------------------

        if backend in (
            "cuda",
            "rocm",
        ):

            status[
                "allocator"
            ] = configure_allocator(
                True
            )

        # ----------------------------------------------------
        # Optional compile note
        # ----------------------------------------------------

        if compile_model:

            status[
                "compile_available"
            ] = hasattr(
                torch,
                "compile",
            )

        else:

            status[
                "compile_available"
            ] = False

    except Exception as exc:

        status["error"] = str(exc)

        logger.exception(
            "flash_mode configuration error"
        )

    return status


# ============================================================
# MODEL MEMORY FORMAT
# ============================================================

def optimize_model_layout(
    model: Any,
    *,
    channels_last: bool = False,
) -> Any:
    """
    Optional model-layout optimization.

    Channels-last can help convolution-heavy networks,
    but it is usually irrelevant for Transformer LLMs.

    Therefore the default is False.
    """

    if model is None:
        return model

    if not channels_last:
        return model

    try:

        import torch

        # Only convert modules that support 4D layout.
        for module in model.modules():

            if isinstance(
                module,
                (
                    torch.nn.Conv1d,
                    torch.nn.Conv2d,
                ),
            ):

                module.to(
                    memory_format=torch.channels_last
                )

    except Exception as exc:

        logger.debug(
            "Model layout optimization skipped: %s",
            exc,
        )

    return model


# ============================================================
# TORCH COMPILE
# ============================================================

def compile_model(
    model: Any,
    *,
    mode: str = "default",
    fullgraph: bool = False,
    dynamic: bool = False,
) -> Any:
    """
    Safely compile a model with torch.compile.

    Returns the original model if compilation is unavailable
    or fails.

    NOTE:
    For LLM training, compile=True should be treated as an
    experiment-specific optimization, not a universal default.
    """

    if model is None:
        return model

    try:

        import torch

        compiler = getattr(
            torch,
            "compile",
            None,
        )

        if compiler is None:

            logger.warning(
                "torch.compile is unavailable."
            )

            return model

        compiled = compiler(
            model,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=dynamic,
        )

        logger.info(
            "torch.compile enabled: mode=%s",
            mode,
        )

        return compiled

    except Exception as exc:

        logger.warning(
            "torch.compile failed; "
            "using original model: %s",
            exc,
        )

        return model


# ============================================================
# AUTOCast
# ============================================================

def autocast_context(
    device_type: Optional[str] = None,
    dtype: Optional[Any] = None,
    enabled: bool = True,
):
    """
    Return a PyTorch autocast context.

    Example:

        with autocast_context(
            "cuda",
            torch.bfloat16,
        ):
            output = model(**batch)
    """

    import torch

    if device_type is None:

        backend = detect_backend()

        if backend in (
            "cuda",
            "rocm",
        ):

            device_type = "cuda"

        elif backend == "xpu":

            device_type = "xpu"

        elif backend == "mps":

            device_type = "mps"

        else:

            device_type = "cpu"

    # New torch.autocast API.
    try:

        return torch.autocast(
            device_type=device_type,
            dtype=dtype,
            enabled=enabled,
        )

    except Exception:

        # Extremely defensive fallback.
        from contextlib import nullcontext

        return nullcontext()


# ============================================================
# SYNCHRONIZATION
# ============================================================

def synchronize(
    device: Optional[Any] = None,
) -> bool:
    """
    Synchronize supported accelerator backends.
    """

    try:

        import torch

        if torch.cuda.is_available():

            torch.cuda.synchronize(
                device=device
            )

            return True

        if (
            hasattr(torch, "xpu")
            and torch.xpu.is_available()
        ):

            torch.xpu.synchronize()

            return True

        # MPS currently does not expose a universally
        # equivalent synchronize() API across versions.
        return False

    except Exception:

        return False


# ============================================================
# GPU TIMING
# ============================================================

def benchmark_callable(
    fn,
    *,
    warmup: int = 2,
    iterations: int = 10,
    synchronize_before_timing: bool = True,
) -> Dict[str, float]:
    """
    Benchmark a callable with accelerator synchronization.

    Useful for comparing:

        eager vs compile
        SDPA modes
        model versions
        merge implementations
    """

    warmup = max(
        0,
        int(warmup),
    )

    iterations = max(
        1,
        int(iterations),
    )

    # --------------------------------------------------------
    # Warmup
    # --------------------------------------------------------

    for _ in range(warmup):

        fn()

    if synchronize_before_timing:

        synchronize()

    # --------------------------------------------------------
    # Timed region
    # --------------------------------------------------------

    start = time.perf_counter()

    for _ in range(iterations):

        fn()

    if synchronize_before_timing:

        synchronize()

    elapsed = (
        time.perf_counter()
        - start
    )

    avg_ms = (
        elapsed
        * 1000.0
        / iterations
    )

    return {
        "total_seconds": elapsed,
        "average_ms": avg_ms,
        "iterations": float(iterations),
    }


# ============================================================
# TOKENS / SECOND BENCHMARK
# ============================================================

def benchmark_tokens(
    fn,
    tokens: int,
    *,
    warmup: int = 2,
    iterations: int = 10,
) -> Dict[str, float]:
    """
    Benchmark a callable that processes a known number of tokens.

    Returns tokens/sec and average latency.
    """

    result = benchmark_callable(
        fn,
        warmup=warmup,
        iterations=iterations,
    )

    avg_seconds = (
        result["average_ms"]
        / 1000.0
    )

    tokens_per_second = (
        tokens
        / max(
            avg_seconds,
            1e-12,
        )
    )

    result[
        "tokens_per_second"
    ] = tokens_per_second

    return result


# ============================================================
# MEMORY-AWARE BATCH SUGGESTION
# ============================================================

def suggest_batch_scaling(
    *,
    allocated_gb: float,
    total_gb: float,
    safety_fraction: float = 0.90,
    current_batch: int = 1,
) -> int:
    """
    Crude memory-aware batch-size suggestion.

    This is deliberately conservative.

    It should not replace a real OOM-aware tuner.
    """

    current_batch = max(
        1,
        int(current_batch),
    )

    if total_gb <= 0:

        return current_batch

    usable = (
        total_gb
        * max(
            0.1,
            min(
                safety_fraction,
                0.99,
            ),
        )
    )

    if allocated_gb <= 0:

        return current_batch

    scale = (
        usable
        / allocated_gb
    )

    proposed = int(
        current_batch
        * scale
    )

    return max(
        1,
        proposed,
    )


# ============================================================
# PERFORMANCE SNAPSHOT
# ============================================================

def performance_snapshot() -> Dict[str, Any]:
    """
    Collect a single comprehensive runtime snapshot.
    """

    snapshot = {
        "timestamp": time.time(),
        "device": get_device_info(),
        "memory": {},
    }

    backend = snapshot[
        "device"
    ].get(
        "backend"
    )

    if backend in (
        "cuda",
        "rocm",
    ):

        snapshot[
            "memory"
        ] = cuda_memory_stats()

    return snapshot


# ============================================================
# DEFAULT FTRAIN CONFIG
# ============================================================

def default_speed_config(
    backend: Optional[str] = None,
) -> SpeedConfig:
    """
    Create sensible defaults based on hardware.
    """

    if backend is None:
        backend = detect_backend()

    cfg = SpeedConfig()

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    if backend == "cpu":

        cfg.tf32 = False
        cfg.cudnn_benchmark = False

        cfg.enable_flash_attention = False
        cfg.enable_mem_efficient_attention = False

        cfg.compile_model = False

    # --------------------------------------------------------
    # MPS
    # --------------------------------------------------------

    elif backend == "mps":

        cfg.tf32 = False
        cfg.cudnn_benchmark = False

        cfg.enable_flash_attention = False
        cfg.enable_mem_efficient_attention = False

    # --------------------------------------------------------
    # XPU
    # --------------------------------------------------------

    elif backend == "xpu":

        cfg.tf32 = False

    # --------------------------------------------------------
    # CUDA / ROCm
    # --------------------------------------------------------

    elif backend in (
        "cuda",
        "rocm",
    ):

        cfg.tf32 = True
        cfg.cudnn_benchmark = True

    return cfg


# ============================================================
# APPLY CONFIG
# ============================================================

def apply_speed_config(
    config: Optional[SpeedConfig] = None,
) -> Dict[str, Any]:
    """
    Apply a complete SpeedConfig.
    """

    cfg = (
        config
        if config is not None
        else default_speed_config()
    )

    status = flash_mode(
        enabled=cfg.enabled,
        tf32=cfg.tf32,
        matmul_precision=cfg.matmul_precision,
        cudnn_benchmark=cfg.cudnn_benchmark,
        attention=(
            cfg.enable_flash_attention
            or cfg.enable_mem_efficient_attention
            or cfg.enable_math_attention
        ),
        compile_model=cfg.compile_model,
        cpu_threads=cfg.cpu_threads,
        cpu_interop_threads=cfg.cpu_interop_threads,
    )

    # --------------------------------------------------------
    # Optional allocator
    # --------------------------------------------------------

    if cfg.expandable_segments:

        status[
            "allocator"
        ] = configure_allocator(
            True
        )

    # --------------------------------------------------------
    # Memory cleanup
    # --------------------------------------------------------

    if cfg.empty_cache:

        clear_memory(
            cuda=True,
            collect_gc=cfg.garbage_collect,
        )

    return status


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

def format_speed_report(
    status: Dict[str, Any],
) -> str:
    """
    Turn speed status into a compact FTRAIN log.
    """

    backend = status.get(
        "backend",
        "unknown",
    )

    lines = [
        "FTRAIN SPEED ENGINE",
        f"Backend: {backend}",
        f"Devices: {status.get('device_count', 0)}",
        f"TF32: {status.get('tf32', False)}",
        (
            "Matmul precision: "
            f"{status.get('matmul_precision', False)}"
        ),
        (
            "cuDNN benchmark: "
            f"{status.get('cudnn_benchmark', False)}"
        ),
        (
            "Flash SDPA: "
            f"{status.get('attention_flash', False)}"
        ),
        (
            "Memory-efficient SDPA: "
            f"{status.get('attention_mem_efficient', False)}"
        ),
        (
            "Math SDPA: "
            f"{status.get('attention_math', False)}"
        ),
        (
            "Allocator: "
            f"{status.get('allocator', False)}"
        ),
    ]

    return "\n".join(lines)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Configuration
    "SpeedConfig",

    # Detection
    "detect_backend",
    "get_device_info",

    # Main optimization
    "flash_mode",
    "apply_speed_config",
    "default_speed_config",

    # Backend controls
    "configure_matmul",
    "configure_attention",
    "configure_cudnn",
    "configure_cpu_threads",
    "configure_allocator",

    # Memory
    "clear_memory",
    "cuda_memory_stats",

    # Model
    "optimize_model_layout",
    "compile_model",

    # Precision
    "autocast_context",

    # Synchronization
    "synchronize",

    # Benchmarking
    "benchmark_callable",
    "benchmark_tokens",

    # Runtime
    "performance_snapshot",
    "suggest_batch_scaling",

    # Reporting
    "format_speed_report",
                ]
