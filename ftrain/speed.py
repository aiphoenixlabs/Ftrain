import logging
from typing import Dict

logger = logging.getLogger(__name__)


def flash_mode(enabled: bool = True, tf32: bool = True) -> Dict[str, bool]:
    """Backend-aware fast-math setup.

    NVIDIA CUDA: cudnn benchmark + TF32 (Ampere+).
    AMD ROCm / Intel XPU / Apple MPS: no global math switches exist; the
    call is a harmless no-op that reports what was applied.
    CPU: pinning thread count is left to the host process.
    """
    status = {
        "cudnn_benchmark": False,
        "tf32": False,
        "matmul_precision": False,
        "backend": "cpu",
    }
    if not enabled:
        return status
    try:
        import torch

        if torch.cuda.is_available():
            status["backend"] = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            torch.backends.cudnn.benchmark = True
            status["cudnn_benchmark"] = True
            if tf32:
                major, _ = torch.cuda.get_device_capability()
                if major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    status["tf32"] = True
                torch.set_float32_matmul_precision("high")
                status["matmul_precision"] = True
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            status["backend"] = "xpu"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            status["backend"] = "mps"
    except Exception:
        pass
    return status
