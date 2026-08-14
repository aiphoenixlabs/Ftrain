import logging
from typing import Dict

logger = logging.getLogger(__name__)

def flash_mode(enabled: bool = True, tf32: bool = True) -> Dict[str, bool]:
    status = {"cudnn_benchmark": False, "tf32": False, "matmul_precision": False}
    if not enabled:
        return status
    try:
        import torch
        if not torch.cuda.is_available():
            return status
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
    except:
        pass
    return status
