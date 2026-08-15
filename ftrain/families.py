from dataclasses import dataclass
from typing import List, Optional, Dict, Any

@dataclass
class ModelPreset:
    lora_targets: List[str]
    learning_rate: float
    lora_r: int
    attn_implementation: Optional[str]

_BASE = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

FAMILY_PRESETS = {
    "qwen": ModelPreset(_BASE, 2e-4, 16, "flash_attention_2"),
    "llama": ModelPreset(_BASE, 2e-4, 16, "flash_attention_2"),
    "deepseek": ModelPreset(_BASE + ["q_a_proj", "q_b_proj", "kv_a_proj_with_mqa"], 1.5e-4, 16, "flash_attention_2"),
    "generic": ModelPreset(_BASE, 2e-4, 16, None)
}

def get_preset(f: str) -> Dict[str, Any]:
    p = FAMILY_PRESETS.get(f.lower(), FAMILY_PRESETS["generic"])
    return {
        "lora_targets": p.lora_targets,
        "learning_rate": p.learning_rate,
        "lora_r": p.lora_r,
        "attn_implementation": p.attn_implementation
    }
