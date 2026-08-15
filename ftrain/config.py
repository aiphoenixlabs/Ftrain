from dataclasses import dataclass, field, asdict
from typing import List, Optional, Callable, Literal, Dict, Any
import json, logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class TrainConfig:
    model_name: str
    captain_model: Optional[str] = None
    output_dir: str = "./ftrain_output"
    logging_dir: str = "./ftrain_logs"
    resume_from_checkpoint: Optional[str] = None
    save_total_limit: int = 3
    family: str = "auto"
    seed: int = 42
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    use_custom_lora: bool = False
    use_dora: bool = False
    auto_lora_targets: bool = False
    lora_target_count: int = 4
    lora_target_modules: Optional[List[str]] = None
    use_unsloth_lora: bool = True
    lora_a_lr_mult: float = 2.0
    lora_b_lr_mult: float = 1.0
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    warmup_steps: int = 0
    min_lr_ratio: float = 0.1
    layerwise_lr_decay: float = 0.85
    swiglu_gate_boost: float = 1.2
    moe_router_lr_multiplier: float = 0.5
    use_lr_finder: bool = False
    lr_finder_start_lr: float = 1e-7
    lr_finder_end_lr: float = 10.0
    lr_finder_iter: int = 100
    use_cosine_restarts: bool = False
    restart_interval: int = 50
    max_grad_norm: float = 1.0
    max_steps: int = 100
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    eval_interval: int = 10
    checkpoint_interval: int = 20
    dataloader_num_workers: int = 2
    pin_memory: bool = True
    gradient_checkpointing_enable: bool = False
    use_adaptive_accumulation: bool = False
    target_batch_tokens: int = 8192
    use_hf_trainer: bool = True
    use_unsloth_trainer: bool = True
    captain_enabled: bool = True
    captain_mode: Literal["rule", "adaptive", "strict", "llm"] = "rule"
    captain_interval: int = 5
    captain_min_interval: float = 2.0
    captain_clamp: List[float] = field(default_factory=lambda: [0.25, 2.5])
    captain_velocity_window: int = 3
    data_sources: Optional[List[str]] = None
    data_balance_strategy: Literal["tokens", "examples", "equal"] = "tokens"
    use_packing: bool = False
    train_on_response_only: bool = False
    mask_thinking: bool = False
    group_by_length: bool = True
    data_perplexity_filter: bool = False
    data_perplexity_keep_pct: float = 0.8
    data_dedup: bool = False
    data_dedup_threshold: float = 0.9
    answer_mode: Literal["auto_yes", "interactive", "strict"] = "auto_yes"
    report_to: Literal["none", "wandb", "tensorboard"] = "none"
    auto_resume: bool = False
    use_dashboard: bool = False
    dashboard_port: int = 7860
    show_model_progress: bool = True
    use_grpo: bool = False
    grpo_num_generations: int = 6
    grpo_reward_funcs: Optional[List[Callable]] = None

    def __post_init__(self):
        if self.lora_r <= 0:
            raise ValueError("lora_r must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

    def to_dict(self):
        d = asdict(self)
        d.pop("grpo_reward_funcs", None)
        return d

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=4)


@dataclass
class MergeConfig:
    model_a: str
    model_b: str
    captain_model: Optional[str] = None
    output_dir: str = "./merged_model"
    name: str = "auto"
    save_dtype: Literal["fp16", "bf16", "fp32"] = "bf16"
    strategy: Literal["intelligent", "linear", "slerp", "ties", "dare"] = "intelligent"
    alpha: float = 0.5
    use_fisher: bool = True
    use_dare: bool = False
    use_task_arithmetic: bool = False
    knowledge_preservation: bool = False
    merge_rollback: bool = False
    system_prompt_merger: str = "You are an expert model merger."
    calibration_data: Optional[Any] = None
    repair_steps: int = 0
    merge_knowledge_distill: bool = False
    force_cuda_merge: bool = False
    hugging: bool = False
    hugging_token: Optional[str] = None
    align_grpo: bool = False
    align_grpo_steps: int = 50
    align_grpo_reward_funcs: Optional[List[Callable]] = None

    def __post_init__(self):
        if self.model_a == self.model_b:
            raise ValueError("model_a and model_b cannot be the same.")

    def to_dict(self):
        d = asdict(self)
        d.pop("align_grpo_reward_funcs", None)
        d.pop("calibration_data", None)
        return d

    def save(self, path):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dict(), f, indent=4)
