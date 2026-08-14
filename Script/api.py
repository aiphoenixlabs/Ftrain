from .config import TrainConfig, MergeConfig
from .data_utils import load_data
from .core import Ftrain
from .merger import Merger
from . import rewards
from typing import Optional, Any

class train:
    @staticmethod
    def fire(Model: str, Data: Any, Steps: int = 100, Captain: Optional[str] = None, Answer: str = "auto_yes", **kwargs) -> Any:
        data = load_data(Data)
        sp = int(0.9 * len(data))
        td = data[:sp]
        vd = data[sp:] if sp < len(data) else None
        cfg = TrainConfig(model_name=Model, captain_model=Captain, max_steps=Steps, answer_mode=Answer, captain_mode="llm" if Captain else "rule", output_dir="./ftrain_output", **kwargs)
        return Ftrain(cfg, td, vd).train()

class merge:
    @staticmethod
    def fire(First: Optional[str] = None, Second: Optional[str] = None, Captain: Optional[str] = None, **kwargs) -> bool:
        ma = First if First else kwargs.pop("Model_a", None)
        mb = Second if Second else kwargs.pop("Model_b", None)
        if not ma or not mb:
            raise ValueError("You must provide 'First' and 'Second' models!")
        cfg = MergeConfig(model_a=ma, model_b=mb, captain_model=Captain, output_dir="./merged_model", **kwargs)
        return Merger(cfg).merge()

def test():
    print("✅ FTRAIN v1.0.0 imported successfully")

xml_format_reward = rewards.xml_format_reward
math_exact_reward = rewards.math_exact_reward
python_exec_reward = rewards.python_exec_reward
