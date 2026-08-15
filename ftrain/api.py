"""
🔥 FTRAIN High-Level API (v8.0.6) 🔥

Main entry points: train, merge, test
"""

from .config import TrainConfig, MergeConfig
from .data_utils import load_data
from .core import Ftrain
from .merger import Merger
from . import rewards
from typing import Optional, List, Callable, Dict, Any

class train:
    @staticmethod
    def fire(Model: str, Data: Any, Steps: int = 100, Captain: Optional[str] = None, Answer: str = "auto_yes", **kwargs) -> Any:
        """
        Fires the FTRAIN training pipeline.
        Allows users to dynamically pass 'output_dir' via kwargs.
        """
        data = load_data(Data)
        split = int(0.9 * len(data))
        train_data = data[:split]
        val_data = data[split:] if split < len(data) else None
        
        # Extract output_dir from kwargs if provided, else use default
        out_dir = kwargs.pop("output_dir", "./ftrain_output")
        
        cfg = TrainConfig(
            model_name=Model, 
            captain_model=Captain, 
            max_steps=Steps, 
            answer_mode=Answer, 
            captain_mode="llm" if Captain else "rule", 
            output_dir=out_dir, 
            **kwargs
        )
        return Ftrain(cfg, train_data, val_data).train()

class merge:
    @staticmethod
    def fire(First: Optional[str] = None, Second: Optional[str] = None, Captain: Optional[str] = None, **kwargs) -> bool:
        """
        Fires the FTRAIN intelligent merging pipeline.
        Allows users to dynamically pass 'output_dir' via kwargs.
        """
        model_a = First if First else kwargs.pop("Model_a", None)
        model_b = Second if Second else kwargs.pop("Model_b", None)
        
        if not model_a or not model_b:
            raise ValueError("You must provide 'First' and 'Second' models!")
            
        # Extract output_dir from kwargs if provided, else use default
        out_dir = kwargs.pop("output_dir", "./merged_model")
        
        cfg = MergeConfig(
            model_a=model_a, 
            model_b=model_b, 
            captain_model=Captain, 
            output_dir=out_dir, 
            **kwargs
        )
        return Merger(cfg).merge()

def test():
    print("✅ FTRAIN v8.0.6 imported successfully")

# Export rewards so users can easily pass them to the API
xml_format_reward = rewards.xml_format_reward
math_exact_reward = rewards.math_exact_reward
python_exec_reward = rewards.python_exec_reward
