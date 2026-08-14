import os, json, logging
from typing import Union, List, Dict, Any

logger = logging.getLogger(__name__)

def load_data(src: Union[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(src, list):
        return src
    if isinstance(src, str):
        if src.startswith("hf://"):
            try:
                from datasets import load_dataset
                return list(load_dataset(src[5:], split="train"))
            except Exception as e:
                raise RuntimeError(f"Failed to load HF dataset '{src}': {e}")
        if os.path.exists(src):
            if src.endswith(".json"):
                return json.load(open(src, "r", encoding="utf-8"))
            elif src.endswith(".jsonl"):
                return [json.loads(l) for l in open(src, "r", encoding="utf-8") if l.strip()]
        raise FileNotFoundError(f"Data source not found: {src}")
