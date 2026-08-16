
import math
import re
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional
from .tensor_stats import compute_tensor_stats
from .similarity import similarity_bundle, aggregate_similarity

CAT_ALPHA = {
    "embedding": 0.85,
    "lm_head": 0.15,
    "norm": 1.00,
    "router": 1.00,
    "shared_expert": 0.60,
    "moe_expert": 0.50,
    "attention": 0.65,
    "ffn": 0.40,
    "other": 0.50
}

def classify_tensor(name: str) -> str:
    n = name.lower()
    if any(k in n for k in ("embed", "wte", "tok_embeddings")):
        return "embedding"
    if any(k in n for k in ("lm_head", "output.weight")):
        return "lm_head"
    if any(k in n for k in ("norm", "ln_f", "layernorm")):
        return "norm"
    if any(k in n for k in ("router", "gate.weight", ".mlp.gate.", "wg")):
        return "router"
    if any(k in n for k in ("shared_experts", "shared_mlp")):
        return "shared_expert"
    if any(k in n for k in ("expert", "mlp.experts")):
        return "moe_expert"
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "wq", "wk", "wv", "wo", "qkv", "kv_a", "kv_b")):
        return "attention"
    if any(k in n for k in ("gate_proj", "up_proj", "down_proj", "w1", "w2", "w3", "mlp")):
        return "ffn"
    return "other"

def extract_layer_depth(name: str, total_layers: int = 32) -> float:
    match = re.search(r'(?:layers|h)\.(\d+)\.', name)
    if match:
        layer_idx = int(match.group(1))
        return min(1.0, max(0.0, layer_idx / max(1, total_layers - 1)))
    return 0.5

@dataclass
class TensorMergePlan:
    name: str
    category: str
    alpha: float
    strategy: str
    projection: str = "identity"
    similarity: float = 0.0
    layer_depth: float = 0.5
    importance_a: float = 0.5
    importance_b: float = 0.5
    reason: str = ""

class MergeAnalyzer:
    def analyze_pair(self, name: str, a: torch.Tensor, b: torch.Tensor, total_layers: int = 32) -> Dict[str, Any]:
        sa = compute_tensor_stats(name, a)
        sb = compute_tensor_stats(name, b)
        sim = 0.0
        if a.shape == b.shape and a.numel() > 0:
            sim = aggregate_similarity(similarity_bundle(a, b))
        layer_depth = extract_layer_depth(name, total_layers)
        category = classify_tensor(name)
        return {
            "a": sa,
            "b": sb,
            "similarity": sim,
            "category": category,
            "layer_depth": layer_depth
        }

class MergePlanner:
    def plan_for_pair(self, name: str, an: Dict[str, Any]) -> TensorMergePlan:
        cat = an["category"]
        sim = an["similarity"]
        depth = an["layer_depth"]
        sa, sb = an["a"], an["b"]
        base_alpha = CAT_ALPHA.get(cat, 0.50)

        if sa.dead and not sb.dead:
            return TensorMergePlan(name, cat, 0.0, "keep_b", "identity", sim, depth, 0.0, 1.0, "Model A tensor dead")
        if sb.dead and not sa.dead:
            return TensorMergePlan(name, cat, 1.0, "keep_a", "identity", sim, depth, 1.0, 0.0, "Model B tensor dead")

        if cat in ("router", "norm"):
            return TensorMergePlan(name, cat, 1.0, "keep_a", "identity", sim, depth, 1.0, 0.0, "Preserving critical structure")

        imp_a = max(sa.l2_norm * (0.5 + sa.effective_rank) * (1.0 + sa.entropy), 1e-6)
        imp_b = max(sb.l2_norm * (0.5 + sb.effective_rank) * (1.0 + sb.entropy), 1e-6)
        rel_imp_a = imp_a / (imp_a + imp_b)

        depth_modifier = 0.1 * math.cos(depth * math.pi * 2)
        
        sim_factor = base_alpha if sim > 0.90 else (0.5 * base_alpha + 0.25 if sim > 0.60 else 0.50)
        entropy_delta = sa.entropy - sb.entropy
        
        alpha = 0.40 * sim_factor + 0.35 * rel_imp_a + 0.15 * (0.5 + 0.5 * entropy_delta) + 0.10 * depth_modifier
        alpha = max(0.05, min(0.95, alpha))

        if sb.std > sa.std * 1.5 and cat in ("attention", "ffn"):
            alpha = max(0.10, alpha - 0.15)

        if sim < 0.20:
            return TensorMergePlan(name, cat, 1.0, "keep_a", "identity", sim, depth, rel_imp_a, 1.0 - rel_imp_a, "Orthogonal feature space")
        if sim > 0.85:
            return TensorMergePlan(name, cat, alpha, "weighted", "identity", sim, depth, rel_imp_a, 1.0 - rel_imp_a, "High alignment weighted blend")
        if sim > 0.55:
            return TensorMergePlan(name, cat, alpha, "slerp", "identity", sim, depth, rel_imp_a, 1.0 - rel_imp_a, "Spherical interpolation")
        if sa.shape == sb.shape and sa.numel >= 16:
            return TensorMergePlan(name, cat, alpha, "ties", "identity", sim, depth, rel_imp_a, 1.0 - rel_imp_a, "TIES sign-resolution merge")

        return TensorMergePlan(name, cat, alpha, "projection", "procrustes", sim, depth, rel_imp_a, 1.0 - rel_imp_a, "Procrustes manifold alignment")
