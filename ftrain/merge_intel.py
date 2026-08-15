from dataclasses import dataclass
from .tensor_stats import compute_tensor_stats
from .similarity import similarity_bundle, aggregate_similarity

def classify_tensor(name):
    n = name.lower()
    if "embed" in n: return "embedding"
    if "lm_head" in n: return "lm_head"
    if "norm" in n: return "norm"
    if "router" in n or ".mlp.gate." in n: return "router"
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj")): return "attention"
    if any(k in n for k in ("gate_proj", "up_proj", "down_proj")): return "ffn"
    return "other"

CAT_ALPHA = {"embedding": 0.9, "lm_head": 0.0, "norm": 1.0, "router": 1.0, "attention": 0.65, "ffn": 0.40, "other": 0.5}

@dataclass
class TensorMergePlan:
    name: str
    category: str
    alpha: float
    strategy: str
    projection: str = "identity"
    similarity: float = 0.0

class MergeAnalyzer:
    def analyze_pair(self, name, a, b):
        sa, sb = compute_tensor_stats(name, a), compute_tensor_stats(name, b)
        sim = aggregate_similarity(similarity_bundle(a, b)) if a.shape == b.shape else 0.0
        return {"a": sa, "b": sb, "similarity": sim, "category": classify_tensor(name)}

class MergePlanner:
    def plan_for_pair(self, name, an):
        cat, base, sim = an["category"], CAT_ALPHA.get(an["category"], 0.5), an["similarity"]
        sa, sb = an["a"], an["b"]
        na = max(sa.l2_norm * (0.5 + sa.effective_rank), 1e-6)
        nb = max(sb.l2_norm * (0.5 + sb.effective_rank), 1e-6)
        imp = na / (na + nb)
        sa_ = base if sim > 0.9 else (0.5 * base + 0.25 if sim > 0.6 else 0.5)
        a = max(0.05, min(0.95, 0.5 * sa_ + 0.3 * imp + 0.2 * (0.5 + 0.5 * (sa.entropy - sb.entropy))))
        if sb.std > sa.std * 1.5 and cat in ("attention", "ffn"):
            a = max(0.1, a - 0.2)
        if sim < 0.2 and cat not in ("norm", "router"):
            return TensorMergePlan(name, cat, 1.0, "keep_a", "identity", sim)
        if cat in ("router", "norm"):
            return TensorMergePlan(name, cat, 1.0, "keep_a", "identity", sim)
        if sa.dead and not sb.dead:
            return TensorMergePlan(name, cat, 0.0, "keep_b", "identity", sim)
        if sb.dead and not sa.dead:
            return TensorMergePlan(name, cat, 1.0, "keep_a", "identity", sim)
        if sim > 0.85:
            return TensorMergePlan(name, cat, a, "weighted", "identity", sim)
        if sim > 0.55:
            return TensorMergePlan(name, cat, a, "slerp", "identity", sim)
        if sa.shape == sb.shape and sa.numel >= 16:
            return TensorMergePlan(name, cat, a, "ties", "identity", sim)
        return TensorMergePlan(name, cat, a, "projection", "procrustes", sim)
