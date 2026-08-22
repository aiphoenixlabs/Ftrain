"""
FTRAIN Merge Safety Utilities v1.1
==================================

Numerical health checks and sanitization for model-merge state dictionaries.

Public API
----------
SafetyReport
check_state_dict
sanitize
validate_and_sanitize

Design goals
------------
- Catch NaN/Inf corruption before saving a merged model.
- Detect suspicious norm explosion/collapse against a trusted baseline.
- Keep integer/bool structural tensors out of floating-point norm logic.
- Prefer baseline restoration for catastrophic failures.
- Rescale ordinary norm anomalies without unnecessary full-state copies.
- Preserve dtype/device.
- Preserve the original public API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch


DEFAULT_EXPLOSION_FACTOR = 5.0
DEFAULT_COLLAPSE_FACTOR = 0.2
DEFAULT_EPS = 1e-8
DEFAULT_FINITE_CLAMP = 1e6

_STRUCTURAL_HINTS = (
    "position_ids",
    "token_type_ids",
    "indices",
    "index",
    "mask",
)


def _floating(t: torch.Tensor) -> bool:
    return bool(t.is_floating_point() or t.is_complex())


def _norm(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    # FP32 reduction avoids fp16 overflow in norm calculation.
    return float(t.detach().float().norm().item())


def _has_nan(t: torch.Tensor) -> bool:
    if not _floating(t):
        return False
    try:
        return bool(torch.isnan(t).any().item())
    except RuntimeError:
        return False


def _has_inf(t: torch.Tensor) -> bool:
    if not _floating(t):
        return False
    try:
        return bool(torch.isinf(t).any().item())
    except RuntimeError:
        return False


def _is_structural(name: str) -> bool:
    name = str(name).lower()
    return any(x in name for x in _STRUCTURAL_HINTS)


def _same_shape(a: torch.Tensor, b: Any) -> bool:
    return torch.is_tensor(b) and tuple(a.shape) == tuple(b.shape)


def _copy_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return source.detach().to(
        device=target.device,
        dtype=target.dtype,
    ).clone()


def _finite_repair(
    t: torch.Tensor,
    clamp_value: float = DEFAULT_FINITE_CLAMP,
) -> torch.Tensor:
    if not _floating(t):
        return t
    return torch.nan_to_num(
        t,
        nan=0.0,
        posinf=clamp_value,
        neginf=-clamp_value,
    )


def _rescale_to_ratio(
    t: torch.Tensor,
    baseline: torch.Tensor,
    target_ratio: float,
) -> torch.Tensor:
    current = _norm(t)
    base = _norm(baseline)

    if current <= DEFAULT_EPS or base <= DEFAULT_EPS:
        return t.detach().clone()

    desired = base * max(float(target_ratio), DEFAULT_EPS)
    scale = desired / max(current, DEFAULT_EPS)

    if not math.isfinite(scale):
        return t.detach().clone()

    out = t.detach().float() * float(scale)
    out = torch.nan_to_num(
        out,
        nan=0.0,
        posinf=DEFAULT_FINITE_CLAMP,
        neginf=-DEFAULT_FINITE_CLAMP,
    )

    return out.to(
        device=t.device,
        dtype=t.dtype,
    )


@dataclass
class SafetyReport:
    ok: bool = True
    total_tensors: int = 0
    safe_tensors: int = 0
    nan_keys: List[str] = field(default_factory=list)
    inf_keys: List[str] = field(default_factory=list)
    exploded_keys: List[str] = field(default_factory=list)
    collapsed_keys: List[str] = field(default_factory=list)
    zero_keys: List[str] = field(default_factory=list)

    # Enhanced diagnostics.
    shape_mismatch_keys: List[str] = field(default_factory=list)
    nonfinite_element_keys: List[str] = field(default_factory=list)
    repaired_keys: List[str] = field(default_factory=list)
    fallback_keys: List[str] = field(default_factory=list)
    structural_skipped: int = 0

    max_ratio: float = 1.0
    min_ratio: float = 1.0
    mean_ratio: float = 1.0

    total_elements: int = 0
    finite_elements: int = 0
    finite_element_ratio: float = 1.0

    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.total_tensors == 0:
            return "⚠️ EMPTY SAFETY REPORT (0 tensors)"

        issues = []
        if self.nan_keys:
            issues.append(f"NaN: {len(self.nan_keys)}")
        if self.inf_keys:
            issues.append(f"Inf: {len(self.inf_keys)}")
        if self.exploded_keys:
            issues.append(f"Explosion: {len(self.exploded_keys)}")
        if self.collapsed_keys:
            issues.append(f"Collapse: {len(self.collapsed_keys)}")
        if self.zero_keys:
            issues.append(f"Zeroed: {len(self.zero_keys)}")
        if self.shape_mismatch_keys:
            issues.append(f"Shape: {len(self.shape_mismatch_keys)}")

        if self.ok and not issues:
            return (
                f"✅ SAFE ({self.safe_tensors}/{self.total_tensors} tensors clean) "
                f"| Ratio {self.min_ratio:.3g}x–{self.max_ratio:.3g}x "
                f"| Finite {self.finite_element_ratio:.2%}"
            )

        return (
            f"❌ UNSAFE ({', '.join(issues) or 'Unknown issue'}) "
            f"| Max Ratio {self.max_ratio:.3g}x "
            f"| Min Ratio {self.min_ratio:.3g}x "
            f"| Mean Ratio {self.mean_ratio:.3g}x "
            f"| Finite {self.finite_element_ratio:.2%}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "total_tensors": self.total_tensors,
            "safe_tensors": self.safe_tensors,
            "nan_keys": list(self.nan_keys),
            "inf_keys": list(self.inf_keys),
            "exploded_keys": list(self.exploded_keys),
            "collapsed_keys": list(self.collapsed_keys),
            "zero_keys": list(self.zero_keys),
            "shape_mismatch_keys": list(self.shape_mismatch_keys),
            "nonfinite_element_keys": list(self.nonfinite_element_keys),
            "repaired_keys": list(self.repaired_keys),
            "fallback_keys": list(self.fallback_keys),
            "structural_skipped": self.structural_skipped,
            "max_ratio": float(self.max_ratio),
            "min_ratio": float(self.min_ratio),
            "mean_ratio": float(self.mean_ratio),
            "total_elements": self.total_elements,
            "finite_elements": self.finite_elements,
            "finite_element_ratio": self.finite_element_ratio,
            "notes": list(self.notes),
        }


def check_state_dict(
    sd: dict,
    baseline: Optional[dict] = None,
    norm_explosion_factor: float = DEFAULT_EXPLOSION_FACTOR,
    norm_collapse_factor: float = DEFAULT_COLLAPSE_FACTOR,
    *,
    zero_norm_epsilon: float = DEFAULT_EPS,
    ignore_structural_tensors: bool = True,
) -> SafetyReport:
    """
    Check numerical health of a state dictionary.

    Relative norm checks are performed only when a matching baseline tensor
    exists. Structural integer/bool tensors are not treated as learned weights.
    """
    if sd is None:
        raise ValueError("sd cannot be None.")
    if norm_explosion_factor <= 0:
        raise ValueError("norm_explosion_factor must be > 0.")
    if norm_collapse_factor < 0:
        raise ValueError("norm_collapse_factor must be >= 0.")

    report = SafetyReport()
    ratios: List[float] = []

    tensor_items = [
        (k, v) for k, v in sd.items() if torch.is_tensor(v)
    ]
    report.total_tensors = len(tensor_items)

    for key, value in tensor_items:
        total = int(value.numel())
        report.total_elements += total

        if _floating(value):
            finite = int(torch.isfinite(value).sum().item()) if total else 0
        else:
            finite = total

        report.finite_elements += finite

        if total and finite != total:
            report.nonfinite_element_keys.append(key)

        if _has_nan(value):
            report.ok = False
            report.nan_keys.append(key)
            continue

        if _has_inf(value):
            report.ok = False
            report.inf_keys.append(key)
            continue

        if not _floating(value):
            report.safe_tensors += 1
            continue

        structural = (
            ignore_structural_tensors
            and _is_structural(key)
        )

        current_norm = _norm(value)

        if (
            current_norm <= zero_norm_epsilon
            and not structural
        ):
            report.ok = False
            report.zero_keys.append(key)
            continue

        if (
            baseline is None
            or key not in baseline
            or not torch.is_tensor(baseline[key])
        ):
            report.safe_tensors += 1
            continue

        base = baseline[key]

        if tuple(value.shape) != tuple(base.shape):
            report.ok = False
            report.shape_mismatch_keys.append(key)
            continue

        if (
            structural
            or not _floating(base)
        ):
            report.structural_skipped += 1
            report.safe_tensors += 1
            continue

        base_norm = _norm(base)

        if base_norm <= zero_norm_epsilon:
            # A near-zero baseline makes relative ratios meaningless.
            report.notes.append(
                f"{key}: baseline norm near zero; ratio check skipped."
            )
            report.safe_tensors += 1
            continue

        ratio = current_norm / max(base_norm, zero_norm_epsilon)
        if not math.isfinite(ratio):
            report.ok = False
            report.exploded_keys.append(key)
            continue

        ratios.append(ratio)
        report.max_ratio = max(report.max_ratio, ratio)
        report.min_ratio = min(report.min_ratio, ratio)

        if ratio > norm_explosion_factor:
            report.ok = False
            report.exploded_keys.append(key)
            continue

        if ratio < norm_collapse_factor:
            report.ok = False
            report.collapsed_keys.append(key)
            continue

        report.safe_tensors += 1

    if ratios:
        report.min_ratio = min(ratios)
        report.max_ratio = max(ratios)
        report.mean_ratio = sum(ratios) / len(ratios)

    if report.total_elements:
        report.finite_element_ratio = (
            report.finite_elements / report.total_elements
        )

    report.ok = report.ok and (
        not report.nonfinite_element_keys
        and not report.shape_mismatch_keys
    )

    return report


def sanitize(
    sd: dict,
    baseline: Optional[dict] = None,
    norm_explosion_factor: float = DEFAULT_EXPLOSION_FACTOR,
    norm_collapse_factor: float = DEFAULT_COLLAPSE_FACTOR,
    mode: str = "rescale",
    inplace: bool = True,
) -> dict:
    """
    Repair a damaged state dictionary.

    Modes
    -----
    rescale:
        Restore ordinary norm anomalies while preserving tensor direction.
        Catastrophic nonfinite tensors use baseline fallback when possible.

    fallback:
        Replace anomalies with baseline whenever a shape-compatible baseline
        exists.

    finite:
        Only repair NaN/Inf. Finite norm anomalies are left untouched.

    hybrid:
        Baseline fallback for catastrophic/collapsed tensors; bounded rescale
        for large explosions.
    """
    if sd is None:
        raise ValueError("sd cannot be None.")

    mode = str(mode).strip().lower()
    valid = {"rescale", "fallback", "finite", "hybrid"}
    if mode not in valid:
        raise ValueError(
            f"Unknown sanitize mode {mode!r}; expected one of {sorted(valid)}."
        )

    out = sd if inplace else dict(sd)

    for key, value in list(sd.items()):
        if not torch.is_tensor(value):
            continue

        if not _floating(value):
            continue

        # 1) NaN/Inf -> baseline if possible, otherwise finite repair.
        if _has_nan(value) or _has_inf(value):
            if (
                baseline is not None
                and key in baseline
                and _same_shape(value, baseline[key])
            ):
                out[key] = _copy_like(baseline[key], value)
            else:
                out[key] = _finite_repair(value)
            continue

        # 2) Without a compatible baseline there is no trustworthy relative
        # norm target.
        if (
            baseline is None
            or key not in baseline
            or not torch.is_tensor(baseline[key])
            or not _same_shape(value, baseline[key])
            or not _floating(baseline[key])
        ):
            if mode == "finite":
                out[key] = _finite_repair(value)
            continue

        base = baseline[key]
        current_norm = _norm(value)
        base_norm = _norm(base)

        # A zero baseline should never be used as a multiplicative target.
        if base_norm <= DEFAULT_EPS:
            if current_norm > DEFAULT_EPS and mode in {"fallback", "hybrid"}:
                out[key] = _copy_like(base, value)
            continue

        ratio = current_norm / max(base_norm, DEFAULT_EPS)

        # Healthy tensor: preserve exactly.
        if (
            current_norm > DEFAULT_EPS
            and norm_collapse_factor <= ratio <= norm_explosion_factor
        ):
            continue

        # 3) Collapsed/zero tensor.
        if (
            current_norm <= DEFAULT_EPS
            or not math.isfinite(ratio)
            or ratio < norm_collapse_factor
        ):
            # Collapses are dangerous: the old implementation could amplify a
            # nearly-zero tensor using a huge scale factor. Baseline fallback is
            # safer when available.
            if mode in {"fallback", "hybrid"}:
                out[key] = _copy_like(base, value)
            elif mode == "rescale" and current_norm > DEFAULT_EPS:
                out[key] = _rescale_to_ratio(
                    value,
                    base,
                    norm_collapse_factor,
                )
            else:
                out[key] = _copy_like(base, value)

            continue

        # 4) Explosion.
        if ratio > norm_explosion_factor:
            if mode == "fallback":
                out[key] = _copy_like(base, value)
            elif mode in {"rescale", "hybrid"}:
                out[key] = _rescale_to_ratio(
                    value,
                    base,
                    norm_explosion_factor,
                )

    return out


def validate_and_sanitize(
    sd: dict,
    baseline: Optional[dict] = None,
    *,
    norm_explosion_factor: float = DEFAULT_EXPLOSION_FACTOR,
    norm_collapse_factor: float = DEFAULT_COLLAPSE_FACTOR,
    mode: str = "hybrid",
    inplace: bool = True,
) -> Tuple[SafetyReport, dict]:
    """
    Check, repair, then verify a state dictionary.

    Returns the pre-repair report and the repaired dictionary.
    """
    report = check_state_dict(
        sd,
        baseline=baseline,
        norm_explosion_factor=norm_explosion_factor,
        norm_collapse_factor=norm_collapse_factor,
    )

    if report.ok:
        return report, sd

    repaired = sanitize(
        sd,
        baseline=baseline,
        norm_explosion_factor=norm_explosion_factor,
        norm_collapse_factor=norm_collapse_factor,
        mode=mode,
        inplace=inplace,
    )

    post = check_state_dict(
        repaired,
        baseline=baseline,
        norm_explosion_factor=norm_explosion_factor,
        norm_collapse_factor=norm_collapse_factor,
    )

    changed = []
    for key, old_value in sd.items():
        new_value = repaired.get(key)
        if torch.is_tensor(old_value) and torch.is_tensor(new_value):
            try:
                if not torch.equal(old_value, new_value):
                    changed.append(key)
            except RuntimeError:
                changed.append(key)

    post.repaired_keys = changed
    post.fallback_keys = [
        key
        for key in changed
        if (
            baseline is not None
            and key in baseline
            and torch.is_tensor(baseline[key])
            and torch.is_tensor(repaired.get(key))
            and torch.equal(
                repaired[key],
                baseline[key].to(
                    device=repaired[key].device,
                    dtype=repaired[key].dtype,
                ),
            )
        )
    ]

    return report, repaired


__all__ = [
    "SafetyReport",
    "check_state_dict",
    "sanitize",
    "validate_and_sanitize",
]
