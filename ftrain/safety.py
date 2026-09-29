"""
FTRAIN Merge Safety Utilities v2.0
==================================

Deep numerical health checks, anomaly detection, repair, and verification
for model-merge state dictionaries.

Public API (v1 compatible)
--------------------------
SafetyReport
check_state_dict
sanitize
validate_and_sanitize

Additional API
--------------
SafetyConfig
TensorSafety
check_tensor
repair_tensor
summarize_report

Design goals
------------
- Catch NaN/Inf corruption before saving a merged model.
- Detect norm explosion/collapse against a trusted baseline.
- Detect extreme values and distribution drift.
- Keep integer/bool structural tensors out of learned-weight logic.
- Prefer baseline restoration for catastrophic failures.
- Rescale ordinary norm explosions when it is safe to do so.
- Preserve dtype and device.
- Avoid mutating inputs unless explicitly requested.
- Verify the result after repair.
- Produce rich diagnostics that FTRAIN/Captain can consume.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_EXPLOSION_FACTOR = 5.0
DEFAULT_COLLAPSE_FACTOR = 0.2
DEFAULT_EPS = 1e-8
DEFAULT_FINITE_CLAMP = 1e6
DEFAULT_ABS_VALUE_LIMIT = 1e6
DEFAULT_STD_RATIO_LIMIT = 6.0
DEFAULT_MEAN_SHIFT_LIMIT = 8.0
DEFAULT_OUTLIER_FRACTION_LIMIT = 0.02

_STRUCTURAL_HINTS = (
    "position_ids",
    "token_type_ids",
    "indices",
    "index",
    "mask",
    "attention_mask",
    "special_tokens_mask",
)

_SENSITIVE_HINTS = (
    "embed_tokens",
    "tok_embeddings",
    "word_embeddings",
    "lm_head",
    "output_projection",
    "layernorm",
    "layer_norm",
    ".norm.",
    ".ln",
)

ACTION_CLEAN = "clean"
ACTION_SKIP = "skip"
ACTION_REPAIR_FINITE = "repair_finite"
ACTION_FALLBACK = "fallback_baseline"
ACTION_RESCALE = "rescale"
ACTION_REVIEW = "review"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SafetyConfig:
    """Centralized configuration for safety analysis."""

    norm_explosion_factor: float = DEFAULT_EXPLOSION_FACTOR
    norm_collapse_factor: float = DEFAULT_COLLAPSE_FACTOR
    zero_norm_epsilon: float = DEFAULT_EPS
    finite_clamp: float = DEFAULT_FINITE_CLAMP
    absolute_value_limit: float = DEFAULT_ABS_VALUE_LIMIT
    std_ratio_limit: float = DEFAULT_STD_RATIO_LIMIT
    mean_shift_limit: float = DEFAULT_MEAN_SHIFT_LIMIT
    outlier_fraction_limit: float = DEFAULT_OUTLIER_FRACTION_LIMIT
    ignore_structural_tensors: bool = True
    protect_sensitive_tensors: bool = True
    allow_rescale: bool = True
    fallback_on_nonfinite: bool = True
    fallback_on_collapse: bool = True

    def validate(self) -> None:
        if self.norm_explosion_factor <= 0:
            raise ValueError("norm_explosion_factor must be > 0")
        if self.norm_collapse_factor < 0:
            raise ValueError("norm_collapse_factor must be >= 0")
        if self.zero_norm_epsilon <= 0:
            raise ValueError("zero_norm_epsilon must be > 0")
        if self.finite_clamp <= 0:
            raise ValueError("finite_clamp must be > 0")
        if self.absolute_value_limit <= 0:
            raise ValueError("absolute_value_limit must be > 0")
        if self.std_ratio_limit <= 0:
            raise ValueError("std_ratio_limit must be > 0")
        if self.mean_shift_limit <= 0:
            raise ValueError("mean_shift_limit must be > 0")
        if not 0 <= self.outlier_fraction_limit <= 1:
            raise ValueError("outlier_fraction_limit must be in [0,1]")


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _floating(t: torch.Tensor) -> bool:
    return bool(t.is_floating_point() or t.is_complex())


def _norm(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    try:
        return float(t.detach().float().norm().item())
    except (RuntimeError, ValueError):
        return float("inf")


def _mean(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    try:
        return float(t.detach().float().mean().item())
    except (RuntimeError, ValueError):
        return float("nan")


def _std(t: torch.Tensor) -> float:
    if t.numel() <= 1:
        return 0.0
    try:
        return float(t.detach().float().std(unbiased=False).item())
    except (RuntimeError, ValueError):
        return float("nan")


def _max_abs(t: torch.Tensor) -> float:
    if t.numel() == 0:
        return 0.0
    try:
        return float(t.detach().float().abs().max().item())
    except (RuntimeError, ValueError):
        return float("inf")


def _finite_count(t: torch.Tensor) -> int:
    if t.numel() == 0:
        return 0
    if not _floating(t):
        return int(t.numel())
    try:
        return int(torch.isfinite(t).sum().item())
    except RuntimeError:
        return 0


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
    lowered = str(name).lower()
    return any(token in lowered for token in _STRUCTURAL_HINTS)


def _is_sensitive(name: str) -> bool:
    lowered = str(name).lower()
    return any(token in lowered for token in _SENSITIVE_HINTS)


def _same_shape(a: torch.Tensor, b: Any) -> bool:
    return torch.is_tensor(b) and tuple(a.shape) == tuple(b.shape)


def _same_dtype_kind(a: torch.Tensor, b: torch.Tensor) -> bool:
    return (
        a.is_floating_point() == b.is_floating_point()
        and a.is_complex() == b.is_complex()
    )


def _copy_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return source.detach().to(
        device=target.device,
        dtype=target.dtype,
    ).clone()


def _finite_repair(t: torch.Tensor, clamp_value: float) -> torch.Tensor:
    if not _floating(t):
        return t
    return torch.nan_to_num(
        t.detach(),
        nan=0.0,
        posinf=clamp_value,
        neginf=-clamp_value,
    ).to(device=t.device, dtype=t.dtype)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / max(abs(float(denominator)), DEFAULT_EPS)


def _rescale_to_ratio(
    t: torch.Tensor,
    baseline: torch.Tensor,
    target_ratio: float,
    finite_clamp: float = DEFAULT_FINITE_CLAMP,
) -> torch.Tensor:
    current = _norm(t)
    base = _norm(baseline)

    if (
        current <= DEFAULT_EPS
        or not math.isfinite(current)
        or base <= DEFAULT_EPS
        or not math.isfinite(base)
    ):
        return t.detach().clone()

    desired = base * max(float(target_ratio), DEFAULT_EPS)
    scale = desired / max(current, DEFAULT_EPS)

    if not math.isfinite(scale):
        return t.detach().clone()

    out = t.detach().float() * float(scale)
    out = torch.nan_to_num(
        out,
        nan=0.0,
        posinf=finite_clamp,
        neginf=-finite_clamp,
    )

    return out.to(device=t.device, dtype=t.dtype)


def _distribution_stats(
    value: torch.Tensor,
    baseline: torch.Tensor,
    eps: float,
) -> Tuple[float, float, float]:
    x = value.detach().float().reshape(-1)
    b = baseline.detach().float().reshape(-1)

    if x.numel() == 0 or b.numel() == 0:
        return 0.0, 1.0, 0.0

    mean_x = float(x.mean().item())
    mean_b = float(b.mean().item())
    std_x = float(x.std(unbiased=False).item())
    std_b = float(b.std(unbiased=False).item())

    # A nearly constant baseline has no useful standard-deviation reference.
    # In that case, distribution drift should not turn a successful norm repair
    # into a false failure.
    if abs(std_b) <= eps:
        mean_shift = abs(mean_x - mean_b) / max(abs(mean_b), 1.0)
        std_ratio = 1.0 if abs(std_x) <= eps else float("inf")
        outlier_fraction = 0.0
        return mean_shift, std_ratio, outlier_fraction

    scale = abs(std_b)
    mean_shift = abs(mean_x - mean_b) / scale
    std_ratio = max(std_x, eps) / scale

    centered = x - mean_b
    outlier_fraction = float(
        (centered.abs() > 6.0 * scale).float().mean().item()
    )

    return mean_shift, std_ratio, outlier_fraction


# ---------------------------------------------------------------------------
# Tensor-level diagnostics
# ---------------------------------------------------------------------------

@dataclass
class TensorSafety:
    key: str
    ok: bool
    action: str
    category: str
    numel: int
    dtype: str
    device: str
    norm: float
    baseline_norm: float
    norm_ratio: float
    mean: float
    baseline_mean: float
    std: float
    baseline_std: float
    max_abs: float
    finite_ratio: float
    mean_shift: float
    std_ratio: float
    outlier_fraction: float
    structural: bool
    sensitive: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def check_tensor(
    key: str,
    value: torch.Tensor,
    *,
    baseline: Optional[torch.Tensor] = None,
    config: Optional[SafetyConfig] = None,
) -> TensorSafety:
    """Perform a conservative numerical-health check for one tensor."""
    cfg = config or SafetyConfig()
    cfg.validate()

    structural = cfg.ignore_structural_tensors and _is_structural(key)
    sensitive = cfg.protect_sensitive_tensors and _is_sensitive(key)

    total = int(value.numel())
    finite_count = _finite_count(value)
    finite_ratio = finite_count / total if total else 1.0

    norm = _norm(value)
    mean = _mean(value)
    std = _std(value)
    max_abs = _max_abs(value)

    dtype = str(value.dtype).replace("torch.", "")
    device = str(value.device)

    if not _floating(value):
        return TensorSafety(
            key, True, ACTION_SKIP,
            "structural" if structural else "non_floating",
            total, dtype, device,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
            0.0, 1.0, 0.0,
            structural, sensitive,
            "non-floating tensor skipped",
        )

    if finite_ratio < 1.0:
        base_norm = _norm(baseline) if torch.is_tensor(baseline) else 0.0
        return TensorSafety(
            key, False, ACTION_FALLBACK, "nonfinite",
            total, dtype, device,
            norm, base_norm,
            float("inf"),
            mean,
            _mean(baseline) if torch.is_tensor(baseline) else 0.0,
            std,
            _std(baseline) if torch.is_tensor(baseline) else 0.0,
            max_abs, finite_ratio,
            float("inf"), float("inf"), 1.0,
            structural, sensitive,
            "NaN/Inf detected",
        )

    if total == 0:
        return TensorSafety(
            key, True, ACTION_CLEAN, "empty",
            0, dtype, device,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0,
            0.0, 1.0, 0.0,
            structural, sensitive,
            "empty tensor",
        )

    if structural:
        return TensorSafety(
            key, True, ACTION_SKIP, "structural",
            total, dtype, device,
            norm, 0.0, 1.0,
            mean, 0.0, std, 0.0,
            max_abs, 1.0,
            0.0, 1.0, 0.0,
            True, sensitive,
            "structural tensor excluded from learned-weight checks",
        )

    if max_abs > cfg.absolute_value_limit:
        base_norm = _norm(baseline) if torch.is_tensor(baseline) else 0.0
        ratio = _safe_ratio(norm, base_norm) if base_norm > cfg.zero_norm_epsilon else 1.0
        return TensorSafety(
            key, False, ACTION_FALLBACK, "extreme_value",
            total, dtype, device,
            norm, base_norm, ratio,
            mean,
            _mean(baseline) if torch.is_tensor(baseline) else 0.0,
            std,
            _std(baseline) if torch.is_tensor(baseline) else 0.0,
            max_abs, 1.0,
            0.0, 1.0, 1.0,
            False, sensitive,
            "absolute value exceeds configured safety limit",
        )

    baseline_ok = (
        baseline is not None
        and torch.is_tensor(baseline)
        and _same_shape(value, baseline)
        and _same_dtype_kind(value, baseline)
        and _floating(baseline)
    )

    if not baseline_ok:
        if baseline is not None and torch.is_tensor(baseline) and not _same_shape(value, baseline):
            category = "shape_mismatch"
            reason = "baseline shape mismatch; relative repair is unsafe"
        else:
            category = "no_baseline"
            reason = "no compatible baseline available"

        return TensorSafety(
            key,
            True if category == "no_baseline" else False,
            ACTION_REVIEW if category == "shape_mismatch" else ACTION_CLEAN,
            category,
            total, dtype, device,
            norm, 0.0, 1.0,
            mean, 0.0, std, 0.0,
            max_abs, 1.0,
            0.0, 1.0, 0.0,
            False, sensitive,
            reason,
        )

    base_norm = _norm(baseline)
    base_mean = _mean(baseline)
    base_std = _std(baseline)

    if base_norm <= cfg.zero_norm_epsilon:
        if norm <= cfg.zero_norm_epsilon:
            return TensorSafety(
                key, True, ACTION_REVIEW, "double_zero",
                total, dtype, device,
                norm, base_norm, 1.0,
                mean, base_mean, std, base_std,
                max_abs, 1.0,
                0.0, 1.0, 0.0,
                False, sensitive,
                "both tensor and baseline are near zero",
            )

        return TensorSafety(
            key, False,
            ACTION_FALLBACK if cfg.fallback_on_collapse else ACTION_REVIEW,
            "baseline_zero",
            total, dtype, device,
            norm, base_norm, float("inf"),
            mean, base_mean, std, base_std,
            max_abs, 1.0,
            0.0, 1.0, 0.0,
            False, sensitive,
            "baseline norm is near zero",
        )

    ratio = norm / max(base_norm, cfg.zero_norm_epsilon)
    mean_shift, std_ratio, outlier_fraction = _distribution_stats(
        value,
        baseline,
        cfg.zero_norm_epsilon,
    )

    if norm <= cfg.zero_norm_epsilon:
        return TensorSafety(
            key, False,
            ACTION_FALLBACK if cfg.fallback_on_collapse else ACTION_REVIEW,
            "zero_collapse",
            total, dtype, device,
            norm, base_norm, ratio,
            mean, base_mean, std, base_std,
            max_abs, 1.0,
            mean_shift, std_ratio, outlier_fraction,
            False, sensitive,
            "tensor norm collapsed to approximately zero",
        )

    if ratio > cfg.norm_explosion_factor:
        action = ACTION_RESCALE if cfg.allow_rescale else ACTION_REVIEW
        if sensitive:
            action = ACTION_FALLBACK
        return TensorSafety(
            key, False, action, "norm_explosion",
            total, dtype, device,
            norm, base_norm, ratio,
            mean, base_mean, std, base_std,
            max_abs, 1.0,
            mean_shift, std_ratio, outlier_fraction,
            False, sensitive,
            "tensor norm exceeds configured explosion factor",
        )

    if ratio < cfg.norm_collapse_factor:
        action = ACTION_FALLBACK if cfg.fallback_on_collapse else ACTION_RESCALE
        return TensorSafety(
            key, False, action, "norm_collapse",
            total, dtype, device,
            norm, base_norm, ratio,
            mean, base_mean, std, base_std,
            max_abs, 1.0,
            mean_shift, std_ratio, outlier_fraction,
            False, sensitive,
            "tensor norm is below configured collapse factor",
        )

    distribution_problem = (
        std_ratio > cfg.std_ratio_limit
        or std_ratio < 1.0 / cfg.std_ratio_limit
        or mean_shift > cfg.mean_shift_limit
        or outlier_fraction > cfg.outlier_fraction_limit
    )

    if distribution_problem:
        return TensorSafety(
            key, False,
            ACTION_FALLBACK if sensitive else ACTION_REVIEW,
            "distribution_drift",
            total, dtype, device,
            norm, base_norm, ratio,
            mean, base_mean, std, base_std,
            max_abs, 1.0,
            mean_shift, std_ratio, outlier_fraction,
            False, sensitive,
            "distribution drift detected against baseline",
        )

    return TensorSafety(
        key, True, ACTION_CLEAN, "clean",
        total, dtype, device,
        norm, base_norm, ratio,
        mean, base_mean, std, base_std,
        max_abs, 1.0,
        mean_shift, std_ratio, outlier_fraction,
        False, sensitive,
        "tensor is within configured thresholds",
    )


# ---------------------------------------------------------------------------
# Safety report
# ---------------------------------------------------------------------------

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

    shape_mismatch_keys: List[str] = field(default_factory=list)
    nonfinite_element_keys: List[str] = field(default_factory=list)
    repaired_keys: List[str] = field(default_factory=list)
    fallback_keys: List[str] = field(default_factory=list)
    review_keys: List[str] = field(default_factory=list)
    extreme_value_keys: List[str] = field(default_factory=list)
    distribution_drift_keys: List[str] = field(default_factory=list)
    dtype_mismatch_keys: List[str] = field(default_factory=list)
    device_mismatch_keys: List[str] = field(default_factory=list)
    structural_skipped: int = 0
    sensitive_keys: List[str] = field(default_factory=list)

    max_ratio: float = 1.0
    min_ratio: float = 1.0
    mean_ratio: float = 1.0
    total_elements: int = 0
    finite_elements: int = 0
    finite_element_ratio: float = 1.0
    max_abs_value: float = 0.0
    mean_abs_mean_shift: float = 0.0
    mean_std_ratio: float = 1.0

    tensor_details: Dict[str, TensorSafety] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.total_tensors == 0:
            return "⚠️ EMPTY SAFETY REPORT (0 tensors)"

        issues: List[str] = []
        groups = (
            ("NaN", self.nan_keys),
            ("Inf", self.inf_keys),
            ("Explosion", self.exploded_keys),
            ("Collapse", self.collapsed_keys),
            ("Zero", self.zero_keys),
            ("Shape", self.shape_mismatch_keys),
            ("Extreme", self.extreme_value_keys),
            ("Drift", self.distribution_drift_keys),
            ("Review", self.review_keys),
        )
        for label, values in groups:
            if values:
                issues.append(f"{label}: {len(values)}")

        status = "✅ SAFE" if self.ok else "❌ UNSAFE"
        suffix = f" | Issues: {', '.join(issues)}" if issues else ""

        return (
            f"{status} ({self.safe_tensors}/{self.total_tensors} tensors clean)"
            f" | Ratio {self.min_ratio:.3g}x–{self.max_ratio:.3g}x"
            f" | Finite {self.finite_element_ratio:.2%}"
            f" | MaxAbs {self.max_abs_value:.4g}"
            f"{suffix}"
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
            "review_keys": list(self.review_keys),
            "extreme_value_keys": list(self.extreme_value_keys),
            "distribution_drift_keys": list(self.distribution_drift_keys),
            "dtype_mismatch_keys": list(self.dtype_mismatch_keys),
            "device_mismatch_keys": list(self.device_mismatch_keys),
            "structural_skipped": self.structural_skipped,
            "sensitive_keys": list(self.sensitive_keys),
            "max_ratio": float(self.max_ratio),
            "min_ratio": float(self.min_ratio),
            "mean_ratio": float(self.mean_ratio),
            "total_elements": self.total_elements,
            "finite_elements": self.finite_elements,
            "finite_element_ratio": float(self.finite_element_ratio),
            "max_abs_value": float(self.max_abs_value),
            "mean_abs_mean_shift": float(self.mean_abs_mean_shift),
            "mean_std_ratio": float(self.mean_std_ratio),
            "tensor_details": {
                key: value.to_dict()
                for key, value in self.tensor_details.items()
            },
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# State-dict check
# ---------------------------------------------------------------------------

def check_state_dict(
    sd: dict,
    baseline: Optional[dict] = None,
    norm_explosion_factor: float = DEFAULT_EXPLOSION_FACTOR,
    norm_collapse_factor: float = DEFAULT_COLLAPSE_FACTOR,
    *,
    zero_norm_epsilon: float = DEFAULT_EPS,
    ignore_structural_tensors: bool = True,
) -> SafetyReport:
    """Check numerical health without modifying the supplied state dict."""
    if sd is None:
        raise ValueError("sd cannot be None")

    cfg = SafetyConfig(
        norm_explosion_factor=norm_explosion_factor,
        norm_collapse_factor=norm_collapse_factor,
        zero_norm_epsilon=zero_norm_epsilon,
        ignore_structural_tensors=ignore_structural_tensors,
    )
    cfg.validate()

    report = SafetyReport()
    tensor_items = [
        (key, value)
        for key, value in sd.items()
        if torch.is_tensor(value)
    ]
    report.total_tensors = len(tensor_items)

    ratios: List[float] = []
    mean_shifts: List[float] = []
    std_ratios: List[float] = []

    for key, value in tensor_items:
        total = int(value.numel())
        report.total_elements += total

        finite_count = _finite_count(value)
        report.finite_elements += finite_count

        if total and finite_count != total:
            report.nonfinite_element_keys.append(key)
        if _has_nan(value):
            report.nan_keys.append(key)
        if _has_inf(value):
            report.inf_keys.append(key)
        if _is_sensitive(key):
            report.sensitive_keys.append(key)

        baseline_value = None
        if baseline is not None and torch.is_tensor(baseline.get(key)):
            baseline_value = baseline[key]

        detail = check_tensor(
            key,
            value,
            baseline=baseline_value,
            config=cfg,
        )
        report.tensor_details[key] = detail
        report.max_abs_value = max(report.max_abs_value, detail.max_abs)

        if detail.structural:
            report.structural_skipped += 1

        category = detail.category
        if category in {"zero_collapse", "norm_collapse", "baseline_zero"}:
            report.collapsed_keys.append(key)
            if category == "zero_collapse":
                report.zero_keys.append(key)
        elif category == "norm_explosion":
            report.exploded_keys.append(key)
        elif category == "extreme_value":
            report.extreme_value_keys.append(key)
        elif category == "distribution_drift":
            report.distribution_drift_keys.append(key)
        elif category == "shape_mismatch":
            report.shape_mismatch_keys.append(key)

        if detail.action == ACTION_REVIEW:
            report.review_keys.append(key)

        if detail.baseline_norm > zero_norm_epsilon and math.isfinite(detail.norm_ratio):
            ratios.append(detail.norm_ratio)
        if math.isfinite(detail.mean_shift):
            mean_shifts.append(detail.mean_shift)
        if math.isfinite(detail.std_ratio):
            std_ratios.append(detail.std_ratio)

        if detail.ok or detail.action == ACTION_SKIP:
            report.safe_tensors += 1

        if baseline_value is not None:
            if tuple(value.shape) != tuple(baseline_value.shape):
                if key not in report.shape_mismatch_keys:
                    report.shape_mismatch_keys.append(key)
            if not _same_dtype_kind(value, baseline_value):
                report.dtype_mismatch_keys.append(key)
            if value.device != baseline_value.device:
                report.device_mismatch_keys.append(key)

    if ratios:
        report.min_ratio = min(ratios)
        report.max_ratio = max(ratios)
        report.mean_ratio = sum(ratios) / len(ratios)
    if mean_shifts:
        report.mean_abs_mean_shift = sum(mean_shifts) / len(mean_shifts)
    if std_ratios:
        report.mean_std_ratio = sum(std_ratios) / len(std_ratios)
    if report.total_elements:
        report.finite_element_ratio = report.finite_elements / report.total_elements

    report.ok = (
        not report.nan_keys
        and not report.inf_keys
        and not report.exploded_keys
        and not report.collapsed_keys
        and not report.zero_keys
        and not report.shape_mismatch_keys
        and not report.dtype_mismatch_keys
        and not report.extreme_value_keys
        and not report.distribution_drift_keys
        and not report.review_keys
        and report.finite_element_ratio >= 1.0
    )

    if report.distribution_drift_keys:
        report.notes.append(
            "Distribution drift is treated as a review signal because it may be a real model change."
        )
    if report.shape_mismatch_keys:
        report.notes.append(
            "Baseline shape mismatches are never repaired by multiplicative rescaling."
        )
    if report.dtype_mismatch_keys:
        report.notes.append(
            "Baseline numeric-kind mismatches restrict automatic fallback."
        )
    if report.finite_element_ratio < 1.0:
        report.notes.append(
            "Non-finite elements were detected. Do not save before repair/verification."
        )

    return report


# ---------------------------------------------------------------------------
# Tensor repair
# ---------------------------------------------------------------------------

def repair_tensor(
    key: str,
    value: torch.Tensor,
    *,
    baseline: Optional[torch.Tensor] = None,
    config: Optional[SafetyConfig] = None,
    mode: str = "hybrid",
) -> Tuple[torch.Tensor, TensorSafety]:
    """Repair one tensor and return ``(tensor, diagnostics)``."""
    cfg = config or SafetyConfig()
    cfg.validate()

    mode = str(mode).strip().lower()
    if mode not in {"finite", "rescale", "fallback", "hybrid"}:
        raise ValueError(
            "mode must be one of: finite, rescale, fallback, hybrid"
        )

    detail = check_tensor(
        key,
        value,
        baseline=baseline,
        config=cfg,
    )

    if detail.ok or detail.action == ACTION_SKIP:
        return value, detail

    # Non-finite corruption is catastrophic. Prefer a trusted baseline.
    if detail.category == "nonfinite":
        if (
            baseline is not None
            and _same_shape(value, baseline)
            and _floating(baseline)
            and cfg.fallback_on_nonfinite
            and mode != "finite"
        ):
            return _copy_like(baseline, value), detail
        return _finite_repair(value, cfg.finite_clamp), detail

    # Extreme values are safer to restore from baseline than to preserve.
    if detail.category == "extreme_value":
        if (
            baseline is not None
            and _same_shape(value, baseline)
            and _floating(baseline)
            and mode in {"fallback", "hybrid"}
        ):
            return _copy_like(baseline, value), detail
        return _finite_repair(value, cfg.finite_clamp), detail

    baseline_usable = (
        baseline is not None
        and torch.is_tensor(baseline)
        and _same_shape(value, baseline)
        and _same_dtype_kind(value, baseline)
        and _floating(baseline)
    )

    if not baseline_usable:
        if mode == "finite":
            return _finite_repair(value, cfg.finite_clamp), detail
        return value, detail

    if detail.category in {"zero_collapse", "baseline_zero", "norm_collapse"}:
        if (
            detail.sensitive
            or mode in {"fallback", "hybrid"}
            or mode == "rescale" and detail.category == "norm_collapse"
        ):
            return _copy_like(baseline, value), detail

    if detail.category == "norm_explosion":
        if mode == "fallback" or (detail.sensitive and mode == "hybrid"):
            return _copy_like(baseline, value), detail
        if mode in {"rescale", "hybrid"}:
            return _rescale_to_ratio(
                value,
                baseline,
                cfg.norm_explosion_factor,
                cfg.finite_clamp,
            ), detail

    # Distribution drift is deliberately not normalized blindly.
    # It can be meaningful learned behavior, so it remains a review item.
    if detail.category == "distribution_drift":
        if detail.sensitive and mode in {"fallback", "hybrid"}:
            return _copy_like(baseline, value), detail
        return value, detail

    return value, detail


# ---------------------------------------------------------------------------
# Full-state sanitization
# ---------------------------------------------------------------------------

def sanitize(
    sd: dict,
    baseline: Optional[dict] = None,
    norm_explosion_factor: float = DEFAULT_EXPLOSION_FACTOR,
    norm_collapse_factor: float = DEFAULT_COLLAPSE_FACTOR,
    mode: str = "rescale",
    inplace: bool = True,
) -> dict:
    """Repair a state dictionary while preserving the legacy API."""
    if sd is None:
        raise ValueError("sd cannot be None")

    cfg = SafetyConfig(
        norm_explosion_factor=norm_explosion_factor,
        norm_collapse_factor=norm_collapse_factor,
    )
    cfg.validate()

    out = sd if inplace else dict(sd)

    for key, value in list(sd.items()):
        if not torch.is_tensor(value):
            continue

        baseline_value = None
        if baseline is not None and torch.is_tensor(baseline.get(key)):
            baseline_value = baseline[key]

        repaired, _detail = repair_tensor(
            key,
            value,
            baseline=baseline_value,
            config=cfg,
            mode=mode,
        )

        if repaired is not value:
            out[key] = repaired

    return out


# ---------------------------------------------------------------------------
# Validate -> repair -> verify
# ---------------------------------------------------------------------------

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
    Check, repair, and verify a state dictionary.

    The returned report is the POST-rePAIR report and includes repair/fallback
    information, making it suitable as a final gate before model saving.
    """
    if sd is None:
        raise ValueError("sd cannot be None")

    pre = check_state_dict(
        sd,
        baseline=baseline,
        norm_explosion_factor=norm_explosion_factor,
        norm_collapse_factor=norm_collapse_factor,
    )

    if pre.ok:
        pre.notes.append(
            "Pre-repair safety check passed; state dictionary unchanged."
        )
        return pre, sd

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

    # With inplace=True the original mapping now points at the repaired tensors,
    # so comparing sd against repaired is not sufficient. Use the pre-repair
    # diagnostics as the authoritative list of tensors that required action.
    changed: List[str] = [
        key
        for key, detail in pre.tensor_details.items()
        if detail.action not in {ACTION_CLEAN, ACTION_SKIP}
    ]

    if not changed and not inplace:
        for key, old_value in sd.items():
            new_value = repaired.get(key)
            if torch.is_tensor(old_value) and torch.is_tensor(new_value):
                try:
                    if not torch.equal(old_value, new_value):
                        changed.append(key)
                except RuntimeError:
                    changed.append(key)

    post.repaired_keys = changed

    fallback_keys: List[str] = []
    if baseline is not None:
        for key in changed:
            base = baseline.get(key)
            current = repaired.get(key)
            if not (torch.is_tensor(base) and torch.is_tensor(current)):
                continue
            if not _same_shape(current, base):
                continue
            try:
                expected = base.to(
                    device=current.device,
                    dtype=current.dtype,
                )
                if torch.equal(current, expected):
                    fallback_keys.append(key)
            except RuntimeError:
                pass

    post.fallback_keys = fallback_keys
    post.notes.extend(
        [
            f"Pre-repair: {pre.summary()}",
            f"Post-repair: {post.summary()}",
            f"Changed tensors: {len(changed)}",
            f"Baseline fallbacks: {len(fallback_keys)}",
        ]
    )

    if post.ok:
        post.notes.append(
            "POST-REPAIR SAFETY CHECK PASSED."
        )
    else:
        post.notes.append(
            "POST-REPAIR SAFETY CHECK FAILED. Do not save/deploy without review."
        )

    return post, repaired


# ---------------------------------------------------------------------------
# Captain-friendly compact summary
# ---------------------------------------------------------------------------

def summarize_report(report: SafetyReport) -> Dict[str, Any]:
    """Return a compact summary suitable for Captain logging/UI."""
    if not isinstance(report, SafetyReport):
        raise TypeError("report must be a SafetyReport")

    if report.nan_keys or report.inf_keys:
        severity = "critical"
    elif report.exploded_keys or report.collapsed_keys or report.extreme_value_keys:
        severity = "high"
    elif report.distribution_drift_keys or report.review_keys:
        severity = "medium"
    else:
        severity = "low"

    return {
        "safe": bool(report.ok),
        "severity": severity,
        "total_tensors": report.total_tensors,
        "safe_tensors": report.safe_tensors,
        "repaired_tensors": len(report.repaired_keys),
        "fallback_tensors": len(report.fallback_keys),
        "finite_ratio": report.finite_element_ratio,
        "max_ratio": report.max_ratio,
        "min_ratio": report.min_ratio,
        "mean_ratio": report.mean_ratio,
        "max_abs_value": report.max_abs_value,
        "review_required": bool(report.review_keys),
    }


# ---------------------------------------------------------------------------
# Explicit exports
# ---------------------------------------------------------------------------

__all__ = [
    "SafetyConfig",
    "TensorSafety",
    "SafetyReport",
    "check_tensor",
    "check_state_dict",
    "repair_tensor",
    "sanitize",
    "validate_and_sanitize",
    "summarize_report",
]
