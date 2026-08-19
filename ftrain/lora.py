"""
FTRAIN Fast LoRA
================

Lightweight, dependency-free LoRA implementation for PyTorch ``nn.Linear``.

Features
--------
• Frozen base weights.
• Trainable LoRA A/B matrices.
• Correct LoRA scaling.
• Optional dropout.
• Device and dtype preservation.
• Safe recursive module injection.
• Duplicate-injection protection.
• LoRA enable/disable.
• Merge/unmerge support.
• State inspection helpers.
• Parameter counting.
• Defensive validation.
• Compatible with normal PyTorch optimizers.
• No PEFT dependency required.

Mathematical form
-----------------

    y = W(x) + scaling * B(A(dropout(x)))

where:

    scaling = alpha / rank

At initialization B is zero, therefore the LoRA branch initially contributes
zero to the model output.
"""

from __future__ import annotations

import math
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn

__all__ = [
    "FastLoraLinear",
    "inject",
    "count_lora_parameters",
    "mark_only_lora_trainable",
]


# =============================================================================
# Helpers
# =============================================================================


def _validate_rank(
    r: int,
) -> int:
    """Validate and normalize LoRA rank."""
    if isinstance(r, bool):
        raise TypeError(
            "LoRA rank 'r' must be an integer, not bool."
        )

    try:
        rank = int(r)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"LoRA rank must be an integer, got {r!r}."
        ) from exc

    if rank <= 0:
        raise ValueError(
            f"LoRA rank must be greater than zero, got {rank}."
        )

    return rank


def _validate_alpha(
    alpha: float,
) -> float:
    """Validate LoRA alpha."""
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"LoRA alpha must be numeric, got {alpha!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            "LoRA alpha must be finite."
        )

    if value <= 0:
        raise ValueError(
            f"LoRA alpha must be greater than zero, got {value}."
        )

    return value


def _validate_dropout(
    dropout: float,
) -> float:
    """Validate dropout probability."""
    try:
        value = float(dropout)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"LoRA dropout must be numeric, got {dropout!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            "LoRA dropout must be finite."
        )

    if not 0.0 <= value < 1.0:
        raise ValueError(
            "LoRA dropout must satisfy 0 <= dropout < 1."
        )

    return value


# =============================================================================
# Fast LoRA Linear
# =============================================================================


class FastLoraLinear(nn.Module):
    """
    LoRA wrapper around an existing ``torch.nn.Linear``.

    Parameters
    ----------
    base:
        Existing linear layer.

    r:
        LoRA rank.

    alpha:
        LoRA scaling parameter.

    dropout:
        Input dropout probability applied before LoRA A.

    Notes
    -----
    The base layer is frozen automatically.

    The LoRA branch is initialized so that:

        FastLoraLinear(x) == base(x)

    immediately after construction.
    """

    def __init__(
        self,
        base: nn.Linear,
        r: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not isinstance(
            base,
            nn.Linear,
        ):
            raise TypeError(
                "FastLoraLinear requires an nn.Linear base layer."
            )

        self.r = _validate_rank(r)
        self.alpha = _validate_alpha(alpha)
        self.dropout_p = _validate_dropout(dropout)

        self.base = base

        # ---------------------------------------------------------------------
        # Freeze base model
        # ---------------------------------------------------------------------

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        # ---------------------------------------------------------------------
        # Preserve base device and dtype
        # ---------------------------------------------------------------------

        device = base.weight.device
        dtype = base.weight.dtype

        if not (
            dtype.is_floating_point
            or dtype.is_complex
        ):
            raise TypeError(
                "LoRA requires a floating-point base weight dtype. "
                f"Got {dtype}."
            )

        # ---------------------------------------------------------------------
        # LoRA layers
        # ---------------------------------------------------------------------

        self.lora_A = nn.Linear(
            base.in_features,
            self.r,
            bias=False,
            device=device,
            dtype=dtype,
        )

        self.lora_B = nn.Linear(
            self.r,
            base.out_features,
            bias=False,
            device=device,
            dtype=dtype,
        )

        # ---------------------------------------------------------------------
        # Initialization
        # ---------------------------------------------------------------------

        # Standard LoRA initialization:
        #
        # A = Kaiming initialization
        # B = zeros
        #
        # Therefore the initial LoRA contribution is exactly zero.
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )

        nn.init.zeros_(
            self.lora_B.weight
        )

        # ---------------------------------------------------------------------
        # Scaling
        # ---------------------------------------------------------------------

        self.scaling = (
            self.alpha
            / float(self.r)
        )

        # Keep the old public attribute for compatibility.
        self.scale = self.scaling

        # ---------------------------------------------------------------------
        # Dropout
        # ---------------------------------------------------------------------

        self.dropout = (
            nn.Dropout(
                p=self.dropout_p
            )
            if self.dropout_p > 0.0
            else nn.Identity()
        )

        # ---------------------------------------------------------------------
        # Runtime state
        # ---------------------------------------------------------------------

        self.enabled = True
        self.merged = False

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute:

            base(x) + scaling * B(A(dropout(x)))
        """

        base_output = self.base(
            x
        )

        if not self.enabled:
            return base_output

        if self.merged:
            # LoRA weights are already incorporated into the base layer.
            return base_output

        lora_output = self.lora_B(
            self.lora_A(
                self.dropout(x)
            )
        )

        return (
            base_output
            + lora_output * self.scaling
        )

    # =========================================================================
    # Enable / disable
    # =========================================================================

    def enable_lora(
        self,
    ) -> None:
        """Enable the LoRA branch."""
        self.enabled = True

    def disable_lora(
        self,
    ) -> None:
        """Disable the LoRA branch without changing its parameters."""
        self.enabled = False

    # =========================================================================
    # Merge
    # =========================================================================

    @torch.no_grad()
    def merge(
        self,
    ) -> None:
        """
        Permanently add the LoRA update into the base weight.

        After merging:

            W <- W + scaling * (B @ A)

        The forward pass then only uses the base linear layer.

        This is useful before exporting a standalone merged model.
        """
        if self.merged:
            return

        delta = self.delta_weight()

        self.base.weight.add_(
            delta.to(
                dtype=self.base.weight.dtype,
                device=self.base.weight.device,
            )
        )

        self.merged = True

    @torch.no_grad()
    def unmerge(
        self,
    ) -> None:
        """
        Remove a previously merged LoRA update from the base weight.
        """
        if not self.merged:
            return

        delta = self.delta_weight()

        self.base.weight.sub_(
            delta.to(
                dtype=self.base.weight.dtype,
                device=self.base.weight.device,
            )
        )

        self.merged = False

    # =========================================================================
    # LoRA delta
    # =========================================================================

    def delta_weight(
        self,
    ) -> torch.Tensor:
        """
        Return the effective LoRA weight update.

        For Linear layers:

            delta_W = scaling * B @ A
        """
        return (
            self.lora_B.weight
            @ self.lora_A.weight
        ) * self.scaling

    # =========================================================================
    # Parameter helpers
    # =========================================================================

    def lora_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        """Yield only trainable LoRA parameters."""
        yield self.lora_A.weight
        yield self.lora_B.weight

    def lora_parameter_count(
        self,
    ) -> int:
        """Return the number of LoRA parameters."""
        return sum(
            parameter.numel()
            for parameter in self.lora_parameters()
        )

    def base_parameter_count(
        self,
    ) -> int:
        """Return the number of parameters in the wrapped base layer."""
        return sum(
            parameter.numel()
            for parameter in self.base.parameters()
        )

    def extra_repr(
        self,
    ) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, "
            f"r={self.r}, "
            f"alpha={self.alpha:g}, "
            f"scaling={self.scaling:g}, "
            f"dropout={self.dropout_p:g}, "
            f"enabled={self.enabled}, "
            f"merged={self.merged}"
        )


# =============================================================================
# Module traversal
# =============================================================================


def _matches_target(
    name: str,
    targets: Sequence[str],
) -> bool:
    """
    Determine whether a module name matches one of the requested LoRA targets.

    Supports:

        q_proj
        model.layers.0.self_attn.q_proj

    without accidentally matching unrelated suffixes.
    """
    if not name:
        return False

    for target in targets:
        target = str(
            target
        ).strip()

        if not target:
            continue

        if (
            name == target
            or name.endswith(
                "." + target
            )
        ):
            return True

    return False


def _get_parent_module(
    model: nn.Module,
    module_name: str,
) -> Tuple[nn.Module, str]:
    """
    Return the parent module and final attribute name for a dotted path.

    Example:

        model.layers.0.self_attn.q_proj

    returns:

        parent = model.layers.0.self_attn
        child  = q_proj
    """
    parent_name, _, child_name = (
        module_name.rpartition(".")
    )

    if not child_name:
        raise ValueError(
            f"Invalid module name: {module_name!r}."
        )

    if parent_name:
        parent = model.get_submodule(
            parent_name
        )
    else:
        parent = model

    return parent, child_name


# =============================================================================
# Injection
# =============================================================================


def inject(
    model: nn.Module,
    targets: Sequence[str],
    r: int,
    alpha: float,
    dropout: float = 0.0,
    *,
    strict: bool = False,
) -> int:
    """
    Inject LoRA into matching ``nn.Linear`` modules.

    Parameters
    ----------
    model:
        Model to modify in-place.

    targets:
        Module suffixes such as:

            ["q_proj", "v_proj"]

    r:
        LoRA rank.

    alpha:
        LoRA alpha.

    dropout:
        LoRA dropout.

    strict:
        If True, raise an error when a requested target is not found.

    Returns
    -------
    int
        Number of modules successfully replaced.

    Notes
    -----
    Injection happens in-place.
    """

    if not isinstance(
        model,
        nn.Module,
    ):
        raise TypeError(
            "model must be a torch.nn.Module."
        )

    if isinstance(
        targets,
        str,
    ):
        targets = [
            targets
        ]

    normalized_targets = tuple(
        dict.fromkeys(
            str(target).strip()
            for target in targets
            if str(target).strip()
        )
    )

    if not normalized_targets:
        raise ValueError(
            "targets must contain at least one non-empty module name."
        )

    # Validate before modifying the model.
    r = _validate_rank(
        r
    )

    alpha = _validate_alpha(
        alpha
    )

    dropout = _validate_dropout(
        dropout
    )

    # Snapshot named_modules because we are going to replace modules while
    # traversing the model.
    candidates = list(
        model.named_modules()
    )

    replaced = 0
    matched_targets = set()

    for name, module in candidates:

        if not _matches_target(
            name,
            normalized_targets,
        ):
            continue

        matched_targets.add(
            next(
                (
                    target
                    for target in normalized_targets
                    if name == target
                    or name.endswith(
                        "." + target
                    )
                ),
                name,
            )
        )

        # ---------------------------------------------------------------------
        # Already injected
        # ---------------------------------------------------------------------

        if isinstance(
            module,
            FastLoraLinear,
        ):
            continue

        # ---------------------------------------------------------------------
        # Only Linear layers are supported
        # ---------------------------------------------------------------------

        if not isinstance(
            module,
            nn.Linear,
        ):
            continue

        parent, child_name = (
            _get_parent_module(
                model,
                name,
            )
        )

        wrapped = FastLoraLinear(
            base=module,
            r=r,
            alpha=alpha,
            dropout=dropout,
        )

        setattr(
            parent,
            child_name,
            wrapped,
        )

        replaced += 1

    if strict:
        missing = [
            target
            for target in normalized_targets
            if target not in matched_targets
        ]

        if missing:
            raise ValueError(
                "The following LoRA targets were not found: "
                + ", ".join(missing)
            )

    return replaced


# =============================================================================
# Model-level utilities
# =============================================================================


def count_lora_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable LoRA parameters in a model.
    """
    total = 0

    for module in model.modules():
        if isinstance(
            module,
            FastLoraLinear,
        ):
            total += module.lora_parameter_count()

    return total


def mark_only_lora_trainable(
    model: nn.Module,
) -> int:
    """
    Freeze everything except FTRAIN LoRA parameters.

    Returns
    -------
    int
        Number of trainable parameters after modification.
    """
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    total = 0

    for module in model.modules():
        if isinstance(
            module,
            FastLoraLinear,
        ):
            for parameter in module.lora_parameters():
                parameter.requires_grad_(True)
                total += parameter.numel()

    return total


# =============================================================================
# Merge utilities
# =============================================================================


@torch.no_grad()
def merge_lora(
    model: nn.Module,
) -> int:
    """
    Merge every FastLoraLinear module in the model.

    Returns the number of merged LoRA layers.
    """
    count = 0

    for module in model.modules():
        if isinstance(
            module,
            FastLoraLinear,
        ):
            module.merge()
            count += 1

    return count


@torch.no_grad()
def unmerge_lora(
    model: nn.Module,
) -> int:
    """
    Unmerge every previously merged FastLoraLinear module.

    Returns the number of unmerged layers.
    """
    count = 0

    for module in model.modules():
        if isinstance(
            module,
            FastLoraLinear,
        ):
            module.unmerge()
            count += 1

    return count


def get_lora_state(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA tensors.

    This is useful when saving a lightweight adapter checkpoint.
    """
    state: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():
        if not isinstance(
            module,
            FastLoraLinear,
        ):
            continue

        state[
            f"{name}.lora_A.weight"
        ] = module.lora_A.weight.detach().clone()

        state[
            f"{name}.lora_B.weight"
        ] = module.lora_B.weight.detach().clone()

    return state


def lora_summary(
    model: nn.Module,
) -> Dict[str, Any]:
    """
    Return a compact LoRA model summary.
    """
    modules: List[str] = []

    total_lora = 0
    total_base = 0

    for name, module in model.named_modules():
        if not isinstance(
            module,
            FastLoraLinear,
        ):
            continue

        modules.append(
            name
        )

        total_lora += (
            module.lora_parameter_count()
        )

        total_base += (
            module.base_parameter_count()
        )

    return {
        "lora_modules": modules,
        "num_lora_modules": len(modules),
        "lora_parameters": total_lora,
        "wrapped_base_parameters": total_base,
    }
