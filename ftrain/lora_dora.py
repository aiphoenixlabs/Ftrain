"""
FTRAIN DoRA
===========

Weight-Decomposed Low-Rank Adaptation (DoRA) for FTRAIN.

DoRA decomposes the adapted weight into:

    W' = magnitude * direction(W + LoRA(W))

For a linear layer, the effective weight is conceptually:

    W_eff = ||W|| * normalize(W + ΔW)

where:

    ΔW = (alpha / r) * B @ A

This implementation is designed for transformer models and supports:

• Arbitrary input dimensions: [features], [batch, features],
  [batch, sequence, features], etc.
• Frozen base weights.
• Trainable LoRA A/B.
• Trainable per-output magnitude.
• Optional dropout.
• Triton fused normalization when available and safe.
• Automatic PyTorch fallback.
• Enable/disable without destroying parameters.
• Safe recursive module injection.
• Duplicate-injection protection.
• Diagnostics and parameter counting.
• Robust dtype/device handling.
• Compatibility with normal PyTorch optimizers.

Important
---------
DoRA is NOT equivalent to ordinary LoRA.

The magnitude vector changes the effective scale of every output feature,
while the LoRA branch primarily learns the direction/update.

Because of this, DoRA cannot generally be merged into the original Linear
weight using the simple:

    W += B @ A

operation used by ordinary LoRA.

A proper DoRA export requires constructing the effective weight after the
adapter has been trained.
"""

from __future__ import annotations

import logging
import math
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Sequence,
    Tuple,
)

import torch
import torch.nn as nn

from .kernels_dora import DoraFusedFunction

logger = logging.getLogger(__name__)

__all__ = [
    "DoRALinear",
    "inject_dora",
    "count_dora_parameters",
    "mark_only_dora_trainable",
    "get_dora_state",
    "dora_summary",
]


# =============================================================================
# Validation helpers
# =============================================================================


def _validate_rank(r: int) -> int:
    """Validate and normalize the DoRA rank."""
    if isinstance(r, bool):
        raise TypeError(
            "DoRA rank 'r' must be an integer, not bool."
        )

    try:
        rank = int(r)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"DoRA rank must be an integer, got {r!r}."
        ) from exc

    if rank <= 0:
        raise ValueError(
            f"DoRA rank must be greater than zero, got {rank}."
        )

    return rank


def _validate_alpha(alpha: float) -> float:
    """Validate DoRA alpha."""
    try:
        value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"DoRA alpha must be numeric, got {alpha!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            "DoRA alpha must be finite."
        )

    if value <= 0:
        raise ValueError(
            f"DoRA alpha must be greater than zero, got {value}."
        )

    return value


def _validate_dropout(dropout: float) -> float:
    """Validate dropout probability."""
    try:
        value = float(dropout)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"DoRA dropout must be numeric, got {dropout!r}."
        ) from exc

    if not math.isfinite(value):
        raise ValueError(
            "DoRA dropout must be finite."
        )

    if not 0.0 <= value < 1.0:
        raise ValueError(
            "DoRA dropout must satisfy 0 <= dropout < 1."
        )

    return value


# =============================================================================
# DoRA Linear
# =============================================================================


class DoRALinear(nn.Module):
    """
    DoRA wrapper for ``torch.nn.Linear``.

    Parameters
    ----------
    base:
        Existing linear layer.

    r:
        LoRA rank.

    alpha:
        LoRA scaling factor.

    dropout:
        Dropout probability applied to the LoRA input branch.

    Notes
    -----
    The base Linear weights are frozen.

    The trainable parameters are:

        lora_A
        lora_B
        magnitude

    The initial effective output is designed to closely reproduce the
    original base layer.
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
                "DoRALinear requires an nn.Linear base layer."
            )

        self.r = _validate_rank(r)
        self.alpha = _validate_alpha(alpha)
        self.dropout_p = _validate_dropout(dropout)

        self.base = base

        # ---------------------------------------------------------------------
        # Freeze the base model
        # ---------------------------------------------------------------------

        for parameter in self.base.parameters():
            parameter.requires_grad_(False)

        # ---------------------------------------------------------------------
        # Device / dtype
        # ---------------------------------------------------------------------

        device = base.weight.device
        dtype = base.weight.dtype

        if not dtype.is_floating_point:
            raise TypeError(
                "DoRA requires a floating-point Linear weight dtype. "
                f"Got {dtype}."
            )

        # ---------------------------------------------------------------------
        # LoRA matrices
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

        # Standard LoRA initialization.
        nn.init.kaiming_uniform_(
            self.lora_A.weight,
            a=math.sqrt(5),
        )

        # B starts at zero, so the LoRA delta initially contributes nothing.
        nn.init.zeros_(
            self.lora_B.weight
        )

        # ---------------------------------------------------------------------
        # DoRA magnitude
        # ---------------------------------------------------------------------

        # A Linear weight has shape:
        #
        #     [out_features, in_features]
        #
        # Therefore every output neuron receives one magnitude value.
        #
        # Keep magnitude in FP32 for numerical stability.
        initial_magnitude = (
            base.weight.detach()
            .float()
            .norm(
                p=2,
                dim=1,
            )
            .clamp_min(1e-8)
        )

        self.magnitude = nn.Parameter(
            initial_magnitude
        )

        # ---------------------------------------------------------------------
        # Scaling
        # ---------------------------------------------------------------------

        self.scaling = (
            self.alpha
            / float(self.r)
        )

        # Compatibility with the old implementation.
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
        # Runtime flags
        # ---------------------------------------------------------------------

        self.enabled = True

        # DoRA does not support a naïve in-place merge like LoRA.
        self.merged = False

        # Triton is an optimization, not a correctness requirement.
        self._triton_available = (
            hasattr(
                DoraFusedFunction,
                "apply",
            )
        )

    # =========================================================================
    # LoRA delta
    # =========================================================================

    def delta_weight(
        self,
    ) -> torch.Tensor:
        """
        Return the LoRA weight update:

            ΔW = (alpha / r) * B @ A
        """
        return (
            self.lora_B.weight
            @ self.lora_A.weight
        ) * self.scaling

    # =========================================================================
    # Effective weight
    # =========================================================================

    def effective_weight(
        self,
    ) -> torch.Tensor:
        """
        Construct the effective DoRA weight.

        Conceptually:

            V = W + ΔW

            W_eff = normalize_rows(V) * magnitude[:, None]
        """

        # Compute normalization in FP32 for stability.
        combined = (
            self.base.weight.float()
            + self.delta_weight().float()
        )

        row_norm = (
            combined
            .norm(
                p=2,
                dim=1,
                keepdim=True,
            )
            .clamp_min(1e-8)
        )

        direction = (
            combined
            / row_norm
        )

        effective = (
            direction
            * self.magnitude.float().unsqueeze(1)
        )

        return effective.to(
            dtype=self.base.weight.dtype
        )

    # =========================================================================
    # Direction calculation
    # =========================================================================

    def _combined_output(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate:

            base(x) + LoRA(x)
        """

        base_output = self.base(
            x
        )

        if not self.enabled:
            return base_output

        # Dropout is applied only to the LoRA branch.
        lora_input = self.dropout(
            x
        )

        lora_output = self.lora_B(
            self.lora_A(
                lora_input
            )
        )

        lora_output = (
            lora_output
            * self.scaling
        )

        return (
            base_output
            + lora_output
        )

    # =========================================================================
    # Fused normalization
    # =========================================================================

    def _normalize_output(
        self,
        combined: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize the output vector and apply DoRA magnitude.

        Supports arbitrary leading dimensions.

        Example:

            [batch, seq, hidden]

        becomes conceptually:

            [batch * seq, hidden]

        for the fused kernel.
        """

        if combined.ndim == 0:
            raise ValueError(
                "DoRALinear received a scalar tensor."
            )

        if combined.shape[-1] != self.base.out_features:
            raise ValueError(
                "Invalid DoRA output shape. Expected last dimension "
                f"{self.base.out_features}, got "
                f"{combined.shape[-1]}."
            )

        # ---------------------------------------------------------------------
        # Triton path
        # ---------------------------------------------------------------------

        #
        # DoraFusedFunction in the current FTRAIN kernel operates on a 2D
        # [M, N] tensor. Transformer inputs, however, are usually 3D.
        #
        # Flatten all leading dimensions safely.
        #
        if (
            combined.is_cuda
            and combined.ndim >= 2
            and self._triton_available
        ):
            original_shape = combined.shape

            flat = combined.reshape(
                -1,
                combined.shape[-1],
            )

            try:
                normalized = DoraFusedFunction.apply(
                    flat,
                    self.magnitude,
                )

                return normalized.reshape(
                    original_shape
                )

            except Exception as exc:
                # Never allow an optional performance kernel to destroy
                # training correctness.
                logger.debug(
                    "DoRA Triton kernel failed; "
                    "falling back to PyTorch: %s",
                    exc,
                    exc_info=True,
                )

        # ---------------------------------------------------------------------
        # Stable PyTorch fallback
        # ---------------------------------------------------------------------

        combined_f32 = combined.float()

        norm = (
            combined_f32
            .norm(
                p=2,
                dim=-1,
                keepdim=True,
            )
            .clamp_min(1e-8)
        )

        normalized = (
            combined_f32
            / norm
        )

        magnitude = (
            self.magnitude
            .float()
        )

        return (
            normalized
            * magnitude
        ).to(
            dtype=combined.dtype
        )

    # =========================================================================
    # Forward
    # =========================================================================

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Training and evaluation use the same mathematical DoRA operation.

        The only difference is dropout behavior, which is controlled by
        ``self.training`` through ``nn.Dropout``.
        """

        if self.merged:
            # A DoRA layer should normally not be marked merged because the
            # effective weight depends on the magnitude vector.
            #
            # Keep this branch defensive rather than silently producing an
            # incorrect result.
            return self.base(
                x
            )

        combined = self._combined_output(
            x
        )

        if not self.enabled:
            return combined

        return self._normalize_output(
            combined
        )

    # =========================================================================
    # Enable / disable
    # =========================================================================

    def enable_dora(
        self,
    ) -> None:
        """Enable the DoRA adaptation."""
        self.enabled = True

    def disable_dora(
        self,
    ) -> None:
        """
        Disable DoRA and return to the frozen base Linear behavior.
        """
        self.enabled = False

    # Compatibility aliases.
    enable_lora = enable_dora
    disable_lora = disable_dora

    # =========================================================================
    # Parameters
    # =========================================================================

    def dora_parameters(
        self,
    ) -> Iterable[nn.Parameter]:
        """Yield only trainable DoRA parameters."""
        yield self.lora_A.weight
        yield self.lora_B.weight
        yield self.magnitude

    def dora_parameter_count(
        self,
    ) -> int:
        """Return the number of trainable DoRA parameters."""
        return sum(
            parameter.numel()
            for parameter in self.dora_parameters()
        )

    def lora_parameter_count(
        self,
    ) -> int:
        """
        Compatibility helper.

        Includes A and B but excludes magnitude.
        """
        return (
            self.lora_A.weight.numel()
            + self.lora_B.weight.numel()
        )

    # =========================================================================
    # Export
    # =========================================================================

    @torch.no_grad()
    def materialize_weight(
        self,
    ) -> torch.Tensor:
        """
        Return the final effective DoRA weight.

        This is the correct weight to use when exporting a standalone model.

        Unlike ordinary LoRA, this cannot generally be produced by simply
        adding B @ A to the original weight.
        """
        return self.effective_weight()

    @torch.no_grad()
    def materialize_bias(
        self,
    ) -> Any:
        """
        Return the original bias.

        DoRA does not modify the bias.
        """
        if self.base.bias is None:
            return None

        return self.base.bias.detach().clone()

    # =========================================================================
    # Representation
    # =========================================================================

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
# Target matching
# =============================================================================


def _matches_target(
    name: str,
    targets: Sequence[str],
) -> bool:
    """Return True when a module name matches a requested target."""
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
    """Resolve a dotted module path into parent + child name."""
    parent_name, _, child_name = (
        module_name.rpartition(".")
    )

    if not child_name:
        raise ValueError(
            f"Invalid module path: {module_name!r}."
        )

    parent = (
        model.get_submodule(parent_name)
        if parent_name
        else model
    )

    return parent, child_name


# =============================================================================
# Injection
# =============================================================================


def inject_dora(
    model: nn.Module,
    targets: Sequence[str],
    r: int,
    alpha: float,
    dropout: float = 0.0,
    *,
    strict: bool = False,
) -> int:
    """
    Inject DoRA into matching Linear modules.

    Parameters
    ----------
    model:
        PyTorch model to modify in-place.

    targets:
        Target module names, e.g.:

            [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ]

    r:
        DoRA rank.

    alpha:
        DoRA alpha.

    dropout:
        LoRA branch dropout.

    strict:
        Raise if a requested target was not found.

    Returns
    -------
    int
        Number of injected modules.
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
            "targets must contain at least one valid module name."
        )

    r = _validate_rank(
        r
    )

    alpha = _validate_alpha(
        alpha
    )

    dropout = _validate_dropout(
        dropout
    )

    # Snapshot BEFORE modification.
    candidates = list(
        model.named_modules()
    )

    replaced = 0
    matched = set()

    for name, module in candidates:

        if not _matches_target(
            name,
            normalized_targets,
        ):
            continue

        matched_target = next(
            (
                target
                for target in normalized_targets
                if (
                    name == target
                    or name.endswith(
                        "." + target
                    )
                )
            ),
            None,
        )

        if matched_target is not None:
            matched.add(
                matched_target
            )

        # Already wrapped.
        if isinstance(
            module,
            DoRALinear,
        ):
            continue

        # DoRA implementation targets Linear modules.
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

        wrapped = DoRALinear(
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
            if target not in matched
        ]

        if missing:
            raise ValueError(
                "The following DoRA targets were not found: "
                + ", ".join(missing)
            )

    return replaced


# =============================================================================
# Model utilities
# =============================================================================


def count_dora_parameters(
    model: nn.Module,
) -> int:
    """Count all DoRA trainable parameters."""
    total = 0

    for module in model.modules():
        if isinstance(
            module,
            DoRALinear,
        ):
            total += (
                module.dora_parameter_count()
            )

    return total


def mark_only_dora_trainable(
    model: nn.Module,
) -> int:
    """
    Freeze the entire model and enable only DoRA parameters.

    Returns
    -------
    int
        Number of trainable parameters.
    """

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    total = 0

    for module in model.modules():
        if isinstance(
            module,
            DoRALinear,
        ):
            for parameter in module.dora_parameters():
                parameter.requires_grad_(True)
                total += parameter.numel()

    return total


# =============================================================================
# State extraction
# =============================================================================


def get_dora_state(
    model: nn.Module,
) -> Dict[str, torch.Tensor]:
    """
    Extract only DoRA adapter parameters.

    The returned dictionary contains:

        lora_A
        lora_B
        magnitude
    """

    state: Dict[str, torch.Tensor] = {}

    for name, module in model.named_modules():

        if not isinstance(
            module,
            DoRALinear,
        ):
            continue

        state[
            f"{name}.lora_A.weight"
        ] = (
            module.lora_A.weight
            .detach()
            .clone()
        )

        state[
            f"{name}.lora_B.weight"
        ] = (
            module.lora_B.weight
            .detach()
            .clone()
        )

        state[
            f"{name}.magnitude"
        ] = (
            module.magnitude
            .detach()
            .clone()
        )

    return state


# =============================================================================
# Diagnostics
# =============================================================================


def dora_summary(
    model: nn.Module,
) -> Dict[str, Any]:
    """
    Return information about DoRA modules in a model.
    """

    modules: List[str] = []

    total_dora = 0
    total_lora = 0
    total_magnitude = 0

    for name, module in model.named_modules():

        if not isinstance(
            module,
            DoRALinear,
        ):
            continue

        modules.append(
            name
        )

        total_dora += (
            module.dora_parameter_count()
        )

        total_lora += (
            module.lora_parameter_count()
        )

        total_magnitude += (
            module.magnitude.numel()
        )

    return {
        "dora_modules": modules,
        "num_dora_modules": len(modules),
        "dora_parameters": total_dora,
        "lora_parameters": total_lora,
        "magnitude_parameters": total_magnitude,
    }


# =============================================================================
# Materialization / export helpers
# =============================================================================


@torch.no_grad()
def materialize_dora_model(
    model: nn.Module,
) -> int:
    """
    Replace DoRALinear modules with ordinary nn.Linear layers containing
    their effective DoRA weights.

    WARNING
    -------
    This modifies the model structure in-place.

    Use this when you want a standalone model without FTRAIN DoRA wrappers.
    """

    candidates = list(
        model.named_modules()
    )

    converted = 0

    for name, module in candidates:

        if not isinstance(
            module,
            DoRALinear,
        ):
            continue

        parent, child_name = (
            _get_parent_module(
                model,
                name,
            )
        )

        base = module.base

        new_layer = nn.Linear(
            base.in_features,
            base.out_features,
            bias=base.bias is not None,
            device=base.weight.device,
            dtype=base.weight.dtype,
        )

        effective_weight = (
            module.materialize_weight()
        )

        new_layer.weight.copy_(
            effective_weight
        )

        if (
            base.bias is not None
            and new_layer.bias is not None
        ):
            new_layer.bias.copy_(
                base.bias
            )

        setattr(
            parent,
            child_name,
            new_layer,
        )

        converted += 1

    return converted
