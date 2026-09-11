"""
FTRAIN Distributed Training Support
===================================

Makes N GPUs behave like one fast GPU for the custom training loop:

- process-group init (NCCL on CUDA, GLOO on CPU) triggered purely by the
  ``WORLD_SIZE`` environment variable — the only thing a user must do is
  launch FTRAIN under torchrun, e.g.::

      torchrun --nproc_per_node=4 --rdzv-backend=c10d \\
               --rdzv-endpoint=localhost:29517 train_script.py

  The ``c10d`` rendezvous backend is recommended on Windows builds without
  libuv (the ``static`` backend hardcodes it and crashes); FTRAIN also
  defaults ``USE_LIBUV=0`` on Windows hosts;
- DDP model wrapping AFTER LoRA/DoRA injection (only trainable adapters
  are synced, so ``find_unused_parameters=False`` stays safe and fast);
- ``DistributedSampler`` sharding with epoch reseeding;
- ``no_sync()`` during gradient-accumulation micro-steps (the difference
  between DDP that scales and DDP that wastes N× the bandwidth);
- cross-rank metric averaging, rank-0-only printing/checkpointing, and
  broadcast of Captain LR decisions so every rank stays bit-identical.

Everything degrades to a no-op when ``WORLD_SIZE <= 1``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

__all__ = [
    "DDPInfo",
    "init_distributed_if_needed",
    "destroy_distributed",
    "wrap_ddp",
    "all_reduce_mean",
    "broadcast_float",
    "broadcast_object",
    "is_main_process",
]


@dataclass
class DDPInfo:
    """Distributed context for this process."""

    enabled: bool
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    backend: str = "none"

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def summary(self) -> str:
        if not self.enabled:
            return "single process"
        return (
            f"DDP rank {self.rank}/{self.world_size} "
            f"(local {self.local_rank}, backend {self.backend})"
        )


def _pick_backend() -> str:
    if torch.cuda.is_available():
        return "nccl"
    return "gloo"


def init_distributed_if_needed() -> DDPInfo:
    """
    Initialize the process group when launched under torchrun/Accelerate.

    Detects real multi-process launches (``WORLD_SIZE > 1``) and joins the
    group. Single-process runs are untouched — this function is the reason
    nothing about FTRAIN changes for one GPU.
    """
    try:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    except (TypeError, ValueError):
        world_size = 1

    if world_size <= 1:
        return DDPInfo(enabled=False)

    # Windows torch builds without libuv crash in TCPStore rendezvous.
    # Opting out is harmless elsewhere and must happen before any store
    # is created (torchrun parent + workers).
    if os.name == "nt":
        os.environ.setdefault("USE_LIBUV", "0")

    if dist.is_available() and dist.is_initialized():
        return DDPInfo(
            enabled=True,
            rank=dist.get_rank(),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=dist.get_world_size(),
            backend=dist.get_backend(),
        )

    backend = _pick_backend()
    try:
        rank = int(os.environ.get("RANK", "0"))
        world = int(os.environ.get("WORLD_SIZE", "1"))

        filestore_dir = os.environ.get("FTRAIN_FILESTORE_DIR")
        if filestore_dir:
            # Windows torch builds without libuv cannot create a TCPStore
            # (both 'static' and 'c10d' rendezvous crash). A FileStore
            # rendezvous needs neither TCP nor libuv, so FTRAIN supports it
            # as an explicit opt-in for those machines.
            os.makedirs(filestore_dir, exist_ok=True)
            dist.init_process_group(
                backend=backend,
                store=dist.FileStore(
                    os.path.join(filestore_dir, "ftrain_store"),
                    world,
                ),
                rank=rank,
                world_size=world,
            )
        elif not (dist.is_available() and dist.is_initialized()):
            dist.init_process_group(backend=backend)

        return DDPInfo(
            enabled=True,
            rank=dist.get_rank(),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=dist.get_world_size(),
            backend=dist.get_backend(),
        )
    except Exception:
        logger.warning(
            "FTRAIN: WORLD_SIZE>1 detected but the process group failed to "
            "initialize; continuing single-process.",
            exc_info=True,
        )
        return DDPInfo(enabled=False)


def destroy_distributed() -> None:
    """Tear the process group down cleanly (best-effort)."""
    try:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        logger.debug("Process-group teardown failed.", exc_info=True)


def wrap_ddp(model: torch.nn.Module, info: DDPInfo) -> torch.nn.Module:
    """Wrap a model in DDP on CUDA (NCCL) or CPU (GLOO) ranks.

    ``find_unused_parameters=False`` is deliberate: FTRAIN freezes the base
    model and trains LoRA/DoRA adapters, so every trainable parameter
    receives gradients and the static graph is both faster and safer.
    MPS/DirectML/XPU have no gradient-sync backend, so those ranks run
    independent single-process training (logged loudly).
    """
    if not info.enabled or model is None:
        return model

    device = next(model.parameters()).device

    if device.type not in ("cuda", "cpu"):
        logger.warning(
            "FTRAIN: multi-process launch detected on %s; gradient sync "
            "has no backend there — running independent single-process "
            "training instead.",
            device,
        )
        return model

    try:
        ddp_kwargs: Dict[str, Any] = {
            "find_unused_parameters": False,
            "broadcast_buffers": False,
        }
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [info.local_rank]
            ddp_kwargs["output_device"] = info.local_rank

        wrapped = torch.nn.parallel.DistributedDataParallel(
            model,
            **ddp_kwargs,
        )
        logger.info("FTRAIN: DDP wrap complete (%s).", info.summary())
        return wrapped
    except Exception:
        logger.warning(
            "FTRAIN: DDP wrap failed; continuing single-process.",
            exc_info=True,
        )
        return model


def unwrap_ddp(model: Any) -> Any:
    """Return the underlying module from a DDP wrapper, if wrapped."""
    return getattr(model, "module", model)


def all_reduce_mean(value: float, info: DDPInfo) -> float:
    """Average a scalar across all ranks (metrics, validation loss)."""
    if not info.enabled:
        return value

    try:
        tensor = torch.tensor([float(value)], dtype=torch.float64)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return float(tensor.item()) / max(1, info.world_size)
    except Exception:
        logger.debug("all_reduce_mean failed; using local value.", exc_info=True)
        return value


def broadcast_float(value: float, info: DDPInfo, source: int = 0) -> float:
    """Rank ``source`` decides a float (e.g. Captain LR multiplier); all agree."""
    if not info.enabled:
        return value

    try:
        tensor = torch.tensor([float(value)], dtype=torch.float64)
        dist.broadcast(tensor, src=source)
        return float(tensor.item())
    except Exception:
        logger.debug("broadcast_float failed; keeping local value.", exc_info=True)
        return value


def broadcast_object(value: Any, info: DDPInfo, source: int = 0) -> Any:
    """Broadcast a picklable object (checkpoint paths, advice dicts)."""
    if not info.enabled:
        return value

    try:
        container = [value if info.rank == source else None]
        dist.broadcast_object_list(container, src=source)
        return container[0]
    except Exception:
        logger.debug("broadcast_object failed; keeping local value.", exc_info=True)
        return value


def is_main_process(info: Optional[DDPInfo]) -> bool:
    """Should this process print/save? (True for all single-process runs.)"""
    if info is None or not info.enabled:
        return True
    return info.is_main
