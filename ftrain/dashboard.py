```python id="f3pd24"
"""
FTRAIN Live Training Dashboard
==============================

Thread-safe lightweight monitoring dashboard for FTRAIN.

The dashboard is intentionally non-blocking from the training process:

    training thread
          │
          ▼
    log_metric(...)
          │
          ▼
    thread-safe queue
          │
          ▼
    Gradio dashboard thread
          │
          ▼
    live loss / validation loss / learning-rate plots

Design goals
------------
• Never block model training on dashboard rendering.
• Never let a dashboard failure terminate training.
• Handle concurrent metric producers safely.
• Bound memory growth for very long runs.
• Gracefully deal with missing/older Gradio versions.
• Reject malformed metrics without crashing.
• Expose dashboard state for higher-level FTRAIN code.
• Keep the existing public interface compatible.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from collections import deque
from typing import Any, Dict, List, Mapping, Optional

logger = logging.getLogger(__name__)

__all__ = ["TrainingDashboard"]


# =============================================================================
# Helpers
# =============================================================================


def _finite_float(
    value: Any,
    default: float = float("nan"),
) -> float:
    """Convert a value into a finite float, otherwise return default."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if math.isfinite(result):
        return result

    return default


def _safe_step(
    value: Any,
) -> Optional[int]:
    """Normalize a training step."""
    try:
        step = int(value)
    except (TypeError, ValueError):
        return None

    if step < 0:
        return None

    return step


# =============================================================================
# Dashboard
# =============================================================================


class TrainingDashboard:
    """
    Live FTRAIN dashboard.

    Parameters
    ----------
    port:
        TCP port used by Gradio.

    max_points:
        Maximum number of historical points retained in memory. This prevents
        a multi-day training job from growing the Python process indefinitely.

    refresh_interval:
        Dashboard refresh period in seconds.

    queue_maxsize:
        Maximum number of pending metric updates. When training produces data
        faster than the UI can consume it, old queued metrics are coalesced
        rather than allowing unbounded memory growth.
    """

    def __init__(
        self,
        port: int = 7860,
        *,
        max_points: int = 10_000,
        refresh_interval: float = 2.0,
        queue_maxsize: int = 2_000,
    ) -> None:
        if isinstance(port, bool):
            raise TypeError("port must be an integer.")

        try:
            port = int(port)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"port must be an integer, got {port!r}."
            ) from exc

        if not 1 <= port <= 65_535:
            raise ValueError(
                f"port must be between 1 and 65535, got {port}."
            )

        try:
            max_points = int(max_points)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"max_points must be an integer, got {max_points!r}."
            ) from exc

        if max_points <= 0:
            raise ValueError(
                "max_points must be greater than zero."
            )

        refresh_interval = _finite_float(
            refresh_interval,
            default=2.0,
        )

        if refresh_interval <= 0:
            raise ValueError(
                "refresh_interval must be greater than zero."
            )

        try:
            queue_maxsize = int(queue_maxsize)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "queue_maxsize must be an integer."
            ) from exc

        if queue_maxsize <= 0:
            raise ValueError(
                "queue_maxsize must be greater than zero."
            )

        self.port = port
        self.max_points = max_points
        self.refresh_interval = refresh_interval
        self.queue_maxsize = queue_maxsize

        # ``queue.Queue`` is thread-safe, unlike manipulating a normal list
        # from multiple threads.
        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(
            maxsize=queue_maxsize
        )

        # Bounded history prevents long training jobs from growing memory
        # without limit.
        self.history: Dict[str, deque] = {
            "step": deque(maxlen=max_points),
            "loss": deque(maxlen=max_points),
            "lr": deque(maxlen=max_points),
            "val_loss": deque(maxlen=max_points),
        }

        self._history_lock = threading.RLock()

        # Dashboard lifecycle.
        self.running = False
        self.started = False
        self.failed = False

        self._stop_event = threading.Event()
        self._state_lock = threading.RLock()

        self._demo: Any = None
        self._server_thread: Optional[threading.Thread] = None
        self._launch_error: Optional[BaseException] = None

        self._last_step: Optional[int] = None
        self._dropped_metrics = 0
        self._received_metrics = 0

        self.url: Optional[str] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> bool:
        """
        Start the Gradio dashboard.

        Returns
        -------
        bool
            True when launch was successfully initiated, False when Gradio is
            unavailable or launch failed.

        Notes
        -----
        This method is safe to call repeatedly. Repeated calls do not start
        multiple dashboard instances.
        """
        with self._state_lock:
            if self.running:
                logger.debug(
                    "FTRAIN dashboard is already running."
                )
                return True

            if self.started and not self.failed:
                return True

            self._stop_event.clear()
            self.failed = False
            self._launch_error = None

        try:
            import gradio as gr
        except ImportError:
            with self._state_lock:
                self.failed = True
                self.started = False
                self._launch_error = ImportError(
                    "Gradio is not installed."
                )

            logger.warning(
                "FTRAIN dashboard disabled: Gradio is not installed."
            )
            return False

        with self._state_lock:
            self.running = True
            self.started = True

        # Launch directly in a daemon thread so training never blocks on
        # Gradio's server startup or event loop.
        self._server_thread = threading.Thread(
            target=self._launch,
            args=(gr,),
            name="ftrain-dashboard",
            daemon=True,
        )

        self._server_thread.start()

        return True

    def _launch(
        self,
        gr: Any,
    ) -> None:
        """
        Build and launch the Gradio application.

        Any server failure is isolated from the training thread.
        """
        try:
            with gr.Blocks(
                title="FTRAIN Live Dashboard",
                theme=gr.themes.Monochrome(),
            ) as demo:
                gr.Markdown(
                    """
# 🔥 FTRAIN Live Dashboard

Real-time training telemetry from the FTRAIN engine.
"""
                )

                with gr.Row():
                    loss_plot = gr.LinePlot(
                        x="step",
                        y="loss",
                        title="Training Loss",
                        height=320,
                    )

                    val_plot = gr.LinePlot(
                        x="step",
                        y="val_loss",
                        title="Validation Loss",
                        height=320,
                    )

                lr_plot = gr.LinePlot(
                    x="step",
                    y="lr",
                    title="Learning Rate",
                    height=260,
                )

                # Small status display is useful when debugging whether the
                # dashboard is receiving fresh telemetry.
                status = gr.Markdown(
                    self._status_text()
                )

                outputs = [
                    loss_plot,
                    val_plot,
                    lr_plot,
                    status,
                ]

                # Newer Gradio versions expose ``every`` differently from
                # older versions, so support both patterns conservatively.
                try:
                    demo.load(
                        self._fetch_dashboard,
                        outputs=outputs,
                        every=self.refresh_interval,
                    )
                except TypeError:
                    # Older versions may not accept ``every`` on load.
                    demo.load(
                        self._fetch_dashboard,
                        outputs=outputs,
                    )

                self._demo = demo

                launch_kwargs: Dict[str, Any] = {
                    "server_port": self.port,
                    "share": False,
                    "prevent_thread_lock": True,
                    "quiet": True,
                }

                # ``show_error`` is not supported by every Gradio version.
                try:
                    demo.launch(
                        **launch_kwargs,
                        show_error=True,
                    )
                except TypeError:
                    demo.launch(
                        **launch_kwargs,
                    )

            logger.info(
                "FTRAIN dashboard launched on port %d.",
                self.port,
            )

        except Exception as exc:
            with self._state_lock:
                self.failed = True
                self.running = False
                self._launch_error = exc

            logger.exception(
                "FTRAIN dashboard failed to launch."
            )

    def stop(self) -> None:
        """
        Request dashboard shutdown.

        Gradio's server lifecycle differs between versions. We therefore
        attempt a supported close method when available, but never raise a
        dashboard shutdown exception into the training loop.
        """
        with self._state_lock:
            self.running = False
            self._stop_event.set()

        demo = self._demo

        if demo is not None:
            try:
                close = getattr(
                    demo,
                    "close",
                    None,
                )

                if callable(close):
                    close()

            except Exception:
                logger.debug(
                    "FTRAIN dashboard close() failed.",
                    exc_info=True,
                )

        logger.info(
            "FTRAIN dashboard stopped."
        )

    # =========================================================================
    # Metrics
    # =========================================================================

    def log_metric(
        self,
        step: Any,
        loss: Any,
        lr: Any,
        val_loss: Any = None,
    ) -> bool:
        """
        Queue one training metric.

        Returns
        -------
        bool
            True if the metric was accepted, False if it was rejected or the
            dashboard is not active.
        """
        with self._state_lock:
            if not self.running:
                return False

        normalized_step = _safe_step(
            step
        )

        if normalized_step is None:
            logger.debug(
                "FTRAIN dashboard ignored invalid step=%r.",
                step,
            )
            return False

        metric = {
            "step": normalized_step,
            "loss": _finite_float(loss),
            "lr": _finite_float(lr),
            "val_loss": (
                float("nan")
                if val_loss is None
                else _finite_float(val_loss)
            ),
        }

        # We don't want every weird NaN to disappear silently, but we also
        # don't want one bad metric to crash the training thread.
        if not math.isfinite(metric["loss"]):
            logger.debug(
                "FTRAIN dashboard received non-finite loss at step %d.",
                normalized_step,
            )

        if not math.isfinite(metric["lr"]):
            logger.debug(
                "FTRAIN dashboard received non-finite LR at step %d.",
                normalized_step,
            )

        with self._state_lock:
            self._received_metrics += 1

        try:
            self.queue.put_nowait(
                metric
            )

            return True

        except queue.Full:
            # The UI can fall behind significantly during high-throughput
            # training. Drop the oldest item and preserve the newest one.
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass

            try:
                self.queue.put_nowait(
                    metric
                )
                with self._state_lock:
                    self._dropped_metrics += 1

                return True

            except queue.Full:
                with self._state_lock:
                    self._dropped_metrics += 1

                return False

    # =========================================================================
    # Data processing
    # =========================================================================

    def _drain_queue(self) -> int:
        """
        Consume all currently queued metrics.

        Important:
        Never use ``queue.empty()`` as a synchronization primitive. A consumer
        can observe empty=True while another thread is just about to enqueue.
        """
        drained = 0

        while True:
            try:
                metric = self.queue.get_nowait()
            except queue.Empty:
                break

            self._append_metric(
                metric
            )

            self.queue.task_done()
            drained += 1

        return drained

    def _append_metric(
        self,
        metric: Mapping[str, Any],
    ) -> None:
        """Validate and append one metric to bounded history."""
        step = _safe_step(
            metric.get("step")
        )

        if step is None:
            return

        loss = _finite_float(
            metric.get("loss")
        )

        lr = _finite_float(
            metric.get("lr")
        )

        val_loss = _finite_float(
            metric.get("val_loss")
        )

        with self._history_lock:
            # Ignore older/out-of-order updates. This prevents an asynchronous
            # producer from drawing the graph backward.
            if (
                self._last_step is not None
                and step < self._last_step
            ):
                logger.debug(
                    "Ignoring out-of-order dashboard metric: "
                    "step=%d < last=%d.",
                    step,
                    self._last_step,
                )
                return

            self.history["step"].append(
                step
            )

            self.history["loss"].append(
                loss
            )

            self.history["lr"].append(
                lr
            )

            self.history["val_loss"].append(
                val_loss
            )

            self._last_step = step

    def _build_dataframe(self):
        """
        Build a pandas DataFrame from a consistent snapshot.

        Pandas is imported lazily so importing FTRAIN does not require pandas
        unless dashboard functionality is actually used.
        """
        import pandas as pd

        with self._history_lock:
            data = {
                "step": list(
                    self.history["step"]
                ),
                "loss": list(
                    self.history["loss"]
                ),
                "lr": list(
                    self.history["lr"]
                ),
                "val_loss": list(
                    self.history["val_loss"]
                ),
            }

        if not data["step"]:
            return pd.DataFrame(
                {
                    "step": [0],
                    "loss": [float("nan")],
                    "lr": [float("nan")],
                    "val_loss": [float("nan")],
                }
            )

        return pd.DataFrame(
            data
        )

    # =========================================================================
    # Gradio callbacks
    # =========================================================================

    def _fetch_dashboard(self):
        """
        Return fresh plot data plus dashboard status.

        This method is intentionally defensive because it runs from Gradio's
        event/update thread rather than the training loop.
        """
        try:
            self._drain_queue()

            dataframe = self._build_dataframe()

            return (
                dataframe,
                dataframe,
                dataframe,
                self._status_text(),
            )

        except Exception as exc:
            logger.exception(
                "FTRAIN dashboard refresh failed."
            )

            return (
                self._build_dataframe(),
                self._build_dataframe(),
                self._build_dataframe(),
                f"⚠️ Dashboard refresh error: `{exc}`",
            )

    # Backward-compatible private alias matching the old implementation.
    def _fetch(self):
        return self._fetch_dashboard()

    # =========================================================================
    # Status / introspection
    # =========================================================================

    def _status_text(self) -> str:
        """Build a compact dashboard state message."""
        with self._state_lock:
            running = self.running
            failed = self.failed
            received = self._received_metrics
            dropped = self._dropped_metrics
            last_step = self._last_step

        if failed:
            state = "🔴 failed"
        elif running:
            state = "🟢 running"
        else:
            state = "⚪ stopped"

        step_text = (
            "N/A"
            if last_step is None
            else str(last_step)
        )

        return (
            f"**Status:** {state}  \n"
            f"**Latest step:** {step_text}  \n"
            f"**Metrics received:** {received}  \n"
            f"**Metrics dropped:** {dropped}"
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Return structured dashboard status for diagnostics/tests.
        """
        with self._state_lock:
            return {
                "running": self.running,
                "started": self.started,
                "failed": self.failed,
                "port": self.port,
                "url": self.url,
                "received_metrics": self._received_metrics,
                "dropped_metrics": self._dropped_metrics,
                "last_step": self._last_step,
                "queued_metrics": self.queue.qsize(),
                "history_points": len(
                    self.history["step"]
                ),
                "error": (
                    str(self._launch_error)
                    if self._launch_error is not None
                    else None
                ),
            }

    def clear_history(self) -> None:
        """Clear stored metric history without stopping the dashboard."""
        with self._history_lock:
            for values in self.history.values():
                values.clear()

            self._last_step = None

        # Also discard stale queued metrics.
        while True:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                break

        logger.debug(
            "FTRAIN dashboard history cleared."
        )
```
