"""
FTRAIN Live Training Dashboard
==============================

Thread-safe, fault-tolerant live monitoring dashboard for FTRAIN.

The dashboard is deliberately designed as a NON-CRITICAL subsystem:
if Gradio is missing, an old Gradio API is installed, a refresh fails, or the
dashboard cannot start, model training must continue normally.

Architecture
------------

    FTRAIN training loop
            │
            │ log_metric(...)
            ▼
    bounded thread-safe queue
            │
            ▼
    dashboard refresh callback
            │
            ▼
    bounded metric history
            │
            ├── Training Loss
            ├── Validation Loss
            └── Learning Rate

Design goals
------------

• Never block the training thread.
• Never allow dashboard failures to crash training.
• Thread-safe metric ingestion.
• Bounded memory usage.
• Safe handling of malformed/non-finite metrics.
• Protection against out-of-order updates.
• Protection against duplicate steps.
• Graceful Gradio compatibility handling.
• Graceful shutdown.
• Runtime diagnostics through get_status().
• Backward-compatible TrainingDashboard API.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
from collections import deque
from typing import Any, Deque, Dict, Mapping, Optional, Tuple

logger = logging.getLogger(__name__)

__all__ = ["TrainingDashboard"]


# =============================================================================
# Constants
# =============================================================================

_DEFAULT_PORT = 7860
_DEFAULT_MAX_POINTS = 10_000
_DEFAULT_REFRESH_INTERVAL = 2.0
_DEFAULT_QUEUE_SIZE = 2_000

_MIN_PORT = 1
_MAX_PORT = 65_535

# Smallest useful refresh interval. Extremely small intervals can create
# unnecessary Gradio/UI overhead and compete with training.
_MIN_REFRESH_INTERVAL = 0.1

# A metric is allowed to move forward by any amount, but never backward.
# Duplicate steps are replaced rather than appended repeatedly.
_EPSILON = 1e-12


# =============================================================================
# Helper functions
# =============================================================================


def _finite_float(
    value: Any,
    default: float = float("nan"),
) -> float:
    """
    Convert a value to float only when it is finite.

    Parameters
    ----------
    value:
        Value to convert.

    default:
        Value returned when conversion fails or the result is non-finite.
    """
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default

    if math.isfinite(result):
        return result

    return default


def _safe_step(value: Any) -> Optional[int]:
    """
    Normalize a training step.

    Rejects:
        • None
        • negative values
        • non-numeric strings
        • booleans

    Returns
    -------
    Optional[int]
        A valid non-negative integer step or None.
    """
    if isinstance(value, bool):
        return None

    try:
        step = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if step < 0:
        return None

    return step


def _safe_positive_int(
    value: Any,
    name: str,
) -> int:
    """Validate a positive integer configuration value."""
    if isinstance(value, bool):
        raise TypeError(
            f"{name} must be an integer, not bool."
        )

    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"{name} must be an integer, got {value!r}."
        ) from exc

    if result <= 0:
        raise ValueError(
            f"{name} must be greater than zero, got {result}."
        )

    return result


def _safe_port(value: Any) -> int:
    """Validate a TCP port."""
    if isinstance(value, bool):
        raise TypeError(
            "port must be an integer, not bool."
        )

    try:
        port = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError(
            f"port must be an integer, got {value!r}."
        ) from exc

    if not _MIN_PORT <= port <= _MAX_PORT:
        raise ValueError(
            f"port must be between {_MIN_PORT} and {_MAX_PORT}, "
            f"got {port}."
        )

    return port


def _safe_refresh_interval(value: Any) -> float:
    """Validate the dashboard refresh interval."""
    interval = _finite_float(
        value,
        default=float("nan"),
    )

    if not math.isfinite(interval):
        raise ValueError(
            f"refresh_interval must be a finite number, got {value!r}."
        )

    if interval < _MIN_REFRESH_INTERVAL:
        raise ValueError(
            f"refresh_interval must be >= {_MIN_REFRESH_INTERVAL} seconds."
        )

    return interval


# =============================================================================
# Training Dashboard
# =============================================================================


class TrainingDashboard:
    """
    Live FTRAIN training dashboard.

    Parameters
    ----------
    port:
        TCP port used by Gradio.

    max_points:
        Maximum number of historical metric points retained in memory.

    refresh_interval:
        Number of seconds between UI refreshes.

    queue_maxsize:
        Maximum number of metrics waiting to be consumed.

    Notes
    -----
    ``log_metric()`` is designed for use directly inside a training loop.
    It never waits for the dashboard/UI and therefore should not meaningfully
    affect training throughput.
    """

    def __init__(
        self,
        port: int = _DEFAULT_PORT,
        *,
        max_points: int = _DEFAULT_MAX_POINTS,
        refresh_interval: float = _DEFAULT_REFRESH_INTERVAL,
        queue_maxsize: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        self.port = _safe_port(port)
        self.max_points = _safe_positive_int(
            max_points,
            "max_points",
        )
        self.queue_maxsize = _safe_positive_int(
            queue_maxsize,
            "queue_maxsize",
        )
        self.refresh_interval = _safe_refresh_interval(
            refresh_interval,
        )

        # ---------------------------------------------------------------------
        # Metric queue
        # ---------------------------------------------------------------------

        self.queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(
            maxsize=self.queue_maxsize,
        )

        # ---------------------------------------------------------------------
        # Bounded history
        # ---------------------------------------------------------------------

        self.history: Dict[str, Deque[Any]] = {
            "step": deque(maxlen=self.max_points),
            "loss": deque(maxlen=self.max_points),
            "lr": deque(maxlen=self.max_points),
            "val_loss": deque(maxlen=self.max_points),
        }

        self._history_lock = threading.RLock()
        self._state_lock = threading.RLock()

        # ---------------------------------------------------------------------
        # Lifecycle
        # ---------------------------------------------------------------------

        self.running = False
        self.started = False
        self.failed = False

        self._stop_event = threading.Event()

        self._demo: Any = None
        self._server_thread: Optional[threading.Thread] = None

        self._launch_error: Optional[BaseException] = None

        # ---------------------------------------------------------------------
        # Runtime statistics
        # ---------------------------------------------------------------------

        self._last_step: Optional[int] = None
        self._received_metrics = 0
        self._accepted_metrics = 0
        self._dropped_metrics = 0
        self._invalid_metrics = 0
        self._duplicate_metrics = 0
        self._out_of_order_metrics = 0
        self._refresh_count = 0
        self._refresh_errors = 0

        # The actual URL may become available only after Gradio launches.
        self.url: Optional[str] = None

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def start(self) -> bool:
        """
        Start the dashboard asynchronously.

        Returns
        -------
        bool
            True if the dashboard launch was initiated successfully.

        Important
        ---------
        A True return value means that startup was accepted, not necessarily
        that the Gradio server is already listening. The actual state can be
        inspected using ``get_status()``.
        """
        with self._state_lock:
            # Already running.
            if self.running:
                logger.debug(
                    "FTRAIN dashboard is already running."
                )
                return True

            # A previous startup thread may still be alive.
            if (
                self._server_thread is not None
                and self._server_thread.is_alive()
            ):
                logger.debug(
                    "FTRAIN dashboard startup is already in progress."
                )
                return True

            self._stop_event.clear()

            self.failed = False
            self._launch_error = None
            self.url = None

        # Import Gradio lazily.
        #
        # This is important because importing FTRAIN itself should not require
        # a GUI dependency.
        try:
            import gradio as gr
        except ImportError as exc:
            with self._state_lock:
                self.running = False
                self.started = False
                self.failed = True
                self._launch_error = exc

            logger.warning(
                "FTRAIN dashboard unavailable because Gradio is not installed."
            )

            return False

        # Mark startup as initiated.
        with self._state_lock:
            self.started = True

        self._server_thread = threading.Thread(
            target=self._launch,
            args=(gr,),
            name="ftrain-dashboard",
            daemon=True,
        )

        self._server_thread.start()

        logger.info(
            "FTRAIN dashboard startup initiated on port %d.",
            self.port,
        )

        return True

    def _launch(self, gr: Any) -> None:
        """
        Construct and launch the Gradio application.

        This method runs outside the training thread.
        """
        demo = None

        try:
            # -----------------------------------------------------------------
            # Build UI
            # -----------------------------------------------------------------

            with gr.Blocks(
                title="FTRAIN Live Dashboard",
                theme=gr.themes.Monochrome(),
            ) as demo:
                gr.Markdown(
                    """
# 🔥 FTRAIN Live Dashboard

### Real-time training telemetry

Monitor training loss, validation loss, and learning rate while FTRAIN
is running.
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
                    height=280,
                )

                status = gr.Markdown(
                    self._status_text(),
                )

                outputs = [
                    loss_plot,
                    val_plot,
                    lr_plot,
                    status,
                ]

                # -------------------------------------------------------------
                # Refresh callback
                # -------------------------------------------------------------

                self._register_refresh(
                    demo=demo,
                    outputs=outputs,
                )

            self._demo = demo

            # If stop() was called while the UI was being constructed, do not
            # launch a server that the caller already asked us to stop.
            if self._stop_event.is_set():
                logger.debug(
                    "FTRAIN dashboard launch cancelled before Gradio startup."
                )
                return

            # -----------------------------------------------------------------
            # Launch Gradio
            # -----------------------------------------------------------------

            launch_kwargs: Dict[str, Any] = {
                "server_port": self.port,
                "share": False,
                "prevent_thread_lock": True,
                "quiet": True,
            }

            launch_result = self._launch_gradio(
                demo,
                launch_kwargs,
            )

            # -------------------------------------------------------------
            # Extract server URL where possible.
            # -------------------------------------------------------------

            self._extract_url(
                launch_result,
                demo,
            )

            with self._state_lock:
                # Do not resurrect a dashboard that was stopped while launch
                # was happening.
                if self._stop_event.is_set():
                    self.running = False
                else:
                    self.running = True
                    self.failed = False

            logger.info(
                "FTRAIN dashboard launched on port %d%s.",
                self.port,
                f" ({self.url})" if self.url else "",
            )

        except Exception as exc:
            with self._state_lock:
                self.running = False
                self.failed = True
                self._launch_error = exc

            logger.exception(
                "FTRAIN dashboard failed to launch."
            )

    def _register_refresh(
        self,
        demo: Any,
        outputs: list,
    ) -> None:
        """
        Register the periodic dashboard refresh.

        Gradio APIs have changed across releases, so this method deliberately
        tries the modern form first and falls back to older behavior.
        """
        try:
            demo.load(
                self._fetch_dashboard,
                inputs=None,
                outputs=outputs,
                every=self.refresh_interval,
            )
            return

        except TypeError:
            pass

        # Older Gradio versions may not accept ``every`` on load.
        #
        # We still register the callback so the dashboard can display initial
        # data. This is preferable to making dashboard startup fatal.
        try:
            demo.load(
                self._fetch_dashboard,
                inputs=None,
                outputs=outputs,
            )

        except Exception:
            logger.exception(
                "Failed to register FTRAIN dashboard refresh callback."
            )
            raise

    def _launch_gradio(
        self,
        demo: Any,
        launch_kwargs: Dict[str, Any],
    ) -> Any:
        """
        Launch Gradio with compatibility fallbacks.
        """
        try:
            return demo.launch(
                **launch_kwargs,
                show_error=True,
            )

        except TypeError:
            # ``show_error`` is unavailable in some Gradio versions.
            return demo.launch(
                **launch_kwargs,
            )

    def _extract_url(
        self,
        launch_result: Any,
        demo: Any,
    ) -> None:
        """
        Attempt to discover the local Gradio URL.

        This is intentionally best-effort because Gradio's return type differs
        across releases.
        """
        candidates = []

        if launch_result is not None:
            candidates.extend(
                [
                    getattr(
                        launch_result,
                        "local_url",
                        None,
                    ),
                    getattr(
                        launch_result,
                        "local_url",
                        None,
                    ),
                ]
            )

            if isinstance(
                launch_result,
                tuple,
            ):
                candidates.extend(
                    launch_result
                )

        candidates.extend(
            [
                getattr(
                    demo,
                    "local_url",
                    None,
                ),
            ]
        )

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                self.url = candidate.strip()
                return

    def stop(
        self,
        *,
        wait: bool = False,
        timeout: float = 5.0,
    ) -> None:
        """
        Stop the dashboard.

        Parameters
        ----------
        wait:
            Whether to wait for the dashboard thread to terminate.

        timeout:
            Maximum amount of time to wait when ``wait=True``.

        Notes
        -----
        Shutdown errors are intentionally swallowed so that calling
        ``dashboard.stop()`` can never break the training process.
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

        thread = self._server_thread

        if (
            wait
            and thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            try:
                thread.join(
                    timeout=max(
                        0.0,
                        float(timeout),
                    )
                )
            except Exception:
                logger.debug(
                    "FTRAIN dashboard thread join failed.",
                    exc_info=True,
                )

        logger.info(
            "FTRAIN dashboard stopped."
        )

    def __enter__(self) -> "TrainingDashboard":
        """Allow ``with TrainingDashboard(...)`` usage."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Stop dashboard when leaving a context manager."""
        self.stop(
            wait=False,
        )

    # =========================================================================
    # Metric ingestion
    # =========================================================================

    def log_metric(
        self,
        step: Any,
        loss: Any,
        lr: Any,
        val_loss: Any = None,
    ) -> bool:
        """
        Queue one training metric without blocking.

        Parameters
        ----------
        step:
            Current global training step.

        loss:
            Training loss.

        lr:
            Current learning rate.

        val_loss:
            Optional validation loss.

        Returns
        -------
        bool
            True when accepted into the dashboard queue.
            False when rejected or the dashboard is inactive.

        Important
        ---------
        This function intentionally does NOT wait for free queue space.
        Training speed should never depend on dashboard speed.
        """
        with self._state_lock:
            if not self.started:
                return False

            # During a launch race, metrics are still useful. They can be
            # queued before Gradio finishes starting.
            if self.failed:
                return False

            self._received_metrics += 1

        normalized_step = _safe_step(
            step,
        )

        if normalized_step is None:
            with self._state_lock:
                self._invalid_metrics += 1

            logger.debug(
                "FTRAIN dashboard rejected invalid step=%r.",
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

        # ---------------------------------------------------------------------
        # Queue without blocking
        # ---------------------------------------------------------------------

        try:
            self.queue.put_nowait(
                metric,
            )

            with self._state_lock:
                self._accepted_metrics += 1

            return True

        except queue.Full:
            pass

        # ---------------------------------------------------------------------
        # Queue is full.
        #
        # Preserve the newest metric because it represents the most current
        # state of the model.
        # ---------------------------------------------------------------------

        try:
            oldest = self.queue.get_nowait()

            try:
                self.queue.task_done()
            except ValueError:
                # Defensive protection against unusual Queue implementations.
                pass

            del oldest

        except queue.Empty:
            pass

        try:
            self.queue.put_nowait(
                metric,
            )

            with self._state_lock:
                self._accepted_metrics += 1
                self._dropped_metrics += 1

            return True

        except queue.Full:
            with self._state_lock:
                self._dropped_metrics += 1

            return False

    # =========================================================================
    # Queue / history processing
    # =========================================================================

    def _drain_queue(self) -> int:
        """
        Drain all currently available metrics.

        Uses ``get_nowait()`` instead of ``queue.empty()`` because ``empty()``
        is not safe as a concurrency decision.
        """
        drained = 0

        while True:
            try:
                metric = self.queue.get_nowait()

            except queue.Empty:
                break

            try:
                self._append_metric(
                    metric,
                )

            except Exception:
                logger.exception(
                    "Failed to process FTRAIN dashboard metric."
                )

            finally:
                try:
                    self.queue.task_done()
                except ValueError:
                    logger.debug(
                        "Dashboard queue task_done() mismatch.",
                        exc_info=True,
                    )

            drained += 1

        return drained

    def _append_metric(
        self,
        metric: Mapping[str, Any],
    ) -> None:
        """
        Validate and append a metric.

        Duplicate steps replace the latest stored value instead of creating
        duplicate x-axis points.

        Older/out-of-order steps are ignored.
        """
        step = _safe_step(
            metric.get("step"),
        )

        if step is None:
            with self._state_lock:
                self._invalid_metrics += 1

            return

        loss = _finite_float(
            metric.get("loss"),
        )

        lr = _finite_float(
            metric.get("lr"),
        )

        val_loss = _finite_float(
            metric.get("val_loss"),
        )

        with self._history_lock:
            # -------------------------------------------------------------
            # Out-of-order metric
            # -------------------------------------------------------------

            if (
                self._last_step is not None
                and step < self._last_step
            ):
                with self._state_lock:
                    self._out_of_order_metrics += 1

                logger.debug(
                    "Ignoring out-of-order dashboard metric: "
                    "step=%d < latest=%d.",
                    step,
                    self._last_step,
                )

                return

            # -------------------------------------------------------------
            # Duplicate step
            # -------------------------------------------------------------

            if (
                self._last_step is not None
                and step == self._last_step
                and self.history["step"]
            ):
                self.history["loss"][-1] = loss
                self.history["lr"][-1] = lr
                self.history["val_loss"][-1] = val_loss

                with self._state_lock:
                    self._duplicate_metrics += 1

                return

            # -------------------------------------------------------------
            # New step
            # -------------------------------------------------------------

            self.history["step"].append(
                step,
            )

            self.history["loss"].append(
                loss,
            )

            self.history["lr"].append(
                lr,
            )

            self.history["val_loss"].append(
                val_loss,
            )

            self._last_step = step

    def _build_dataframe(self):
        """
        Build a pandas DataFrame from a consistent history snapshot.

        Pandas is imported lazily so that simply importing FTRAIN does not
        require dashboard dependencies.
        """
        import pandas as pd

        with self._history_lock:
            data = {
                "step": list(
                    self.history["step"],
                ),
                "loss": list(
                    self.history["loss"],
                ),
                "lr": list(
                    self.history["lr"],
                ),
                "val_loss": list(
                    self.history["val_loss"],
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
            data,
        )

    # =========================================================================
    # Gradio refresh callbacks
    # =========================================================================

    def _fetch_dashboard(
        self,
    ) -> Tuple[Any, Any, Any, str]:
        """
        Process pending metrics and return all dashboard outputs.

        The same DataFrame is intentionally reused for the three LinePlots.
        Gradio selects the appropriate x/y columns for each component.
        """
        try:
            self._drain_queue()

            dataframe = self._build_dataframe()

            with self._state_lock:
                self._refresh_count += 1

            return (
                dataframe,
                dataframe,
                dataframe,
                self._status_text(),
            )

        except Exception as exc:
            with self._state_lock:
                self._refresh_errors += 1

            logger.exception(
                "FTRAIN dashboard refresh failed."
            )

            # Try to provide a valid UI response even after an internal
            # refresh failure.
            try:
                dataframe = self._build_dataframe()
            except Exception:
                import pandas as pd

                dataframe = pd.DataFrame(
                    {
                        "step": [0],
                        "loss": [float("nan")],
                        "lr": [float("nan")],
                        "val_loss": [float("nan")],
                    }
                )

            return (
                dataframe,
                dataframe,
                dataframe,
                f"⚠️ Dashboard refresh error: `{exc}`",
            )

    # Backward-compatible alias for older internal FTRAIN code.
    def _fetch(self):
        return self._fetch_dashboard()

    # =========================================================================
    # Status
    # =========================================================================

    def _status_text(self) -> str:
        """
        Generate the human-readable dashboard status panel.
        """
        with self._state_lock:
            running = self.running
            started = self.started
            failed = self.failed

            received = self._received_metrics
            accepted = self._accepted_metrics
            dropped = self._dropped_metrics
            invalid = self._invalid_metrics
            duplicate = self._duplicate_metrics
            out_of_order = self._out_of_order_metrics
            refreshes = self._refresh_count
            refresh_errors = self._refresh_errors

            last_step = self._last_step
            error = self._launch_error

        if failed:
            state = "🔴 **FAILED**"
        elif running:
            state = "🟢 **RUNNING**"
        elif started:
            state = "🟡 **STARTING / STOPPING**"
        else:
            state = "⚪ **STOPPED**"

        step_text = (
            "N/A"
            if last_step is None
            else str(last_step)
        )

        error_text = ""

        if error is not None:
            error_text = (
                f"\n\n**Last error:** `{error}`"
            )

        return (
            f"**Dashboard:** {state}  \n"
            f"**Latest step:** `{step_text}`  \n"
            f"**Metrics received:** `{received}`  \n"
            f"**Metrics accepted:** `{accepted}`  \n"
            f"**Metrics dropped:** `{dropped}`  \n"
            f"**Invalid metrics:** `{invalid}`  \n"
            f"**Duplicate steps:** `{duplicate}`  \n"
            f"**Out-of-order:** `{out_of_order}`  \n"
            f"**Refreshes:** `{refreshes}`  \n"
            f"**Refresh errors:** `{refresh_errors}`"
            f"{error_text}"
        )

    def get_status(self) -> Dict[str, Any]:
        """
        Return structured runtime diagnostics.

        Useful for:
            • tests
            • CLI diagnostics
            • FTRAIN core
            • debugging dashboard startup
        """
        with self._state_lock:
            running = self.running
            started = self.started
            failed = self.failed

            received = self._received_metrics
            accepted = self._accepted_metrics
            dropped = self._dropped_metrics
            invalid = self._invalid_metrics
            duplicate = self._duplicate_metrics
            out_of_order = self._out_of_order_metrics
            refresh_count = self._refresh_count
            refresh_errors = self._refresh_errors

            last_step = self._last_step
            error = self._launch_error

        with self._history_lock:
            history_points = len(
                self.history["step"]
            )

        server_thread = self._server_thread

        return {
            "running": running,
            "started": started,
            "failed": failed,
            "port": self.port,
            "url": self.url,
            "received_metrics": received,
            "accepted_metrics": accepted,
            "dropped_metrics": dropped,
            "invalid_metrics": invalid,
            "duplicate_metrics": duplicate,
            "out_of_order_metrics": out_of_order,
            "refresh_count": refresh_count,
            "refresh_errors": refresh_errors,
            "last_step": last_step,
            "queued_metrics": self.queue.qsize(),
            "history_points": history_points,
            "max_points": self.max_points,
            "queue_maxsize": self.queue_maxsize,
            "refresh_interval": self.refresh_interval,
            "server_thread_alive": (
                server_thread.is_alive()
                if server_thread is not None
                else False
            ),
            "error": (
                str(error)
                if error is not None
                else None
            ),
        }

    # =========================================================================
    # History management
    # =========================================================================

    def clear_history(self) -> None:
        """
        Clear all stored metrics and pending queue data.

        The dashboard itself remains running.
        """
        with self._history_lock:
            for values in self.history.values():
                values.clear()

            self._last_step = None

        while True:
            try:
                self.queue.get_nowait()

            except queue.Empty:
                break

            else:
                try:
                    self.queue.task_done()
                except ValueError:
                    pass

        logger.debug(
            "FTRAIN dashboard history cleared."
        )

    # =========================================================================
    # Convenience properties
    # =========================================================================

    @property
    def is_running(self) -> bool:
        """Whether the dashboard is currently considered running."""
        with self._state_lock:
            return self.running

    @property
    def is_healthy(self) -> bool:
        """
        Whether the dashboard is running without a recorded startup failure.
        """
        with self._state_lock:
            return self.running and not self.failed

    @property
    def latest_step(self) -> Optional[int]:
        """Return the latest processed training step."""
        with self._state_lock:
            return self._last_step

    @property
    def pending_metrics(self) -> int:
        """Return the number of metrics currently waiting in the queue."""
        return self.queue.qsize()

    # =========================================================================
    # Cleanup
    # =========================================================================

    def __del__(self) -> None:
        """
        Best-effort cleanup.

        Destructors should never raise, especially for an optional dashboard
        subsystem.
        """
        try:
            self.stop(
                wait=False,
            )
        except Exception:
            pass
