"""
FTRAIN UI v1.1
==============
Beautiful, robust, dependency-free terminal UI for the FTRAIN engine.

Features
--------
- FTRAIN v1.1 branding
- Cross-platform ANSI/color detection
- Beautiful fire header
- Smart progress bars
- Training dashboard rows
- Merge progress
- Captain reports
- Stage banners
- Metric cards
- Final summaries
- Animated loading bars
- Thread-safe terminal writes
- No crashes when output is redirected
- Safe formatting of arbitrary values
- Backward-compatible public API
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from typing import Any, Dict, Iterable, Mapping, Optional


# ============================================================================
# Terminal / ANSI
# ============================================================================

CLEAR_LINE = "\033[2K\033[G"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

# Foreground colors
BLACK = "\033[30m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

# Bright foreground
BRIGHT_RED = "\033[91m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
BRIGHT_CYAN = "\033[96m"
BRIGHT_WHITE = "\033[97m"

# Background
BG_BLUE = "\033[48;5;19m"
BG_CYAN = "\033[48;5;45m"
BG_YELLOW = "\033[48;5;226m"
BG_ORANGE = "\033[48;5;208m"
BG_RED = "\033[48;5;196m"
BG_GRAY = "\033[48;5;236m"
BG_DARK = "\033[48;5;234m"

# 256-color foregrounds used by FTRAIN's visual identity
ORANGE = "\033[38;5;208m"
GOLD = "\033[38;5;214m"
GRAY = "\033[38;5;245m"
DARK_GRAY = "\033[38;5;239m"
NEON_CYAN = "\033[38;5;51m"
NEON_GREEN = "\033[38;5;46m"
NEON_BLUE = "\033[38;5;39m"

_OUTPUT_LOCK = threading.RLock()


def _supports_color() -> bool:
    """Return whether ANSI color output should be used."""
    try:
        if os.environ.get("FTRAIN_NO_COLOR", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return False

        if os.environ.get("NO_COLOR") is not None:
            return False

        stream = sys.stdout
        if not hasattr(stream, "isatty"):
            return False

        return bool(stream.isatty())

    except Exception:
        return False


USE_COLOR = _supports_color()


def _c(code: str, text: Any) -> str:
    """Apply an ANSI code when supported."""
    value = str(text)
    return f"{code}{value}{RESET}" if USE_COLOR else value


def _emit(text: str = "", *, end: str = "\n", flush: bool = True) -> None:
    """Thread-safe terminal output."""
    with _OUTPUT_LOCK:
        try:
            sys.stdout.write(text + end)
            if flush:
                sys.stdout.flush()
        except (BrokenPipeError, OSError):
            # UI must never terminate training because stdout disappeared.
            pass


def _rewrite(text: str) -> None:
    """Thread-safe single-line rewrite."""
    with _OUTPUT_LOCK:
        try:
            sys.stdout.write(f"\r{CLEAR_LINE}{text}")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass


def _terminal_width(default: int = 88) -> int:
    try:
        return max(60, min(shutil.get_terminal_size((default, 20)).columns, 140))
    except Exception:
        return default


# ============================================================================
# Formatting helpers
# ============================================================================

def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default

    if result != result:  # NaN
        return default

    if result in (float("inf"), float("-inf")):
        return default

    return result


def _format_metric(
    value: Any,
    decimals: int = 4,
    fallback: str = "N/A",
) -> str:
    number = _safe_float(value)
    if number is None:
        return fallback
    return f"{number:.{decimals}f}"


def _format_lr(value: Any) -> str:
    number = _safe_float(value)
    return "N/A" if number is None else f"{number:.2e}"


def _format_duration(seconds: Any) -> str:
    value = _safe_float(seconds, 0.0) or 0.0
    value = max(0.0, value)

    if value < 60:
        return f"{value:.1f}s"

    minutes = int(value // 60)
    secs = int(value % 60)

    if minutes < 60:
        return f"{minutes}m {secs:02d}s"

    hours = minutes // 60
    minutes %= 60

    if hours < 24:
        return f"{hours}h {minutes:02d}m"

    days = hours // 24
    hours %= 24
    return f"{days}d {hours:02d}h"


def _human_number(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "N/A"

    absolute = abs(number)

    if absolute >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"

    if absolute >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"

    if absolute >= 1_000:
        return f"{number / 1_000:.2f}K"

    if number.is_integer():
        return str(int(number))

    return f"{number:.2f}"


def _truncate(text: Any, width: int) -> str:
    value = str(text).replace("\n", " ").replace("\r", " ")
    if len(value) <= width:
        return value
    if width <= 3:
        return value[:width]
    return value[: width - 3] + "..."


def _sanitize_text(text: Any) -> str:
    return str(text).replace("\x1b", "").replace("\r", "")


# ============================================================================
# Box / separator helpers
# ============================================================================

def _rule(width: Optional[int] = None, char: str = "─") -> str:
    width = width or _terminal_width()
    return _c(DARK_GRAY, char * width)


def _box_title(title: str, width: Optional[int] = None) -> str:
    width = width or _terminal_width()
    inner = max(8, width - 4)
    return (
        _c(DARK_GRAY, "╭" + "─" * (inner + 2) + "╮")
        + "\n"
        + _c(DARK_GRAY, "│ ")
        + _c(BOLD + NEON_CYAN, _truncate(title, inner))
        + _c(DARK_GRAY, " " * max(0, inner - len(_truncate(title, inner))) + " │")
        + "\n"
        + _c(DARK_GRAY, "╰" + "─" * (inner + 2) + "╯")
    )


def _card_line(label: str, value: Any, width: int = 34) -> str:
    label_text = _truncate(label, width)
    return (
        f"{_c(BOLD, label_text):<{width + (0 if not USE_COLOR else 0)}} "
        f"{_c(NEON_CYAN, value)}"
    )


# ============================================================================
# Progress bars
# ============================================================================

def gradient_bar(
    progress: float,
    width: int = 24,
    from_blue_to_orange: bool = False,
    *,
    fill_char: str = " ",
    empty_char: str = " ",
    show_percent: bool = False,
) -> str:
    """
    Render a smooth terminal progress bar.

    Backward compatible with the original signature.
    """
    try:
        value = float(progress)
    except (TypeError, ValueError):
        value = 0.0

    value = max(0.0, min(1.0, value))
    width = max(4, int(width))
    filled = int(round(value * width))

    if from_blue_to_orange:
        if value < 0.34:
            fill_code = BG_BLUE
        elif value < 0.67:
            fill_code = BG_ORANGE
        else:
            fill_code = BG_RED
    else:
        if value < 0.25:
            fill_code = BG_BLUE
        elif value < 0.50:
            fill_code = BG_CYAN
        elif value < 0.75:
            fill_code = BG_YELLOW
        else:
            fill_code = BG_ORANGE

    if not USE_COLOR:
        fill = "━"
        empty = "─"
        result = "█" * filled + "░" * (width - filled)
    else:
        fill = f"{fill_code}{fill_char}{RESET}"
        empty = f"{BG_GRAY}{empty_char}{RESET}"
        result = fill * filled + empty * (width - filled)

    if show_percent:
        return f"[{result}] {value * 100:6.2f}%"

    return f"|{result}|"


def _thin_bar(progress: float, width: int = 30) -> str:
    value = max(0.0, min(1.0, float(progress)))
    filled = int(round(value * width))

    if not USE_COLOR:
        return "[" + "━" * filled + "─" * (width - filled) + "]"

    return (
        "["
        + _c(ORANGE, "━" * filled)
        + _c(DARK_GRAY, "─" * (width - filled))
        + "]"
    )


# ============================================================================
# Header
# ============================================================================

_FTRAIN_ART = r"""
 ███████╗████████╗██████╗  █████╗ ██╗███╗   ██╗
 ██╔════╝╚══██╔══╝██╔══██╗██╔══██╗██║████╗  ██║
 █████╗     ██║   ██████╔╝███████║██║██╔██╗ ██║
 ██╔══╝     ██║   ██╔══██╗██╔══██║██║██║╚██╗██║
 ██║        ██║   ██║  ██║██║  ██║██║██║ ╚████║
 ╚═╝        ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝
"""


def fire_header(
    version: str = "1.1.0",
    subtitle: str = "Adaptive Training • Intelligent Merging • Captain AI",
) -> None:
    """Print the main FTRAIN v1.1 startup banner."""
    width = min(_terminal_width(), 96)
    sep = "═" * width

    if USE_COLOR:
        _emit("")
        _emit(_c(ORANGE, _FTRAIN_ART.rstrip()))
        _emit(_c(DARK_GRAY, sep))
        _emit(
            _c(
                BOLD + GOLD,
                f"🔥🔥  FTRAIN ENGINE v{version}  🔥🔥",
            )
        )
        _emit(_c(NEON_CYAN, _truncate(subtitle, width)))
        _emit(_c(DARK_GRAY, sep))
        _emit(
            _c(
                DIM + GRAY,
                "   TRAIN  •  ADAPT  •  MERGE  •  EVOLVE",
            )
        )
        _emit("")
    else:
        _emit("")
        _emit(_FTRAIN_ART.rstrip())
        _emit(sep)
        _emit(f"🔥🔥  FTRAIN ENGINE v{version}  🔥🔥")
        _emit(subtitle)
        _emit(sep)
        _emit("   TRAIN  •  ADAPT  •  MERGE  •  EVOLVE")
        _emit("")


# ============================================================================
# Stage / status UI
# ============================================================================

def print_stage(
    title: str,
    message: str = "",
    icon: str = "🔥",
    status: str = "RUNNING",
) -> None:
    """Beautiful stage banner for core training/merging phases."""
    width = min(_terminal_width(), 96)
    status_upper = str(status).upper()

    if status_upper in {"DONE", "SUCCESS", "OK"}:
        status_text = _c(BOLD + NEON_GREEN, "● DONE")
    elif status_upper in {"ERROR", "FAILED"}:
        status_text = _c(BOLD + BRIGHT_RED, "● FAILED")
    elif status_upper in {"WARN", "WARNING"}:
        status_text = _c(BOLD + BRIGHT_YELLOW, "● WARNING")
    else:
        status_text = _c(BOLD + NEON_CYAN, "● RUNNING")

    _emit("")
    _emit(_c(DARK_GRAY, "╭" + "─" * (width - 2) + "╮"))
    header = f"{icon}  {title}"
    _emit(
        _c(DARK_GRAY, "│ ")
        + _c(BOLD + GOLD, _truncate(header, width - 10))
        + " "
        + status_text
        + _c(DARK_GRAY, " " * max(0, width - len(_strip_ansi(header)) - 17) + "│")
    )
    if message:
        _emit(
            _c(DARK_GRAY, "│ ")
            + _c(GRAY, _truncate(message, width - 4))
            + _c(DARK_GRAY, " " * max(0, width - len(_sanitize_text(message)) - 4) + "│")
        )
    _emit(_c(DARK_GRAY, "╰" + "─" * (width - 2) + "╯"))


def print_status(
    message: str,
    *,
    level: str = "info",
    icon: Optional[str] = None,
) -> None:
    """Print a compact colored status message."""
    level = str(level).lower()

    if icon is None:
        icon = {
            "info": "ℹ️",
            "success": "✅",
            "warning": "⚠️",
            "error": "❌",
            "brain": "🧠",
            "merge": "🧩",
            "train": "🔥",
        }.get(level, "•")

    color = {
        "info": NEON_CYAN,
        "success": NEON_GREEN,
        "warning": BRIGHT_YELLOW,
        "error": BRIGHT_RED,
        "brain": NEON_CYAN,
        "merge": ORANGE,
        "train": GOLD,
    }.get(level, WHITE)

    _emit(f"{icon} {_c(color, _sanitize_text(message))}")


# ============================================================================
# Training UI
# ============================================================================

def print_train_table(
    step,
    total_steps,
    loss,
    val_loss,
    lr,
    grad_norm,
    captain_msg="",
    *,
    elapsed: Optional[float] = None,
    tokens_per_second: Optional[float] = None,
    epoch: Optional[Any] = None,
) -> None:
    """Print a rich training row while preserving the original API."""
    total = _safe_float(total_steps, 0.0) or 0.0
    current = _safe_float(step, 0.0) or 0.0
    progress = current / total if total > 0 else 0.0
    progress = max(0.0, min(1.0, progress))

    step_text = (
        f"{int(current):>5}/{int(total):<5}"
        if total.is_integer()
        else f"{current:>5.1f}/{total:<5.1f}"
    )

    loss_str = _format_metric(loss)
    val_str = _format_metric(val_loss)
    lr_str = _format_lr(lr)
    grad_str = _format_metric(grad_norm)
    epoch_str = _truncate(epoch, 8) if epoch is not None else None

    chunks = [
        f"🔥 {_c(BOLD, 'Step')} {_c(NEON_CYAN, step_text)}",
        f"Loss {_c(ORANGE, loss_str)}",
        f"Val {_c(NEON_BLUE, val_str)}",
        f"LR {_c(NEON_CYAN, lr_str)}",
        f"Grad {_c(GOLD, grad_str)}",
    ]

    if epoch_str is not None:
        chunks.append(f"Epoch {_c(NEON_GREEN, epoch_str)}")

    if elapsed is not None:
        chunks.append(f"Time {_c(GRAY, _format_duration(elapsed))}")

    if tokens_per_second is not None:
        chunks.append(
            f"Tok/s {_c(NEON_GREEN, _human_number(tokens_per_second))}"
        )

    if captain_msg:
        chunks.append(
            f"🧠 {_c(NEON_CYAN, _truncate(captain_msg, 34))}"
        )

    line = (
        " ".join(chunks)
        + "  "
        + _thin_bar(progress, 18)
        + f" {_c(BOLD + GOLD, f'{progress * 100:6.2f}%')}"
    )

    _rewrite(line)

    # Keep old behavior: training rows are emitted as real lines.
    with _OUTPUT_LOCK:
        try:
            sys.stdout.write("\n")
            sys.stdout.flush()
        except (BrokenPipeError, OSError):
            pass


def print_training_metrics(
    *,
    step: int,
    total_steps: int,
    loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    lr: Optional[float] = None,
    grad_norm: Optional[float] = None,
    grad_health: Optional[float] = None,
    captain: Optional[str] = None,
    throughput: Optional[float] = None,
) -> None:
    """Dedicated v1.1 metrics row."""
    parts = [
        f"{_c(BOLD, 'STEP')} {_c(NEON_CYAN, f'{step}/{total_steps}')}",
        f"L={_c(ORANGE, _format_metric(loss))}",
        f"V={_c(NEON_BLUE, _format_metric(val_loss))}",
        f"LR={_c(NEON_CYAN, _format_lr(lr))}",
        f"G={_c(GOLD, _format_metric(grad_norm))}",
    ]

    if grad_health is not None:
        parts.append(
            f"GH={_c(NEON_GREEN, f'{float(grad_health):.2%}')}"
        )

    if throughput is not None:
        parts.append(
            f"T={_c(NEON_GREEN, f'{throughput:.1f}/s')}"
        )

    if captain:
        parts.append(
            f"🧠 {_c(NEON_CYAN, _truncate(captain, 30))}"
        )

    progress = (
        step / total_steps
        if total_steps
        else 0.0
    )

    _rewrite(
        " │ ".join(parts)
        + "  "
        + gradient_bar(
            progress,
            16,
            from_blue_to_orange=False,
        )
    )


# ============================================================================
# Merge UI
# ============================================================================

def print_merge_progress(
    current,
    total,
    message="",
    *,
    matched: Optional[int] = None,
    projected: Optional[int] = None,
    rejected: Optional[int] = None,
    strategy: Optional[str] = None,
) -> None:
    """Enhanced merge progress; old 3-argument call remains valid."""
    total_value = _safe_float(total, 0.0) or 0.0
    current_value = _safe_float(current, 0.0) or 0.0
    progress = (
        current_value / total_value
        if total_value > 0
        else 0.0
    )
    progress = max(0.0, min(1.0, progress))

    parts = [
        f"🧩 {_c(BOLD + NEON_CYAN, 'MERGE')}",
        f"{_c(BOLD, f'{progress * 100:6.2f}%')}",
        gradient_bar(
            progress,
            24,
            from_blue_to_orange=True,
        ),
    ]

    if matched is not None:
        parts.append(f"M:{_c(NEON_GREEN, matched)}")

    if projected is not None:
        parts.append(f"P:{_c(GOLD, projected)}")

    if rejected is not None:
        parts.append(f"R:{_c(BRIGHT_RED, rejected)}")

    if strategy:
        parts.append(
            f"[{_c(NEON_CYAN, _truncate(strategy, 18))}]"
        )

    if message:
        parts.append(
            f"{_c(GRAY, '(' + _truncate(message, 36) + ')')}"
        )

    _rewrite(" ".join(parts))

    if progress >= 1.0:
        with _OUTPUT_LOCK:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                pass


# ============================================================================
# Captain
# ============================================================================

def print_captain_report(report: str) -> None:
    """Pretty-print the Captain's textual analysis."""
    width = min(_terminal_width(), 90)

    _emit("")
    _emit(_c(DARK_GRAY, "╭" + "─" * (width - 2) + "╮"))
    _emit(
        _c(DARK_GRAY, "│ ")
        + _c(
            BOLD + NEON_CYAN,
            "🧠 CAPTAIN ANALYSIS".center(width - 4),
        )
        + _c(DARK_GRAY, " │")
    )
    _emit(_c(DARK_GRAY, "├" + "─" * (width - 2) + "┤"))

    for raw_line in str(report).splitlines() or [""]:
        line = _truncate(_sanitize_text(raw_line), width - 6)
        _emit(
            _c(DARK_GRAY, "│ ")
            + _c(WHITE, line)
            + " " * max(0, width - len(line) - 4)
            + _c(DARK_GRAY, "│")
        )

    _emit(_c(DARK_GRAY, "╰" + "─" * (width - 2) + "╯"))
    _emit("")


def print_captain_advice(
    action: str,
    multiplier: Optional[float] = None,
    reason: Optional[str] = None,
) -> None:
    """Compact v1.1 Captain decision card."""
    _emit(
        f"🧠 {_c(BOLD + NEON_CYAN, 'CAPTAIN')} "
        f"{_c(GOLD, _sanitize_text(action))}"
    )

    if multiplier is not None:
        _emit(
            f"   LR multiplier: "
            f"{_c(BOLD + ORANGE, f'x{float(multiplier):.3f}')}"
        )

    if reason:
        _emit(
            f"   {_c(GRAY, _truncate(reason, _terminal_width() - 6))}"
        )


# ============================================================================
# Final summaries
# ============================================================================

def print_final_summary(stats: Dict[str, Any]) -> None:
    """Render a beautiful final process summary."""
    width = min(_terminal_width(), 96)

    _emit("")
    _emit(_c(DARK_GRAY, "╔" + "═" * (width - 2) + "╗"))
    _emit(
        _c(DARK_GRAY, "║ ")
        + _c(
            BOLD + NEON_GREEN,
            "✅ FTRAIN PROCESS COMPLETED".center(width - 4),
        )
        + _c(DARK_GRAY, " ║")
    )
    _emit(_c(DARK_GRAY, "╠" + "═" * (width - 2) + "╣"))

    for key, value in stats.items():
        label = str(key).replace("_", " ").title()
        display = _sanitize_text(value)

        if "http://" in display or "https://" in display:
            display = _c(NEON_CYAN + "\033[4m", display)
        elif any(
            token in label.lower()
            for token in (
                "loss",
                "accuracy",
                "improvement",
                "progress",
            )
        ):
            display = _c(GOLD, display)
        else:
            display = _c(WHITE, display)

        label_plain = _truncate(label, 27)
        _emit(
            _c(DARK_GRAY, "║ ")
            + _c(BOLD, f"{label_plain:<27}")
            + _c(DARK_GRAY, " │ ")
            + display
        )

    _emit(_c(DARK_GRAY, "╚" + "═" * (width - 2) + "╝"))
    _emit("")


def print_metric_summary(
    title: str,
    metrics: Mapping[str, Any],
    *,
    icon: str = "📊",
) -> None:
    """Generic metric-card renderer."""
    width = min(_terminal_width(), 90)

    _emit("")
    _emit(_c(DARK_GRAY, "╭" + "─" * (width - 2) + "╮"))
    _emit(
        _c(DARK_GRAY, "│ ")
        + _c(BOLD + NEON_CYAN, f"{icon} {title}")
        + " " * max(0, width - len(_sanitize_text(title)) - 7)
        + _c(DARK_GRAY, "│")
    )
    _emit(_c(DARK_GRAY, "├" + "─" * (width - 2) + "┤"))

    for key, value in metrics.items():
        label = str(key).replace("_", " ").title()
        if isinstance(value, float):
            if "rate" in key.lower() or "ratio" in key.lower():
                display = f"{value:.2%}"
            else:
                display = f"{value:.5f}"
        else:
            display = str(value)

        line = (
            f"  {_c(BOLD, _truncate(label, 25))}"
            f"{' ' * max(1, 27 - len(_truncate(label, 25)))}"
            f"{_c(NEON_CYAN, _truncate(display, width - 31))}"
        )

        _emit(
            _c(DARK_GRAY, "│")
            + line
            + " " * max(0, width - len(_sanitize_text(line)) - 1)
            + _c(DARK_GRAY, "│")
        )

    _emit(_c(DARK_GRAY, "╰" + "─" * (width - 2) + "╯"))


# ============================================================================
# Loading bar
# ============================================================================

class LoadingBar:
    """
    Thread-safe loading/progress bar.

    Backward compatible with:
        LoadingBar(message="...", real_progress=True)
        start()
        update(current, total)
        done()

    ``real_progress=False`` creates an animated indeterminate bar.
    """

    def __init__(
        self,
        message: str = "Loading model",
        real_progress: bool = True,
        *,
        width: int = 22,
        update_interval: float = 0.05,
    ) -> None:
        self.message = str(message)
        self.real = bool(real_progress)
        self.width = max(8, int(width))
        self.update_interval = max(0.01, float(update_interval))

        self.stop_event = threading.Event()
        self._progress = 0.0
        self._running = False
        self.thread: Optional[threading.Thread] = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.done()

    @property
    def progress(self) -> float:
        return self._progress

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self.stop_event.clear()

        if self.real:
            _rewrite(
                f"📦 {_c(BOLD, self.message)} "
                f"{gradient_bar(0.0, self.width, from_blue_to_orange=True)} 0.0%"
            )
            return

        self.thread = threading.Thread(
            target=self._run,
            name="ftrain-loading-bar",
            daemon=True,
        )
        self.thread.start()

    def _run(self) -> None:
        position = 0
        direction = 1

        while not self.stop_event.is_set():
            position += direction

            if position >= self.width - 1:
                position = self.width - 1
                direction = -1

            elif position <= 0:
                position = 0
                direction = 1

            if USE_COLOR:
                bar = (
                    "["
                    + _c(BG_BLUE, " " * position)
                    + _c(BG_ORANGE, "██")
                    + _c(BG_GRAY, " " * max(0, self.width - position - 2))
                    + "]"
                )
            else:
                bar = (
                    "["
                    + " " * position
                    + "██"
                    + " " * max(0, self.width - position - 2)
                    + "]"
                )

            _rewrite(
                f"📦 {_c(BOLD, self.message)} "
                f"{bar} {_c(GOLD, 'working…')}"
            )

            self.stop_event.wait(
                self.update_interval
            )

    def update(self, current, total) -> None:
        try:
            current_value = float(current)
            total_value = float(total)
        except (TypeError, ValueError):
            return

        progress = (
            current_value / total_value
            if total_value > 0
            else 0.0
        )

        progress = max(
            0.0,
            min(1.0, progress),
        )

        self._progress = progress

        _rewrite(
            f"📦 {_c(BOLD, self.message)} "
            f"{gradient_bar(progress, self.width, from_blue_to_orange=True)} "
            f"{progress * 100:6.2f}%"
        )

    def done(self) -> None:
        if not self._running:
            return

        self._progress = 1.0
        self.stop_event.set()

        if (
            self.thread is not None
            and self.thread.is_alive()
            and self.thread is not threading.current_thread()
        ):
            self.thread.join(timeout=1.0)

        self.thread = None
        self._running = False

        _rewrite(
            f"📦 {_c(BOLD, self.message)} "
            f"{gradient_bar(1.0, self.width, from_blue_to_orange=True)} "
            f"{_c(BOLD + NEON_GREEN, '100.00%')} ✅"
        )

        with _OUTPUT_LOCK:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except (BrokenPipeError, OSError):
                pass


# ============================================================================
# Convenience helpers for the enhanced FTRAIN core
# ============================================================================

def print_model_info(
    *,
    model_name: Optional[str] = None,
    family: Optional[str] = None,
    parameters: Optional[Any] = None,
    trainable: Optional[Any] = None,
    device: Optional[str] = None,
    dtype: Optional[str] = None,
    quantized: Optional[bool] = None,
) -> None:
    """Compact model information card."""
    metrics: Dict[str, Any] = {}

    if model_name is not None:
        metrics["Model"] = _truncate(model_name, 58)
    if family is not None:
        metrics["Family"] = family
    if parameters is not None:
        metrics["Parameters"] = _human_number(parameters)
    if trainable is not None:
        metrics["Trainable"] = _human_number(trainable)
    if device is not None:
        metrics["Device"] = device
    if dtype is not None:
        metrics["DType"] = dtype
    if quantized is not None:
        metrics["4-bit"] = "Enabled" if quantized else "Disabled"

    print_metric_summary(
        "MODEL READY",
        metrics,
        icon="🧠",
    )


def print_merge_summary(
    *,
    matched: int,
    total: int,
    projected: int = 0,
    preserved: int = 0,
    rejected: int = 0,
    strategy: str = "intelligent",
) -> None:
    total = max(0, int(total))
    matched = max(0, int(matched))

    coverage = (
        matched / total
        if total > 0
        else 0.0
    )

    print_metric_summary(
        "MERGE REPORT",
        {
            "Matched tensors": matched,
            "Coverage": coverage,
            "Projected tensors": projected,
            "Preserved tensors": preserved,
            "Rejected tensors": rejected,
            "Strategy": strategy,
        },
        icon="🧬",
    )


def print_training_result(
    *,
    initial_loss: Optional[float] = None,
    final_loss: Optional[float] = None,
    steps: Optional[int] = None,
    best_loss: Optional[float] = None,
) -> None:
    """Dedicated post-training result card."""
    metrics: Dict[str, Any] = {}

    if initial_loss is not None:
        metrics["Initial loss"] = float(initial_loss)

    if final_loss is not None:
        metrics["Final loss"] = float(final_loss)

    if best_loss is not None:
        metrics["Best loss"] = float(best_loss)

    if steps is not None:
        metrics["Training steps"] = int(steps)

    if (
        initial_loss is not None
        and final_loss is not None
        and initial_loss != 0
    ):
        metrics["Improvement"] = (
            (float(initial_loss) - float(final_loss))
            / abs(float(initial_loss))
        )

    print_metric_summary(
        "TRAINING RESULT",
        metrics,
        icon="📈",
    )


# ============================================================================
# Internal ANSI helper
# ============================================================================

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _strip_ansi(value: str) -> str:
    return _ANSI_RE.sub("", str(value))


__all__ = [
    "CLEAR_LINE",
    "RESET",
    "fire_header",
    "gradient_bar",
    "print_train_table",
    "print_training_metrics",
    "print_merge_progress",
    "print_captain_report",
    "print_captain_advice",
    "print_final_summary",
    "print_metric_summary",
    "print_model_info",
    "print_merge_summary",
    "print_training_result",
    "print_stage",
    "print_status",
    "LoadingBar",
    "USE_COLOR",
]
