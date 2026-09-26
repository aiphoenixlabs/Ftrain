"""
FTRAIN UI v1.2 "Aurora"
=======================
Beautiful, robust, dependency-free terminal UI for the FTRAIN engine.

v1.2 highlights
---------------
- Truecolor (24-bit) gradients for banners, bars and pills, with a graceful
  256-color fallback and a plain-text fallback when color is unavailable.
- Windows legacy console support: enables ANSI virtual-terminal processing
  via the Win32 API instead of silently dropping every escape sequence.
- Display-width-aware box drawing: emoji, CJK and other wide characters no
  longer break panel alignment.
- Smooth multi-stop gradient progress bars with milestone coloring.
- Markdown-flavored Captain reports (headings, bullets, bold, code, links).
- Animated braille-spinner loading bar with elapsed/duration reporting.
- Fixed-width aligned training/merge rows so repeated lines stop jittering.
- Backward-compatible public API: every v1.1 name and signature is kept.

Environment switches
--------------------
- ``FTRAIN_NO_COLOR`` or ``NO_COLOR``: disable all ANSI styling.
- ``FTRAIN_FORCE_COLOR`` / ``FORCE_COLOR``: force styling even when stdout
  is redirected (useful for demos and captured logs).
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ============================================================================
# Terminal / ANSI
# ============================================================================

CLEAR_LINE = "\033[2K\033[G"
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"
UNDERLINE = "\033[4m"

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
BG_GREEN = "\033[48;5;35m"
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


def _enable_windows_vt() -> bool:
    """
    Enable ANSI virtual-terminal processing on legacy Windows consoles.

    Modern Windows Terminal reports VT support natively; classic conhost
    (Windows 10/11) accepts escape sequences only after the console mode is
    switched on. Without this, every color code renders as ``←[38;5;214m``.
    """
    if os.name != "nt":
        return False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        if not handle or handle == -1:
            return False

        mode = ctypes.c_uint32()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            return False

        if mode.value & 0x0004:  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            return True

        return bool(kernel32.SetConsoleMode(handle, mode.value | 0x0004))
    except Exception:
        return False


def _force_color_env() -> bool:
    for name in ("FTRAIN_FORCE_COLOR", "FORCE_COLOR"):
        value = os.environ.get(name, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value:
            return False
    return False


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

        if _force_color_env():
            _enable_windows_vt()
            return True

        stream = sys.stdout
        if not hasattr(stream, "isatty"):
            return False

        if not stream.isatty():
            return False

        if os.name == "nt":
            # Terminal-native VT support (WT_SESSION) still benefits from an
            # explicit enable on some shells; it is a no-op when already set.
            _enable_windows_vt()

        return True

    except Exception:
        return False


USE_COLOR = _supports_color()


def _detect_truecolor() -> bool:
    if not USE_COLOR:
        return False
    try:
        colorterm = os.environ.get("COLORTERM", "").strip().lower()
        if "truecolor" in colorterm or "24bit" in colorterm:
            return True
        if os.environ.get("WT_SESSION"):
            return True
        term = os.environ.get("TERM", "").strip().lower()
        if "truecolor" in term or "24bit" in term:
            return True
        # iTerm2 and compatible terminals advertise TERM_PROGRAM.
        if os.environ.get("TERM_PROGRAM", "").strip().lower() in {
            "iterm.app",
            "wezterm",
            "vscode",
        }:
            return True
    except Exception:
        pass
    return False


USE_TRUECOLOR = _detect_truecolor()


def _c(code: str, text: Any) -> str:
    """Apply an ANSI code when supported."""
    value = str(text)
    return f"{code}{value}{RESET}" if USE_COLOR else value


# ============================================================================
# Truecolor primitives
# ============================================================================

def _rgb(r: int, g: int, b: int) -> Tuple[int, int, int]:
    return (max(0, min(255, int(r))), max(0, min(255, int(g))), max(0, min(255, int(b))))


def _blend(a: Tuple[int, int, int], b: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    t = max(0.0, min(1.0, float(t)))
    return _rgb(
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    )


def _sample_gradient(stops: Sequence[Tuple[int, int, int]], t: float) -> Tuple[int, int, int]:
    """Sample a multi-stop gradient at position ``t`` in [0, 1]."""
    if not stops:
        return _rgb(255, 255, 255)
    if len(stops) == 1:
        return stops[0]

    t = max(0.0, min(1.0, float(t)))
    segments = len(stops) - 1
    position = t * segments
    index = min(int(position), segments - 1)
    local = position - index
    return _blend(stops[index], stops[index + 1], local)


def _rgb_to_256(color: Tuple[int, int, int]) -> int:
    """Map an RGB triple to the nearest xterm-256 palette index."""
    r, g, b = color

    if abs(r - g) < 8 and abs(g - b) < 8 and abs(r - b) < 8:
        if r < 8:
            return 16
        if r > 238:
            return 231
        return 232 + max(0, min(23, round((r - 8) / 10.0)))

    cube = (0, 95, 135, 175, 215, 255)

    def nearest(channel: int) -> int:
        return min(range(6), key=lambda i: (cube[i] - channel) ** 2)

    ri, gi, bi = nearest(r), nearest(g), nearest(b)
    cube_error = (
        (cube[ri] - r) ** 2
        + (cube[gi] - g) ** 2
        + (cube[bi] - b) ** 2
    )

    gray = 232 + max(0, min(23, round(((r + g + b) / 3.0 - 8) / 10.0)))
    gray_level = 8 + 10 * (gray - 232)
    gray_error = (gray_level - r) ** 2 + (gray_level - g) ** 2 + (gray_level - b) ** 2

    if gray_error < cube_error:
        return gray
    return 16 + 36 * ri + 6 * gi + bi


def _fg(color: Tuple[int, int, int]) -> str:
    if not USE_COLOR:
        return ""
    if USE_TRUECOLOR:
        return f"\033[38;2;{color[0]};{color[1]};{color[2]}m"
    return f"\033[38;5;{_rgb_to_256(color)}m"


def _bg(color: Tuple[int, int, int]) -> str:
    if not USE_COLOR:
        return ""
    if USE_TRUECOLOR:
        return f"\033[48;2;{color[0]};{color[1]};{color[2]}m"
    return f"\033[48;5;{_rgb_to_256(color)}m"


def _paint(color: Tuple[int, int, int], text: Any) -> str:
    """Colorize text with an RGB/256-auto foreground color."""
    value = str(text)
    if not USE_COLOR:
        return value
    return f"{_fg(color)}{value}{RESET}"


# Aurora palette ------------------------------------------------------------
# Named roles keep the visual identity consistent across every renderer.

AURORA_BLUE = _rgb(59, 130, 246)
AURORA_CYAN = _rgb(34, 211, 238)
AURORA_MINT = _rgb(16, 185, 129)
AURORA_AMBER = _rgb(250, 204, 21)
AURORA_ORANGE = _rgb(249, 115, 22)
AURORA_RED = _rgb(220, 38, 38)
AURORA_VIOLET = _rgb(139, 92, 246)
AURORA_GOLD = _rgb(255, 191, 73)
AURORA_TEXT = _rgb(226, 232, 240)
AURORA_MUTED = _rgb(122, 132, 150)
AURORA_RULE = _rgb(64, 72, 90)
AURORA_TRACK = _rgb(52, 58, 72)
AURORA_SUCCESS = _rgb(52, 211, 153)
AURORA_DANGER = _rgb(248, 113, 113)
AURORA_WARN = _rgb(251, 191, 36)

# Progress gradients
GRADIENT_TRAIN: Tuple[Tuple[int, int, int], ...] = (
    AURORA_BLUE,
    AURORA_CYAN,
    AURORA_MINT,
    AURORA_AMBER,
    AURORA_ORANGE,
)
GRADIENT_MERGE: Tuple[Tuple[int, int, int], ...] = (
    AURORA_BLUE,
    AURORA_CYAN,
    AURORA_AMBER,
    AURORA_ORANGE,
    AURORA_RED,
)
GRADIENT_FIRE: Tuple[Tuple[int, int, int], ...] = (
    _rgb(255, 214, 112),
    AURORA_GOLD,
    AURORA_ORANGE,
    _rgb(235, 68, 68),
)

# ============================================================================
# Thread-safe output
# ============================================================================

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
# Display-width helpers (wide glyph aware)
# ============================================================================

# Codepoint ranges that render two cells wide in nearly every terminal
# (emoji with mandatory presentation + CJK/fullwidth via east_asian_width).
_WIDE_RANGES: Tuple[Tuple[int, int], ...] = (
    (0x1100, 0x115F),
    (0x2329, 0x232A),
    (0x2E80, 0x303E),
    (0x3041, 0x33FF),
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xA000, 0xA4CF),
    (0xAC00, 0xD7A3),
    (0xF900, 0xFAFF),
    (0xFE10, 0xFE19),
    (0xFE30, 0xFE6F),
    (0xFF00, 0xFF60),
    (0xFFE0, 0xFFE6),
    (0x1F000, 0x1F0FF),
    (0x1F100, 0x1F1FF),
    (0x1F200, 0x1F2FF),
    (0x1F300, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FAFF),
    (0x20000, 0x3FFFD),
)

# Emoji that sit below 0x1F000 but still render wide by default.
_WIDE_SINGLETONS: Tuple[int, ...] = (
    0x231A, 0x231B, 0x23E9, 0x23EA, 0x23EB, 0x23EC, 0x23F0, 0x23F3,
    0x25FD, 0x25FE, 0x2614, 0x2615, 0x2648, 0x2649, 0x2650, 0x2651,
    0x2652, 0x2653, 0x267F, 0x2693, 0x26A1, 0x26AA, 0x26AB, 0x26BD,
    0x26BE, 0x26C4, 0x26C5, 0x26CE, 0x26D4, 0x26EA, 0x26F2, 0x26F3,
    0x26F5, 0x26FA, 0x26FD, 0x2705, 0x270A, 0x270B, 0x2728, 0x274C,
    0x274E, 0x2753, 0x2754, 0x2755, 0x2757, 0x2795, 0x2796, 0x2797,
    0x27B0, 0x27BF, 0x2B1B, 0x2B1C, 0x2B50, 0x2B55,
)


def _char_width(ch: str) -> int:
    code = ord(ch)
    if unicodedata.combining(ch) or code in (0x200D,) or 0xFE00 <= code <= 0xFE0F:
        return 0
    for low, high in _WIDE_RANGES:
        if low <= code <= high:
            return 2
    if code in _WIDE_SINGLETONS:
        return 2
    return 1


def _display_width(text: str) -> int:
    """Rendered width of ``text`` ignoring ANSI escape sequences."""
    plain = _strip_ansi(text)
    return sum(_char_width(ch) for ch in plain)


def _pad(text: str, width: int, *, align: str = "left") -> str:
    """Pad ``text`` (which may contain ANSI codes) to a display width."""
    delta = width - _display_width(text)
    if delta <= 0:
        return text
    if align == "right":
        return " " * delta + text
    if align == "center":
        left = delta // 2
        return " " * left + text + " " * (delta - left)
    return text + " " * delta


def _truncate_width(text: str, width: int) -> str:
    """Truncate a plain string to at most ``width`` display cells."""
    if width <= 0:
        return ""
    if _display_width(text) <= width:
        return text

    accumulated = 0
    for index, ch in enumerate(text):
        accumulated += _char_width(ch)
        if accumulated > width - 1:
            return text[:index] + "…"
    return text


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


def _format_value(value: Any) -> str:
    """Type-aware scalar rendering used by summary cards."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return _format_metric(value)
    if isinstance(value, int):
        return f"{value:,}" if abs(value) >= 10_000 else str(value)
    return str(value)


def _value_color(key: str, value: Any) -> Tuple[Tuple[int, int, int], Optional[str]]:
    """Pick (rgb color, optional prefix format) for a metric-card value."""
    key_l = str(key).lower()

    if isinstance(value, bool):
        return (AURORA_SUCCESS if value else AURORA_DANGER), None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if any(token in key_l for token in ("rate", "ratio", "coverage", "improvement", "percent", "pct", "accuracy", "health")):
            number = _safe_float(value, 0.0)
            return AURORA_GOLD, f"{number * 100:.2f}%"
        return AURORA_GOLD, None

    text = str(value)
    if text.startswith(("http://", "https://")):
        return AURORA_CYAN, None
    return AURORA_TEXT, None


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


def _gradient_rule(width: int, stops: Sequence[Tuple[int, int, int]] = GRADIENT_FIRE) -> str:
    """Horizontal rule whose color flows across the terminal."""
    if not USE_COLOR or width <= 0:
        return "─" * width

    chars: List[str] = []
    for index in range(width):
        t = index / max(1, width - 1)
        color = _sample_gradient(stops, t)
        chars.append(f"{_fg(color)}━")
    return "".join(chars) + RESET


def _box_title(title: str, width: Optional[int] = None) -> str:
    width = width or _terminal_width()
    inner = max(8, width - 4)
    title_text = _truncate_width(_sanitize_text(title), inner)
    return (
        _c(DARK_GRAY, "╭" + "─" * (inner + 2) + "╮")
        + "\n"
        + _c(DARK_GRAY, "│ ")
        + _c(BOLD + NEON_CYAN, title_text)
        + _c(DARK_GRAY, " " * max(0, inner - _display_width(title_text)) + " │")
        + "\n"
        + _c(DARK_GRAY, "╰" + "─" * (inner + 2) + "╯")
    )


def _card_line(label: str, value: Any, width: int = 34) -> str:
    label_text = _truncate_width(_sanitize_text(label), width)
    return f"{_c(BOLD, label_text)} {_c(NEON_CYAN, value)}"


def _panel_open(width: int) -> str:
    return _paint(AURORA_RULE, "╭" + "─" * (width - 2) + "╮")


def _panel_close(width: int) -> str:
    return _paint(AURORA_RULE, "╰" + "─" * (width - 2) + "╯")


def _panel_divider(width: int) -> str:
    return _paint(AURORA_RULE, "├" + "─" * (width - 2) + "┤")


def _panel_line(content: str, width: int, *, pad_left: int = 2) -> str:
    """A boxed row: `│ content<pad> │` with display-width-correct padding.

    Content must already be wrapped/truncated to fit; the padding is purely
    display-width aware (ANSI codes and wide glyphs are measured correctly).
    """
    inner_width = width - 2 - pad_left - 1
    padding = " " * max(0, inner_width - _display_width(content))
    return (
        _paint(AURORA_RULE, "│")
        + " " * pad_left
        + content
        + padding
        + " "
        + _paint(AURORA_RULE, "│")
    )


def _panel_title(title: str, width: int, *, icon: str = "") -> str:
    label = f"{icon} {title}".strip()
    title_text = _pad(
        _c(BOLD + BRIGHT_WHITE, _truncate_width(_sanitize_text(label), width - 8)),
        width - 4,
        align="left",
    )
    return (
        _paint(AURORA_RULE, "│")
        + " "
        + title_text
        + " "
        + _paint(AURORA_RULE, "│")
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

    v1.2 renders every filled cell with its own color sampled from a
    multi-stop gradient (truecolor when available, 256-color otherwise),
    producing a continuous color flow instead of four flat bands.

    Backward compatible with the original signature.
    """
    try:
        value = float(progress)
    except (TypeError, ValueError):
        value = 0.0

    value = max(0.0, min(1.0, value))
    width = max(4, int(width))
    filled = int(round(value * width))
    stops = GRADIENT_MERGE if from_blue_to_orange else GRADIENT_TRAIN

    if not USE_COLOR:
        result = "█" * filled + "░" * (width - filled)
    else:
        cells: List[str] = []
        for index in range(width):
            t = index / max(1, width - 1)
            if index < filled:
                color = _sample_gradient(stops, t if filled > 1 else 0.0)
                cells.append(
                    f"{_bg(color)}{fill_char}{RESET}"
                )
            else:
                cells.append(f"{_bg(AURORA_TRACK)}{empty_char}{RESET}")
        result = "".join(cells)

    if show_percent:
        percent = _paint(AURORA_GOLD, f"{value * 100:6.2f}%")
        return f"[{result}] {percent}"

    return f"|{result}|"


def _thin_bar(progress: float, width: int = 30) -> str:
    value = max(0.0, min(1.0, float(progress)))
    filled = int(round(value * width))

    if not USE_COLOR:
        return "[" + "━" * filled + "─" * (width - filled) + "]"

    cells: List[str] = []
    for index in range(width):
        if index < filled:
            t = index / max(1, width - 1)
            color = _sample_gradient(GRADIENT_TRAIN, t if filled > 1 else 0.0)
            cells.append(f"{_fg(color)}━")
        else:
            cells.append(f"{_fg(AURORA_TRACK)}─")
    return "[" + "".join(cells) + RESET + "]"


def _milestone_color(value: float) -> Tuple[int, int, int]:
    if value >= 1.0:
        return AURORA_SUCCESS
    if value >= 0.67:
        return AURORA_MINT
    if value >= 0.34:
        return AURORA_CYAN
    return AURORA_GOLD


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


def _art_lines() -> List[str]:
    return [line for line in _FTRAIN_ART.strip("\n").splitlines()]


def fire_header(
    version: str = "1.1.0",
    subtitle: str = "Adaptive Training • Intelligent Merging • Captain AI",
) -> None:
    """Print the main FTRAIN startup banner with a fire-gradient wordmark."""
    width = min(_terminal_width(), 96)

    if USE_COLOR:
        _emit("")

        art_lines = _art_lines()
        count = max(1, len(art_lines) - 1)
        for index, line in enumerate(art_lines):
            t = index / count
            color = _sample_gradient(GRADIENT_FIRE, t)
            _emit(_paint(color, line))

        _emit(_gradient_rule(width))
        _emit("")
        _emit(
            _pad(
                _c(BOLD + BRIGHT_WHITE, "🔥 FTRAIN ENGINE ")
                + _c(BOLD + GOLD, f"v{version}")
                + _c(BOLD + BRIGHT_WHITE, " 🔥"),
                width,
                align="center",
            )
        )
        _emit("")
        _emit(
            _pad(
                _c(DIM + GRAY, _truncate_width(subtitle, width - 4)),
                width,
                align="center",
            )
        )
        _emit("")
        _emit(
            _pad(
                _gradient_rule(min(64, width), GRADIENT_TRAIN)
                + RESET
                + _c(DIM + GRAY, "  TRAIN • ADAPT • MERGE • EVOLVE  ")
                + _gradient_rule(min(64, width), GRADIENT_MERGE)
                + RESET,
                width,
                align="center",
            )
        )
        _emit("")
    else:
        _emit("")
        _emit(_FTRAIN_ART.rstrip())
        _emit("═" * width)
        _emit(f"🔥 FTRAIN ENGINE v{version} 🔥".center(width))
        _emit(subtitle.center(width))
        _emit("═" * width)
        _emit("TRAIN • ADAPT • MERGE • EVOLVE".center(width))
        _emit("")


# ============================================================================
# Stage / status UI
# ============================================================================

_STATUS_PILLS: Mapping[str, Tuple[str, Tuple[int, int, int]]] = {
    "done": ("DONE", AURORA_SUCCESS),
    "success": ("SUCCESS", AURORA_SUCCESS),
    "ok": ("DONE", AURORA_SUCCESS),
    "error": ("FAILED", AURORA_DANGER),
    "failed": ("FAILED", AURORA_DANGER),
    "warn": ("WARNING", AURORA_WARN),
    "warning": ("WARNING", AURORA_WARN),
    "running": ("RUNNING", AURORA_CYAN),
}


def _status_pill(status: str) -> str:
    """A solid-background status chip: `● RUNNING` on a colored pill."""
    key = str(status).strip().lower()
    label, color = _STATUS_PILLS.get(key, ("RUNNING", AURORA_CYAN))

    if not USE_COLOR:
        return f"[{label}]"

    fill = _bg(color)
    text = f"● {label}"
    # Dark text on bright pills reads best on both dark and light themes.
    return f"{fill}\033[38;5;16m{text}{RESET}"


def print_stage(
    title: str,
    message: str = "",
    icon: str = "🔥",
    status: str = "RUNNING",
) -> None:
    """Beautiful stage banner for core training/merging phases."""
    width = min(_terminal_width(), 96)

    header = f"{icon}  {_sanitize_text(title)}"
    pill = _status_pill(status)
    header_pad = max(0, width - 4 - _display_width(header) - _display_width(pill))

    _emit("")
    _emit(_panel_open(width))
    _emit(
        _paint(AURORA_RULE, "│ ")
        + _c(BOLD + GOLD, _truncate_width(header, width - 6))
        + " " * header_pad
        + " "
        + pill
        + " "
        + _paint(AURORA_RULE, "│")
    )
    if message:
        _emit(
            _panel_line(
                _c(GRAY, _truncate_width(_sanitize_text(message), width - 8)),
                width,
            )
        )
    _emit(_panel_close(width))


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
        "info": AURORA_CYAN,
        "success": AURORA_SUCCESS,
        "warning": AURORA_WARN,
        "error": AURORA_DANGER,
        "brain": AURORA_CYAN,
        "merge": AURORA_ORANGE,
        "train": AURORA_GOLD,
    }.get(level, AURORA_TEXT)

    _emit(f"{icon} {_paint(color, _sanitize_text(message))}")


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
    """Print an aligned training row while preserving the original API."""
    total = _safe_float(total_steps, 0.0) or 0.0
    current = _safe_float(step, 0.0) or 0.0
    progress = current / total if total > 0 else 0.0
    progress = max(0.0, min(1.0, progress))

    digits = max(len(str(int(total))), 1)
    step_text = f"{int(current):0{digits}d}/{int(total):0{digits}d}"

    loss_str = _format_metric(loss)
    val_str = _format_metric(val_loss)
    lr_str = _format_lr(lr)
    grad_str = _format_metric(grad_norm)
    epoch_str = _truncate_width(str(epoch), 8) if epoch is not None else None

    chunks = [
        f"🔥 {_c(BOLD, 'Step')} {_paint(AURORA_CYAN, step_text)}",
        f"Loss {_paint(AURORA_GOLD, loss_str)}",
        f"Val {_paint(AURORA_CYAN, val_str)}",
        f"LR {_paint(AURORA_BLUE, lr_str)}",
        f"Grad {_paint(AURORA_ORANGE, grad_str)}",
    ]

    if epoch_str is not None:
        chunks.append(f"Ep {_paint(AURORA_SUCCESS, epoch_str)}")

    if elapsed is not None:
        chunks.append(f"Time {_c(GRAY, _format_duration(elapsed))}")

    if tokens_per_second is not None:
        chunks.append(
            f"Tok/s {_paint(AURORA_SUCCESS, _human_number(tokens_per_second))}"
        )

    if captain_msg:
        chunks.append(
            f"🧠 {_paint(AURORA_CYAN, _truncate_width(_sanitize_text(captain_msg), 34))}"
        )

    line = (
        " ".join(chunks)
        + "  "
        + _thin_bar(progress, 18)
        + " "
        + _paint(_milestone_color(progress), f"{progress * 100:6.2f}%")
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
    """Dedicated v1.1 metrics row with v1.2 gradient bar."""
    parts = [
        f"{_c(BOLD, 'STEP')} {_paint(AURORA_CYAN, f'{step}/{total_steps}')}",
        f"L={_paint(AURORA_GOLD, _format_metric(loss))}",
        f"V={_paint(AURORA_CYAN, _format_metric(val_loss))}",
        f"LR={_paint(AURORA_BLUE, _format_lr(lr))}",
        f"G={_paint(AURORA_ORANGE, _format_metric(grad_norm))}",
    ]

    if grad_health is not None:
        parts.append(
            f"GH={_paint(AURORA_SUCCESS, f'{float(grad_health):.2%}')}"
        )

    if throughput is not None:
        parts.append(
            f"T={_paint(AURORA_SUCCESS, f'{throughput:.1f}/s')}"
        )

    if captain:
        parts.append(
            f"🧠 {_paint(AURORA_CYAN, _truncate_width(_sanitize_text(captain), 30))}"
        )

    progress = (
        step / total_steps
        if total_steps
        else 0.0
    )
    progress = max(0.0, min(1.0, progress))

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
        f"🧩 {_c(BOLD + BRIGHT_WHITE, 'MERGE')}",
        _paint(_milestone_color(progress), f"{progress * 100:6.2f}%"),
        gradient_bar(
            progress,
            24,
            from_blue_to_orange=True,
        ),
    ]

    if matched is not None:
        parts.append(f"M:{_paint(AURORA_SUCCESS, matched)}")

    if projected is not None:
        parts.append(f"P:{_paint(AURORA_GOLD, projected)}")

    if rejected is not None:
        parts.append(f"R:{_paint(AURORA_DANGER, rejected)}")

    if strategy:
        parts.append(
            f"[{_paint(AURORA_CYAN, _truncate_width(_sanitize_text(strategy), 18))}]"
        )

    if message:
        parts.append(
            f"{_c(GRAY, '(' + _truncate(_sanitize_text(message), 36) + ')')}"
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

_URL_RE = re.compile(r"(https?://[^\s)\]»]+)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_CODE_RE = re.compile(r"`([^`]+)`")


def _style_inline(text: str) -> str:
    """Apply markdown-ish inline styling when color is available."""
    if not USE_COLOR:
        return text

    text = _BOLD_RE.sub(lambda m: f"{BOLD}{m.group(1)}{RESET}", text)
    text = _CODE_RE.sub(lambda m: f"{GOLD}{m.group(1)}{RESET}", text)
    text = _URL_RE.sub(
        lambda m: f"{NEON_CYAN}{UNDERLINE}{m.group(1)}{RESET}", text
    )
    return text


def _captain_body_lines(report: str, width: int) -> List[str]:
    """Convert a Captain report into styled, wrapped box lines."""
    inner = max(20, width - 6)
    lines: List[str] = []

    for raw_line in str(report).splitlines() or [""]:
        line = _sanitize_text(raw_line).rstrip()

        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue

        # Headings: #, ##, ###
        heading = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if heading:
            text = _truncate_width(heading.group(2).strip(), inner)
            if USE_COLOR:
                lines.append(f"{BOLD}{NEON_CYAN}{text}{RESET}")
                lines.append(_paint(AURORA_RULE, "─" * min(inner, _display_width(text) + 2)))
            else:
                lines.append(text)
            continue

        # Bullets: -, *, •
        bullet = re.match(r"^([-*•])\s+(.*)$", stripped)
        if bullet:
            body = _style_inline(bullet.group(2))
            glyph = _paint(AURORA_ORANGE, "▸") if USE_COLOR else "-"
            wrapped = _wrap_plain(body, inner - 2)
            for index, segment in enumerate(wrapped):
                lines.append(f"{glyph} {segment}" if index == 0 else f"  {segment}")
            continue

        # Numbered items
        numbered = re.match(r"^(\d+)[.)]\s+(.*)$", stripped)
        if numbered:
            body = _style_inline(numbered.group(2))
            marker = _paint(AURORA_GOLD, f"{numbered.group(1)}.") if USE_COLOR else f"{numbered.group(1)}."
            wrapped = _wrap_plain(body, inner - 3)
            for index, segment in enumerate(wrapped):
                lines.append(f"{marker} {segment}" if index == 0 else f"   {segment}")
            continue

        lines.extend(_wrap_plain(_style_inline(stripped), inner))

    return lines or [""]


def _wrap_plain(text: str, width: int) -> List[str]:
    """Wrap styled text by measuring display width, preserving ANSI codes."""
    words = text.split(" ")
    wrapped: List[str] = []
    current = ""

    def width_of(value: str) -> int:
        return _display_width(value)

    for word in words:
        candidate = f"{current} {word}".strip()
        if width_of(candidate) <= width or not current:
            current = candidate
        else:
            wrapped.append(current)
            current = word

    if current:
        wrapped.append(current)

    return wrapped or [""]


def print_captain_report(report: str) -> None:
    """Pretty-print the Captain's textual analysis with markdown styling."""
    width = min(_terminal_width(), 90)

    _emit("")
    _emit(_panel_open(width))
    _emit(
        _paint(AURORA_RULE, "│")
        + _pad(
            _c(BOLD + NEON_CYAN, "🧠 CAPTAIN ANALYSIS"),
            width - 2,
            align="center",
        )
        + _paint(AURORA_RULE, "│")
    )
    _emit(_panel_divider(width))

    for line in _captain_body_lines(report, width):
        _emit(_panel_line(line, width))

    _emit(_panel_close(width))
    _emit("")


def print_captain_advice(
    action: str,
    multiplier: Optional[float] = None,
    reason: Optional[str] = None,
) -> None:
    """Compact v1.1 Captain decision card."""
    _emit(
        f"🧠 {_c(BOLD + NEON_CYAN, 'CAPTAIN')} "
        f"{_paint(AURORA_GOLD, _sanitize_text(action))}"
    )

    if multiplier is not None:
        value = _safe_float(multiplier, 1.0)
        if value > 1.0:
            color = AURORA_SUCCESS
        elif value < 1.0:
            color = AURORA_WARN
        else:
            color = AURORA_TEXT
        _emit(
            f"   LR multiplier: {_paint(color, f'x{value:.3f}')}"
        )

    if reason:
        _emit(
            f"   {_c(GRAY, _truncate(_sanitize_text(reason), _terminal_width() - 6))}"
        )


# ============================================================================
# Final summaries
# ============================================================================

def _leader(label: str, value: str, width: int) -> str:
    """A metric row with dotted leaders between label and value.

    Label is padded while plain, then styled, so column math stays correct.
    """
    label_width = 27
    label_plain = _truncate_width(label, label_width)
    value_text = _truncate_width(str(value), max(8, width - label_width - 8))
    value_width = _display_width(value_text)
    dots_width = max(3, width - 4 - label_width - value_width)

    return (
        f"  {_c(BOLD, f'{label_plain:<{label_width}}')}"
        + _c(DARK_GRAY, "·" * dots_width)
        + " "
        + value_text
    )


def print_final_summary(stats: Dict[str, Any]) -> None:
    """Render a beautiful final process summary."""
    width = min(_terminal_width(), 96)

    _emit("")
    _emit(_c(DARK_GRAY, "╔" + "═" * (width - 2) + "╗"))
    _emit(
        _c(DARK_GRAY, "║ ")
        + _pad(
            _c(BOLD + NEON_GREEN, "✅ FTRAIN PROCESS COMPLETED"),
            width - 4,
            align="center",
        )
        + _c(DARK_GRAY, " ║")
    )
    _emit(_c(DARK_GRAY, "╠" + "═" * (width - 2) + "╣"))

    for key, value in stats.items():
        label = str(key).replace("_", " ").title()
        label_plain = _truncate_width(label, 27)

        if isinstance(value, float):
            key_l = key.lower()
            if any(token in key_l for token in ("improvement", "rate", "ratio", "coverage")):
                display = f"{value * 100:.2f}%"
            else:
                display = _format_metric(value)
            colored = _paint(AURORA_GOLD, display)
        elif isinstance(value, bool):
            colored = _paint(
                AURORA_SUCCESS if value else AURORA_DANGER,
                "yes" if value else "no",
            )
        elif isinstance(value, int):
            colored = _paint(AURORA_TEXT, f"{value:,}")
        else:
            display = _sanitize_text(value)
            if "http://" in display or "https://" in display:
                colored = _c(NEON_CYAN + UNDERLINE, display)
            else:
                colored = _c(WHITE, display)

        row = (
            _c(DARK_GRAY, "║ ")
            + _c(BOLD, f"{label_plain:<27}")
            + " "
            + _c(DARK_GRAY, "·" * max(3, 28 - len(label_plain)))
            + " "
            + colored
        )
        _emit(row)

    _emit(_c(DARK_GRAY, "╚" + "═" * (width - 2) + "╝"))
    _emit("")


def print_metric_summary(
    title: str,
    metrics: Mapping[str, Any],
    *,
    icon: str = "📊",
) -> None:
    """Generic metric-card renderer with dotted leaders."""
    width = min(_terminal_width(), 90)

    _emit("")
    _emit(_panel_open(width))
    _emit(_panel_title(title, width, icon=icon))
    _emit(_panel_divider(width))

    for key, value in metrics.items():
        label = str(key).replace("_", " ").title()
        label_text = _truncate_width(label, 25)

        color, formatted = _value_color(key, value)
        if formatted is not None:
            display = formatted
        elif isinstance(value, float):
            if "rate" in key.lower() or "ratio" in key.lower():
                display = f"{value:.2%}"
            else:
                display = f"{value:.5f}"
        else:
            display = _format_value(value)

        row = (
            f"  {_c(BOLD, label_text)}"
            + " "
            + _c(DARK_GRAY, "·" * max(3, 26 - len(label_text)))
            + " "
            + _paint(color, _truncate_width(_sanitize_text(display), width - 36))
        )

        _emit(_panel_line(row, width))

    _emit(_panel_close(width))


# ============================================================================
# Loading bar
# ============================================================================

SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class LoadingBar:
    """
    Thread-safe loading/progress bar.

    Backward compatible with:
        LoadingBar(message="...", real_progress=True)
        start()
        update(current, total)
        done()

    ``real_progress=False`` creates an animated indeterminate bar with a
    braille spinner and a live elapsed-time readout.
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
        self._start_time: Optional[float] = None
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
        self._start_time = time.monotonic()
        self.stop_event.clear()

        if self.real:
            _rewrite(
                f"📦 {_c(BOLD, self.message)} "
                f"{gradient_bar(0.0, self.width, from_blue_to_orange=True)} "
                f"{_paint(AURORA_MUTED, '0.00%')}"
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
        frame = 0

        while not self.stop_event.is_set():
            frame += 1
            position += direction

            if position >= self.width - 2:
                position = self.width - 2
                direction = -1

            elif position <= 0:
                position = 0
                direction = 1

            elapsed = max(0.0, time.monotonic() - (self._start_time or time.monotonic()))

            if USE_COLOR:
                trail: List[str] = []
                for index in range(self.width):
                    if index == position:
                        trail.append(f"{_bg(AURORA_ORANGE)}  ")
                    elif abs(index - position) == 1:
                        trail.append(f"{_bg(_blend(AURORA_ORANGE, AURORA_TRACK, 0.5))}  ")
                    else:
                        trail.append(f"{_bg(AURORA_TRACK)}  ")
                bar = "".join(trail) + RESET
            else:
                bar = (
                    "["
                    + " " * position
                    + "██"
                    + " " * max(0, self.width - position - 2)
                    + "]"
                )

            spinner = SPINNER_FRAMES[frame % len(SPINNER_FRAMES)]
            spinner_part = _paint(AURORA_GOLD, spinner) if USE_COLOR else ""
            elapsed_part = (
                _paint(AURORA_MUTED, _format_duration(elapsed))
                if USE_COLOR
                else _format_duration(elapsed)
            )

            _rewrite(
                f"📦 {_c(BOLD, self.message)} {bar} "
                f"{spinner_part} {elapsed_part}"
            )

            self.stop_event.wait(self.update_interval)

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

        elapsed = max(0.0, time.monotonic() - (self._start_time or time.monotonic()))
        _rewrite(
            f"📦 {_c(BOLD, self.message)} "
            f"{gradient_bar(progress, self.width, from_blue_to_orange=True)} "
            f"{_paint(_milestone_color(progress), f'{progress * 100:6.2f}%')} "
            f"{_paint(AURORA_MUTED, _format_duration(elapsed))}"
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

        elapsed = max(0.0, time.monotonic() - (self._start_time or time.monotonic()))
        duration = (
            f" {_paint(AURORA_MUTED, '(' + _format_duration(elapsed) + ')')}"
            if USE_COLOR
            else f" ({_format_duration(elapsed)})"
        )

        _rewrite(
            f"📦 {_c(BOLD, self.message)} "
            f"{gradient_bar(1.0, self.width, from_blue_to_orange=True)} "
            f"{_c(BOLD + NEON_GREEN, '100.00%')} ✅{duration}"
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

def print_divider(label: str = "", char: str = "─") -> None:
    """A dim divider line, optionally centered around a label. (New in v1.2.)"""
    width = _terminal_width()
    label_text = f" {_sanitize_text(label)} " if label else ""

    if not label_text:
        _emit(_rule(width, char))
        return

    label_width = _display_width(label_text)
    side = max(2, (width - label_width) // 2)
    _emit(
        _c(DARK_GRAY, char * side)
        + _c(DIM + GRAY, label_text)
        + _c(DARK_GRAY, char * max(2, width - side - label_width))
    )


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
        metrics["Model"] = _truncate_width(_sanitize_text(model_name), 58)
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
    "BOLD",
    "DIM",
    "ITALIC",
    "UNDERLINE",
    "USE_COLOR",
    "USE_TRUECOLOR",
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
    "print_divider",
    "LoadingBar",
]
