import logging
import os
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# ANSI codes
# ---------------------------------------------------------------------------

_R   = "\033[0m"
_B   = "\033[1m"
_DIM = "\033[2m"

_RED    = "\033[91m"
_YELLOW = "\033[93m"
_GREEN  = "\033[92m"
_CYAN   = "\033[96m"
_BLUE   = "\033[94m"
_WHITE  = "\033[97m"
_GREY   = "\033[90m"


# ---------------------------------------------------------------------------
# Level → color mapping
# ---------------------------------------------------------------------------

_LEVEL_COLOR = {
    logging.DEBUG:    _GREY,
    logging.INFO:     _WHITE,
    logging.WARNING:  _YELLOW,
    logging.ERROR:    _RED + _B,
    logging.CRITICAL: _RED + _B,
}


def _msg_color(msg: str) -> str:
    """Pick a highlight color based on message content."""
    if any(c in msg for c in ("╔", "╚", "╠")):
        return _CYAN + _B
    if any(c in msg for c in ("┌", "└", "═" * 3)):
        return _CYAN
    if "Phase 1/2" in msg or "Phase 2/2" in msg:
        return _BLUE + _B
    if "Summary" in msg or "accuracy" in msg or "complete" in msg.lower():
        return _GREEN + _B
    if any(k in msg for k in ("ready", "saved", "Done", "done")):
        return _GREEN
    if any(k in msg for k in ("Terminat", "Killing", "Killed", "free")):
        return _DIM + _YELLOW
    if any(k in msg for k in ("already in use", "GPU memory", "OOM")):
        return _YELLOW
    return ""


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        level_color = _LEVEL_COLOR.get(record.levelno, "")
        if record.levelno == logging.INFO:
            content_color = _msg_color(record.getMessage())
            color = content_color if content_color else level_color
        else:
            color = level_color
        return f"{color}{base}{_R}" if color else base


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

_FILE_FMT = logging.Formatter(
    fmt="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def setup_logging(
    level: int = logging.INFO,
    log_file: str | None = None,
    verbose: bool = False,
) -> None:
    """Configure root logger with colored, timestamped output.

    Args:
        level    : Base log level (overridden to DEBUG when verbose=True).
        log_file : Script name (no extension). When set, writes a plain-text
                   log to logs/<name>_<YYYYMMDD_HHMMSS>.log in the CWD.
        verbose  : Shorthand to set level=DEBUG.
    """
    if verbose:
        level = logging.DEBUG

    console_fmt = ColoredFormatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(console_fmt)
    handlers: list[logging.Handler] = [console]

    if log_file:
        logs_dir = Path.cwd() / "logs"
        logs_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = logs_dir / f"{log_file}_{timestamp}.log"
        file_handler = logging.FileHandler(log_path, "w")
        file_handler.setFormatter(_FILE_FMT)
        handlers.append(file_handler)

    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)
