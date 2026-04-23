import logging


# ---------------------------------------------------------------------------
# ANSI codes
# ---------------------------------------------------------------------------

_R  = "\033[0m"    # reset
_B  = "\033[1m"    # bold
_DIM = "\033[2m"   # dim

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
    # Run-comparison banners  (╔══  ╚══  ┌──  └──)
    if any(c in msg for c in ("╔", "╚", "╠")):
        return _CYAN + _B
    if any(c in msg for c in ("┌", "└", "═" * 3)):
        return _CYAN
    # Phase headers
    if "Phase 1/2" in msg or "Phase 2/2" in msg:
        return _BLUE + _B
    # Summary / accuracy lines
    if "Summary" in msg or "accuracy" in msg or "complete" in msg.lower():
        return _GREEN + _B
    # Positive milestones
    if any(k in msg for k in ("ready", "saved", "Done", "done")):
        return _GREEN
    # Cleanup / teardown
    if any(k in msg for k in ("Terminat", "Killing", "Killed", "free")):
        return _DIM + _YELLOW
    # Warnings embedded in INFO (port in use, GPU memory)
    if any(k in msg for k in ("already in use", "GPU memory", "OOM")):
        return _YELLOW
    return ""   # use level default


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
# Public helpers
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure root logger with colored, timestamped output to stdout.
    Call once at the top of your script or notebook before running anything.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    ))
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()
    root.addHandler(handler)


def init_logging(log_file: str = None, verbose: bool = False) -> logging.Logger:
    import os
    level = logging.DEBUG if verbose else logging.INFO

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        fmt="%(asctime)s : %(levelname)s : [%(filename)s:%(lineno)s - %(funcName)s()] : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    handlers: list[logging.Handler] = [handler]
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        handlers.append(logging.FileHandler(log_file, "w"))

    logging.basicConfig(level=level, handlers=handlers, force=True)
    return logging.getLogger()
