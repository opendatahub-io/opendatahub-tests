"""
OpenDataHub logging utilities using structlog.

Structlog renders plain text for console-friendly output. JSON formatting for the
log file is handled by ThirdPartyJSONFormatter applied on the file handler in setup_logging().

When --readable-logs is passed, both console and file output are human-readable text.

Example:
    from utilities.opendatahub_logger import get_logger

    logger = get_logger("myapp")
    logger.info("User logged in", user_id=123)
    # Console: 2024-01-01 myapp INFO User logged in [user_id=123]
    # File:    {"timestamp": "...", "logger": "myapp", "level": "info", "event": "User logged in [user_id=123]"}
"""

import inspect
import json
import logging
import traceback
from datetime import UTC, datetime
from typing import Any

import structlog


class DuplicateFilter:
    """Filter duplicate log messages."""

    def __init__(self) -> None:
        self.msgs: set[str] = set()

    def filter(self, record: logging.LogRecord) -> bool:
        msg = str(record.getMessage())
        if msg not in self.msgs:
            self.msgs.add(msg)
            return True
        return False


_FG_CODES = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
}

_BG_CODES = {
    "black": "\033[40m",
    "red": "\033[41m",
    "green": "\033[42m",
    "yellow": "\033[43m",
    "blue": "\033[44m",
    "magenta": "\033[45m",
    "cyan": "\033[46m",
    "white": "\033[47m",
}

_RESET = "\033[0m"


class WrapperLogFormatter(logging.Formatter):
    """
    Formatter with color support for console output.
    Compatible with python-simple-logger's WrapperLogFormatter.
    """

    def __init__(
        self,
        fmt: str | None = None,
        log_colors: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(fmt, **kwargs)
        self.log_colors = log_colors or {}

    def format(self, record: logging.LogRecord) -> str:
        if self.log_colors and record.levelname in self.log_colors:
            record.log_color = self._get_color_code(color=self.log_colors[record.levelname])
            record.reset = _RESET
        else:
            record.log_color = ""
            record.reset = ""
        return super().format(record)

    def _get_color_code(self, color: str) -> str:
        """Convert color name to ANSI escape code."""
        if "," in color:
            fg_color, bg_color = color.split(",", 1)
            bg_code = _BG_CODES.get(bg_color[3:], "") if bg_color.startswith("bg_") else ""
            return _FG_CODES.get(fg_color, "") + bg_code

        return _FG_CODES.get(color, "")


class JSONOnlyFormatter(logging.Formatter):
    """Custom formatter that outputs only the message (for pure JSON output)"""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class ThirdPartyJSONFormatter(logging.Formatter):
    """Custom formatter that converts third-party logging to JSON format.

    If the message is already valid JSON (e.g. from structlog), it is passed through as-is.
    """

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        try:
            json.loads(msg)
            return msg
        except json.JSONDecodeError, TypeError:
            return json.dumps({
                "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "logger": record.name,
                "level": record.levelname.lower(),
                "event": msg,
                "filename": record.pathname.split("/")[-1] if record.pathname else "",
                "lineno": str(record.lineno),
            })


_initialized = False
_human_readable = False


def set_human_readable(enabled: bool) -> None:
    """Set log output format and re-initialize structlog if already configured."""
    global _human_readable, _initialized
    _human_readable = enabled
    _initialized = False
    _initialize()


_LEVEL_COLORS = {
    "DEBUG": _FG_CODES["cyan"],
    "INFO": _FG_CODES["green"],
    "WARNING": _FG_CODES["yellow"],
    "ERROR": _FG_CODES["red"],
    "CRITICAL": _FG_CODES["red"] + _BG_CODES["white"],
}


class ThirdPartyHumanReadableFormatter(logging.Formatter):
    """Formatter for third-party logs in human-readable mode with colors and source location."""

    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.fromtimestamp(record.created, tz=UTC).isoformat()
        color = _LEVEL_COLORS.get(record.levelname, "")
        reset = _RESET if color else ""
        filename = record.pathname.rsplit("/", 1)[-1] if record.pathname else ""
        msg = record.getMessage()
        return f"{timestamp} {record.name} {color}{record.levelname}{reset} {msg} ({filename}:{record.lineno})"


def _plain_text_renderer(_logger: Any, _method_name: str, event_dict: dict[str, Any]) -> str:
    """Render structlog events as plain text for console-friendly output."""
    event = event_dict.pop("event", "")
    # Remove fields already present on the log record (added by structlog processors)
    for key in ("logger", "level", "timestamp"):
        event_dict.pop(key, None)
    if event_dict:
        extras = " ".join(f"{k}={v}" for k, v in event_dict.items())
        return f"{event} [{extras}]"
    return str(event)


def _initialize() -> None:
    """One-time setup for structlog."""
    global _initialized
    if _initialized:
        return

    if _human_readable:
        processors = [
            structlog.processors.KeyValueRenderer(key_order=["event"]),
        ]
    else:
        processors = [
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO", utc=True),
            _plain_text_renderer,
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )

    _initialized = True


class StructlogWrapper:
    """Wrapper for structlog logger to provide simple_logger-compatible interface"""

    def __init__(self, name: str) -> None:
        self.name = name
        _initialize()
        self._logger = structlog.get_logger(name=name)

    def _log(self, level: str, msg: Any, *args: Any, **kwargs: Any) -> None:
        msg_str = str(msg)
        if args:
            msg_str = msg_str % args

        if _human_readable:
            std_logger = logging.getLogger(self.name)
            log_method = getattr(std_logger, level.lower())
            extra_str = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
            log_method(f"{msg_str} {extra_str}" if extra_str else msg_str, stacklevel=3)  # noqa: FCN001
        else:
            log_method = getattr(self._logger, level.lower())
            log_method(event=msg_str, **kwargs)

    def info(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log("info", msg, *args, **kwargs)  # noqa: FCN001

    def debug(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log("debug", msg, *args, **kwargs)  # noqa: FCN001

    def warning(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log("warning", msg, *args, **kwargs)  # noqa: FCN001

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log("error", msg, *args, **kwargs)  # noqa: FCN001

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log("critical", msg, *args, **kwargs)  # noqa: FCN001

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        kwargs["exception"] = traceback.format_exc()
        self._log("error", msg, *args, **kwargs)  # noqa: FCN001


def get_logger(name: str | None = None) -> StructlogWrapper:
    """
    Get a structlog logger instance.

    Args:
        name: Logger name (defaults to caller's module name)

    Returns:
        StructlogWrapper instance
    """
    if name is None:
        frame = inspect.currentframe()
        try:
            caller_frame = frame.f_back if frame else None
            name = caller_frame.f_globals.get("__name__", "unknown") if caller_frame else "unknown"
        finally:
            del frame

    return StructlogWrapper(name=name or "unknown")
