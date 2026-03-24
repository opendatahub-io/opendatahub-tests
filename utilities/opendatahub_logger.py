"""
OpenDataHub logging utilities using structlog with third-party logging integration.

This module provides structured JSON logging using structlog with automatic
third-party library logging integration.

Example:
    from utilities.opendatahub_logger import get_logger

    logger = get_logger("myapp")
    logger.info("User logged in", user_id=123)
    # Output: {"timestamp": "...", "logger": "myapp", "level": "info", "event": "User logged in", "user_id": 123}
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


def _initialize() -> None:
    """One-time setup for structlog and third-party logging."""
    global _initialized
    if _initialized:
        return

    structlog.configure(
        processors=[
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="ISO", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Patch addHandler so new loggers get JSON formatting
    original_add_handler = logging.Logger.addHandler

    def patched_add_handler(self: logging.Logger, hdlr: logging.Handler) -> None:
        if not isinstance(hdlr.formatter, (ThirdPartyJSONFormatter, JSONOnlyFormatter)):
            hdlr.setFormatter(fmt=ThirdPartyJSONFormatter())
        original_add_handler(self, hdlr)  # noqa: FCN001

    logging.Logger.addHandler = patched_add_handler  # type: ignore[method-assign]

    # Apply JSON formatter to all existing handlers on all loggers
    all_loggers = [logging.getLogger()] + [
        logger for logger in logging.root.manager.loggerDict.values() if isinstance(logger, logging.Logger)
    ]
    for logger in all_loggers:
        for handler in logger.handlers:
            if isinstance(handler.formatter, (logging.Formatter, type(None))):
                handler.setFormatter(fmt=ThirdPartyJSONFormatter())

    _initialized = True


class StructlogWrapper:
    """Wrapper for structlog logger to provide simple_logger-compatible interface"""

    def __init__(self, name: str) -> None:
        self.name = name
        _initialize()
        self._logger = structlog.get_logger(name=name)

        underlying_logger = logging.getLogger(name)
        for handler in underlying_logger.handlers:
            if isinstance(handler.formatter, (logging.Formatter, type(None))):
                handler.setFormatter(fmt=JSONOnlyFormatter())

    def _log(self, level: str, msg: Any, *args: Any, **kwargs: Any) -> None:
        msg_str = str(msg)
        if args:
            msg_str = msg_str % args

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
