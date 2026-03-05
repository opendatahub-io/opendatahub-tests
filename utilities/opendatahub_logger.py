"""
OpenDataHub logging utilities using structlog with third-party logging integration.

This module provides:
1. Structured JSON logging using structlog
2. Backward compatibility with simple_logger API
3. Automatic third-party library logging integration

Third-Party Integration:
When you call get_logger() with configure_third_party=True (default), this module will:
- Configure all existing loggers to output JSON
- Patch logging.getLogger() to auto-configure new loggers
- Ensure consistent JSON output from all libraries

Features:
- Pure JSON output for your code and third-party libraries
- Automatic logger patching for new third-party loggers
- Backward compatible with existing simple_logger API

Example:
    from utilities.opendatahub_logger import get_logger

    # Your logger (structlog-based)
    logger = get_logger("myapp")
    logger.info("User logged in", user_id=123)
    # Output: {"timestamp": "...", "logger": "myapp", "level": "info", "event": "User logged in", "user_id": 123}

    # Third-party libraries automatically output JSON
    import requests  # or any library
    # Their logging will automatically be JSON formatted
"""

import inspect
import logging
from datetime import UTC
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


class WrapperLogFormatter(logging.Formatter):
    """
    Formatter with color support for console output.
    Compatible with python-simple-logger's WrapperLogFormatter.
    """

    def __init__(
        self,
        fmt: str | None = None,
        log_colors: dict[str, str] | None = None,
        secondary_log_colors: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(fmt, **kwargs)
        self.log_colors = log_colors or {}
        self.secondary_log_colors = secondary_log_colors or {}

    def format(self, record: logging.LogRecord) -> str:
        # Add color fields to the record for format string interpolation
        if self.log_colors and record.levelname in self.log_colors:
            record.log_color = self._get_color_code(color=self.log_colors[record.levelname])
            record.reset = self._get_reset_code()
        else:
            record.log_color = ""
            record.reset = ""
        return super().format(record)

    def _get_color_code(self, color: str) -> str:
        """Convert color name to ANSI escape code."""
        color_codes = {
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            "reset": "\033[0m",
        }

        # Handle background colors (e.g., "red,bg_white")
        if "," in color:
            fg_color, bg_color = color.split(",", 1)
            bg_code = ""
            if bg_color.startswith("bg_"):
                bg_name = bg_color[3:]  # Remove "bg_" prefix
                bg_codes = {
                    "black": "\033[40m",
                    "red": "\033[41m",
                    "green": "\033[42m",
                    "yellow": "\033[43m",
                    "blue": "\033[44m",
                    "magenta": "\033[45m",
                    "cyan": "\033[46m",
                    "white": "\033[47m",
                }
                bg_code = bg_codes.get(bg_name, "")
            return color_codes.get(fg_color, "") + bg_code

        return color_codes.get(color, "")

    def _get_reset_code(self) -> str:
        """Get ANSI reset code."""
        return "\033[0m"


class JSONOnlyFormatter(logging.Formatter):
    """Custom formatter that outputs only the message (for pure JSON output)"""

    def format(self, record: logging.LogRecord) -> str:
        return record.getMessage()


class ThirdPartyJSONFormatter(logging.Formatter):
    """Custom formatter that converts third-party logging to JSON format"""

    def format(self, record: logging.LogRecord) -> str:
        import json
        from datetime import datetime

        # Create JSON log entry for third-party libraries
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "logger": record.name,
            "level": record.levelname.lower(),
            "event": record.getMessage(),
        }

        # Add any extra fields from the record
        if hasattr(record, "pathname"):
            log_entry["filename"] = record.pathname.split("/")[-1] if record.pathname else ""
        if hasattr(record, "lineno"):
            log_entry["lineno"] = str(record.lineno)

        try:
            return json.dumps(log_entry)
        except (TypeError, ValueError):
            # Fallback for non-serializable data
            fallback_entry = {
                "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                "logger": record.name,
                "level": record.levelname.lower(),
                "event": str(record.getMessage()),
            }
            return json.dumps(fallback_entry)


class StructlogWrapper:
    """Wrapper for structlog logger to provide simple_logger-compatible interface"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.handlers: list[logging.Handler] = []  # For compatibility

        # Always use JSON format with structlog
        processors = [
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO", utc=True),
            structlog.processors.JSONRenderer(),
        ]

        # Use standard logging integration to ensure pytest log capture
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.stdlib.BoundLogger,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        self._logger = structlog.get_logger(name=name)

        # Configure the underlying logger to use JSONOnlyFormatter for pure JSON output
        underlying_logger = logging.getLogger(name)
        for handler in underlying_logger.handlers:
            if isinstance(handler.formatter, (logging.Formatter, type(None))):
                handler.setFormatter(fmt=JSONOnlyFormatter())

    def _log(self, level: str, msg: Any, *args: Any, **kwargs: Any) -> None:
        """Common logging method."""
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

    def warn(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self.warning(msg, *args, **kwargs)  # noqa: FCN001

    def error(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log("error", msg, *args, **kwargs)  # noqa: FCN001

    def critical(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        self._log("critical", msg, *args, **kwargs)  # noqa: FCN001

    def exception(self, msg: Any, *args: Any, **kwargs: Any) -> None:
        import traceback

        kwargs["exception"] = traceback.format_exc()
        self._log("error", msg, *args, **kwargs)  # noqa: FCN001

    def setLevel(self, level: int | str) -> None:
        # For compatibility - structlog doesn't filter by level in this setup
        pass

    def addHandler(self, hdlr: logging.Handler) -> None:
        # For compatibility - keep track of handlers
        self.handlers.append(hdlr)

    def addFilter(self, flt: Any) -> None:
        # For compatibility
        pass


def configure_third_party_logging() -> None:
    """
    Configure all third-party loggers to use JSON formatting.
    This makes external libraries produce consistent JSON logs.
    """
    # Configure root logger to use JSON formatting
    root_logger = logging.getLogger()

    # Apply JSON formatter to all existing handlers
    for handler in root_logger.handlers:
        if isinstance(handler.formatter, (logging.Formatter, type(None))):
            handler.setFormatter(fmt=ThirdPartyJSONFormatter())

    # Configure all existing loggers to use JSON formatting
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            for handler in logger.handlers:
                if isinstance(handler.formatter, (logging.Formatter, type(None))):
                    handler.setFormatter(fmt=ThirdPartyJSONFormatter())


def setup_global_json_logging() -> None:
    """
    Set up global logging configuration to ensure all new loggers use JSON format.
    This is a one-time setup that affects all future logger creation.
    """
    # Store original getLogger function
    original_getLogger = logging.getLogger

    def patched_getLogger(name: str | None = None) -> logging.Logger:
        """Patched getLogger that automatically applies JSON formatting to new loggers."""
        logger = original_getLogger(name=name)

        # Skip if this is our structlog wrapper or if already configured
        if (name and ("opendatahub_logger" in name or "structlog" in name)) or hasattr(logger, "_json_configured"):
            return logger

        # Apply JSON formatter to any handlers this logger might have
        for handler in logger.handlers:
            if isinstance(handler.formatter, (logging.Formatter, type(None))):
                handler.setFormatter(fmt=ThirdPartyJSONFormatter())

        # Mark as configured to avoid repeated processing
        logger._json_configured = True  # type: ignore[attr-defined]
        return logger

    # Patch the logging.getLogger function
    logging.getLogger = patched_getLogger


# Global flag to ensure we only set up global logging once
_global_logging_configured = False


def get_logger(
    name: str | None = None,
    level: str = "INFO",
    log_to_file: bool = True,
    log_file: str | None = None,
    filename: str | None = None,  # Compatibility with external libraries
    log_to_console: bool = True,
    json_format: bool = True,  # Kept for backward compatibility but ignored
    configure_third_party: bool = True,  # New parameter to control third-party logging
) -> StructlogWrapper:
    """
    Get a structlog logger instance compatible with simple_logger.logger.get_logger()

    Always returns JSON formatted logs and optionally configures third-party logging.

    Args:
        name: Logger name (defaults to caller's module name)
        level: Log level
        log_to_file: Whether to log to file
        log_file: Log file path
        filename: Alternative name for log_file (external library compatibility)
        log_to_console: Whether to log to console
        json_format: Kept for backward compatibility, always uses JSON format
        configure_third_party: Whether to configure third-party loggers for JSON output

    Returns:
        StructlogWrapper instance that behaves like logging.Logger
    """
    global _global_logging_configured

    if name is None:
        # Get the caller's module name
        frame = inspect.currentframe()
        try:
            if frame is not None:
                caller_frame = frame.f_back
                if caller_frame is not None:
                    name = caller_frame.f_globals.get("__name__", "unknown")
                else:
                    name = "unknown"
            else:
                name = "unknown"
        finally:
            if frame is not None:
                del frame

    # Configure third-party logging if requested (one-time setup)
    if configure_third_party and not _global_logging_configured:
        setup_global_json_logging()
        configure_third_party_logging()
        _global_logging_configured = True

    return StructlogWrapper(name=name or "unknown")


def test_third_party_logging() -> None:
    """
    Test function to demonstrate third-party logging integration.
    Tests our logger and third-party libraries.
    """
    # Test our structlog-based logger
    our_logger = get_logger(name="test.our_app")
    our_logger.info(msg="Message from our application", user_id=123, action="test")

    # Test third-party library simulation
    third_party_logger = logging.getLogger(name="third_party.library")
    third_party_logger.info(msg="Message from third-party library")
    third_party_logger.warning(msg="Warning from third-party library", extra={"component": "http_client"})

    print("Third-party logging should now be in JSON format!")
