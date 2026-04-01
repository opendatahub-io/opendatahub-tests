"""
OpenDataHub logging utilities using structlog.

Provides structured logging with two output modes:
  - Console: human-readable, colorized via structlog's ConsoleRenderer
  - File: structured JSON via structlog's JSONRenderer

Third-party libraries (ocp_resources, timeout_sampler, kubernetes) that use
stdlib logging are automatically captured via the foreign pre-chain pattern
(structlog.stdlib.ProcessorFormatter).

Usage:
    import structlog

    LOGGER = structlog.get_logger(name=__name__)
    LOGGER.info("User logged in", user_id=123)
"""

import logging
import multiprocessing
import shutil
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any

import structlog

_SHARED_PROCESSORS: list[structlog.types.Processor] = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_logger_name,
    structlog.stdlib.add_log_level,
    structlog.stdlib.PositionalArgumentsFormatter(),
    structlog.processors.TimeStamper(fmt="iso", utc=True),
]


def configure_structlog() -> None:
    """Configure structlog with stdlib integration.

    Both structlog loggers and plain stdlib loggers (from third-party libraries)
    are processed through the same pipeline and rendered by the same handlers.
    """
    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _get_console_formatter(thread_name: str | None = None) -> structlog.stdlib.ProcessorFormatter:
    """Create a ProcessorFormatter for human-readable console output."""
    return structlog.stdlib.ProcessorFormatter(
        processors=[
            _prepend_thread_name(thread_name) if thread_name else _noop,
            _strip_basic_metadata,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(),
        ],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )


def _get_json_formatter(thread_name: str | None = None) -> structlog.stdlib.ProcessorFormatter:
    """Create a ProcessorFormatter for structured JSON file output."""
    return structlog.stdlib.ProcessorFormatter(
        processors=[
            _prepend_thread_name(thread_name) if thread_name else _noop,
            _strip_basic_metadata,
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
        foreign_pre_chain=_SHARED_PROCESSORS,
    )


def _strip_basic_metadata(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    if event_dict.get("logger") == "basic":
        event_dict.pop("logger", None)
        event_dict.pop("timestamp", None)
        event_dict.pop("level", None)
    return event_dict


def _noop(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    return event_dict


def _prepend_thread_name(
    thread_name: str,
) -> structlog.types.Processor:
    """Return a processor that prepends [thread_name] to the event message."""

    def processor(
        logger: structlog.types.WrappedLogger,
        method_name: str,
        event_dict: structlog.types.EventDict,
    ) -> structlog.types.EventDict:
        event_dict["event"] = f"[{thread_name}] {event_dict['event']}"
        return event_dict

    return processor


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


class _StructlogQueueHandler(QueueHandler):
    """QueueHandler that preserves structlog's event dict through the queue.

    The default QueueHandler.prepare() calls self.format(record) which converts
    record.msg to a string. structlog's ProcessorFormatter on the listener side
    expects record.msg to still be a dict, so we skip formatting here.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        if record.exc_info:
            if not record.exc_text:
                record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
        record.stack_info = None
        return record


_original_add_handler = logging.Logger.addHandler


def _patch_add_handler(queue_handler: _StructlogQueueHandler) -> None:
    """Monkey-patch Logger.addHandler to force all logging through our queue.

    Third-party libraries (e.g. openshift-python-wrapper / simple_logger) add
    their own handlers at import or first-use time — long after setup_logging()
    runs.  This patch silently replaces those handlers with our queue handler so
    every log record goes through the ProcessorFormatter pipeline.
    """

    def _patched(self: logging.Logger, hdlr: logging.Handler) -> None:
        if isinstance(hdlr, _StructlogQueueHandler):
            _original_add_handler(self, hdlr)  # noqa: FCN001
            return
        if not any(isinstance(h, _StructlogQueueHandler) for h in self.handlers):
            _original_add_handler(self, queue_handler)  # noqa: FCN001

    logging.Logger.addHandler = _patched  # type: ignore[method-assign]


class RedactedString(str):
    """
    Used to redact the representation of a sensitive string.
    """

    def __new__(cls, *, value: object) -> RedactedString:  # noqa: PYI034
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return "'***REDACTED***'"


def setup_logging(
    log_level: int,
    log_file: str = "/tmp/pytest-tests.log",
    thread_name: str | None = None,
    enable_console: bool = True,
) -> QueueListener:
    """
    Setup structlog and root logging using QueueHandler/QueueListener
    to consolidate log messages into a single stream to be written to multiple outputs.

    Console output uses structlog's ConsoleRenderer (human-readable, colorized).
    File output uses structlog's JSONRenderer (structured JSON).

    Args:
        log_level (int): log level
        log_file (str): logging output file
        thread_name (str | None): optional thread_name id prefix, e.g., [gw0]
        enable_console (bool): whether to enable console output

    Returns:
        QueueListener: Process monitoring the log Queue

    Eg:
       all loggers -> QueueHandler -> Queue -> QueueListener -> StreamHandler (ConsoleRenderer)
                                                              -> FileHandler  (JSONRenderer)
    """
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    configure_structlog()

    log_file_handler = RotatingFileHandler(filename=log_file, maxBytes=100 * 1024 * 1024, backupCount=20)
    log_file_handler.setLevel(level=log_level)
    log_file_handler.setFormatter(fmt=_get_json_formatter(thread_name=thread_name))

    handlers: list[Any] = [log_file_handler]

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level=log_level)
        console_handler.setFormatter(fmt=_get_console_formatter(thread_name=thread_name))
        handlers.append(console_handler)

    log_queue: multiprocessing.Queue[Any] = multiprocessing.Queue(maxsize=-1)
    log_listener = QueueListener(log_queue, *handlers, respect_handler_level=True)

    queue_handler = _StructlogQueueHandler(queue=log_queue)

    # Redirect all existing loggers (including third-party like ocp_resources,
    # timeout_sampler) through our queue so they get the same formatting.
    for logger_ref in logging.root.manager.loggerDict.values():
        if isinstance(logger_ref, logging.Logger):
            logger_ref.handlers.clear()
            logger_ref.addHandler(hdlr=queue_handler)
            logger_ref.propagate = False

    # Configure the root logger to catch any new loggers created later.
    logging.root.handlers.clear()
    logging.root.setLevel(level=log_level)
    logging.root.addHandler(hdlr=queue_handler)
    logging.root.addFilter(filter=DuplicateFilter())

    # Monkey-patch addHandler so third-party libraries (e.g. ocp_resources /
    # simple_logger) that add handlers after setup still route through our queue.
    _patch_add_handler(queue_handler=queue_handler)

    # Keep a "basic" logger for separator/banner messages (message-only, no metadata).
    basic_logger = logging.getLogger(name="basic")
    basic_logger.setLevel(level=log_level)
    basic_logger.handlers.clear()
    basic_logger.addHandler(hdlr=queue_handler)
    basic_logger.propagate = False

    log_listener.start()
    return log_listener


def separator(symbol_: str, val: str | None = None) -> str:
    terminal_width = shutil.get_terminal_size(fallback=(120, 40))[0]
    if not val:
        return f"{symbol_ * terminal_width}"

    sepa = int((terminal_width - len(val) - 2) // 2)
    return f"{symbol_ * sepa} {val} {symbol_ * sepa}"
