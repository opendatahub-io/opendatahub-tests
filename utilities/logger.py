import logging
import multiprocessing
import shutil
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from typing import Any

from simple_logger.logger import DuplicateFilter, WrapperLogFormatter

LOGGER = logging.getLogger(__name__)


class RedactedString(str):
    """
    Used to redact the representation of a sensitive string.
    """

    def __new__(cls, *, value: object) -> "RedactedString":  # noqa: PYI034
        return super().__new__(cls, value)

    def __repr__(self) -> str:
        return "'***REDACTED***'"


def setup_logging(
    log_level: int, log_file: str = "/tmp/pytest-tests.log", thread_name: str | None = None, enable_console: bool = True
) -> QueueListener:
    """
    Setup basic/root logging using QueueHandler/QueueListener
    to consolidate log messages into a single stream to be written to multiple outputs.

    Args:
        log_level (int): log level
        log_file (str): logging output file
        thread_name (str | None): optional thread_name id prefix, e.g., [gw0]

    Returns:
        QueueListener: Process monitoring the log Queue

    Eg:
       root QueueHandler ┐                         ┌> StreamHandler
                         ├> Queue -> QueueListener ┤
      basic QueueHandler ┘                         └> FileHandler
    """
    basic_fmt_str = "%(message)s"
    root_fmt_str = "%(asctime)s %(name)s %(log_color)s%(levelname)s%(reset)s %(message)s"

    if thread_name:
        basic_fmt_str = f"[{thread_name}] {basic_fmt_str}"
        root_fmt_str = f"[{thread_name}] {root_fmt_str}"

    basic_log_formatter = logging.Formatter(fmt=basic_fmt_str)
    root_log_formatter = WrapperLogFormatter(
        fmt=root_fmt_str,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
    )

    log_file_handler = RotatingFileHandler(filename=log_file, maxBytes=100 * 1024 * 1024, backupCount=20)
    log_file_handler.setLevel(level=log_level)  # Set the file handler log level

    handlers: list[Any] = [log_file_handler]

    # Convert log_level to int if it's a string
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)

    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level=log_level)  # Set the console handler log level
        handlers.append(console_handler)

    log_queue = multiprocessing.Queue(maxsize=-1)  # type: ignore[var-annotated]
    log_listener = QueueListener(log_queue, *handlers)

    basic_log_queue_handler = QueueHandler(queue=log_queue)
    basic_log_queue_handler.set_name(name="basic")
    basic_log_queue_handler.setFormatter(fmt=basic_log_formatter)

    basic_logger = logging.getLogger(name="basic")
    basic_logger.setLevel(level=log_level)
    basic_logger.handlers.clear()
    basic_logger.addHandler(hdlr=basic_log_queue_handler)

    root_log_queue_handler = QueueHandler(queue=log_queue)
    root_log_queue_handler.set_name(name="root")
    root_log_queue_handler.setFormatter(fmt=root_log_formatter)

    root_logger = logging.getLogger(name="root")
    root_logger.setLevel(level=log_level)
    root_logger.handlers.clear()
    root_logger.addHandler(hdlr=root_log_queue_handler)
    root_logger.addFilter(filter=DuplicateFilter())

    root_logger.propagate = False
    basic_logger.propagate = False

    # Always configure all loggers to use our queue system
    # This ensures test loggers and third-party loggers respect our console setting
    for name, logger in logging.root.manager.loggerDict.items():
        if isinstance(logger, logging.Logger) and (name not in ("root", "basic")):
            logger.handlers.clear()
            logger.addHandler(hdlr=root_log_queue_handler)
            logger.propagate = False

    # Configure the root logger to catch any new loggers that inherit from it
    # First, completely clear any existing configuration
    logging.root.handlers.clear()
    logging.root.setLevel(level=log_level)  # Set root logger to respect our log level
    logging.root.addHandler(hdlr=root_log_queue_handler)

    # Also ensure the root logger doesn't have any lingering configuration
    for handler in logging.root.handlers[:]:
        if handler != root_log_queue_handler:
            logging.root.removeHandler(hdlr=handler)

    log_listener.start()
    return log_listener


def separator(symbol_: str, val: str | None = None) -> str:
    terminal_width = shutil.get_terminal_size(fallback=(120, 40))[0]
    if not val:
        return f"{symbol_ * terminal_width}"

    sepa = int((terminal_width - len(val) - 2) // 2)
    return f"{symbol_ * sepa} {val} {symbol_ * sepa}"
