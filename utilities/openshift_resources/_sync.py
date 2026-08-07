"""Sync-to-async bridge for running async methods from sync callers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is None or _loop.is_closed():
            _loop = asyncio.new_event_loop()
            _thread = threading.Thread(target=_loop.run_forever, daemon=True)
            _thread.start()
    return _loop


def _run_sync(coro: Any) -> Any:
    """Run an async coroutine synchronously using a background event loop.

    Safe to call from sync code (tests, fixtures, scripts). The background
    event loop is created once and reused, so aiohttp sessions stay alive
    across calls.
    """
    loop = _get_loop()
    future = asyncio.run_coroutine_threadsafe(coro=coro, loop=loop)
    return future.result()
