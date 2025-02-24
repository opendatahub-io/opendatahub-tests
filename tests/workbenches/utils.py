#
# Copyright Skodjob authors.
# License: Apache License 2.0 (see the file LICENSE or http://apache.org/licenses/LICENSE-2.0.html).
#
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator


@contextmanager
def step(description: str) -> Generator[None, None, None]:
    yield
