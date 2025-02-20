#
# Copyright Skodjob authors.
# License: Apache License 2.0 (see the file LICENSE or http://apache.org/licenses/LICENSE-2.0.html).
#
from __future__ import annotations

from typing import Callable
import typing

T = typing.TypeVar("T")  # noqa: FCN001
StepRType = tuple[str, str]


def Desc(value: str) -> str:
    return value


def Step(
    value: str,
    expected: str,
) -> StepRType:
    return value, expected


def SuiteDoc(
    description: str,
    before_test_steps: set[StepRType],
    after_test_steps: set[StepRType],
) -> Callable[[T], T]:
    return lambda x: x


def Contact(
    name: str,
    email: str,
) -> tuple[str, str]:
    return name, email


def TestDoc(
    description: str,
    contact: tuple[str, str],
    steps: set[StepRType],
) -> Callable[[T], T]:
    return lambda x: x
