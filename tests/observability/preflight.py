"""Explicit release and environment preflight dispositions."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum


class PreflightDisposition(StrEnum):
    """Overall release gate outcome."""

    READY = "ready"
    BLOCKED = "blocked"
    FAILED = "failed"


@dataclass(frozen=True)
class PreflightCheck:
    """One prerequisite and its owner category."""

    name: str
    present: bool
    category: str
    detail: str = ""

    def __post_init__(self) -> None:
        if self.category not in {"environment", "product"}:
            raise ValueError("category must be 'environment' or 'product'")
        if not self.name:
            raise ValueError("preflight check name must not be empty")


@dataclass(frozen=True)
class PreflightReport:
    """Sanitized preflight result suitable for evidence."""

    disposition: PreflightDisposition
    blocked: tuple[str, ...]
    failed: tuple[str, ...]
    details: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Return a machine-readable representation."""
        return {
            "disposition": self.disposition.value,
            "blocked": list(self.blocked),
            "failed": list(self.failed),
            "details": dict(self.details),
        }


def parse_preflight_present(value: object) -> bool:
    """Accept only JSON booleans for prerequisite presence values."""
    if not isinstance(value, bool):
        raise TypeError("preflight present must be a JSON boolean")
    return value


def authorization_preflight_checks(records: Iterable[tuple[str, str]]) -> list[PreflightCheck]:
    """Turn unresolved authorization declarations into product release blockers."""
    return [
        PreflightCheck(
            name=f"authorization-contract:{identifier}",
            present=response != "review-required",
            category="product",
            detail=(
                f"authorization response for {identifier} must be reviewed before namespace assertions run"
                if response == "review-required"
                else ""
            ),
        )
        for identifier, response in records
    ]


def evaluate_preflight(checks: list[PreflightCheck]) -> PreflightReport:
    """Evaluate prerequisites without collapsing environment and product failures."""
    blocked = tuple(check.name for check in checks if not check.present and check.category == "environment")
    failed = tuple(check.name for check in checks if not check.present and check.category == "product")
    details = {check.name: check.detail for check in checks if not check.present and check.detail}
    disposition = (
        PreflightDisposition.FAILED
        if failed
        else PreflightDisposition.BLOCKED
        if blocked
        else PreflightDisposition.READY
    )
    return PreflightReport(disposition=disposition, blocked=blocked, failed=failed, details=details)
