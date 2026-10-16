"""Resolve a stable, safe correlation identifier for one observability run."""

from __future__ import annotations

import re
import secrets
from collections.abc import Mapping
from dataclasses import dataclass

RUN_ID_MAX_LENGTH = 63
CI_RUN_ID_VARIABLES = ("CI_JOB_ID", "BUILD_ID", "BUILD_NUMBER", "GITHUB_RUN_ID")


@dataclass(frozen=True)
class ResolvedRunID:
    """A normalized run identifier and the non-sensitive source used to resolve it."""

    value: str
    source: str

    def to_dict(self) -> dict[str, str]:
        """Return the only run metadata that is safe to persist in evidence."""
        return {"value": self.value, "source": self.source}


def resolve_run_id(environment: Mapping[str, str] | None = None) -> ResolvedRunID:
    """Resolve one run ID using explicit, CI, then cryptographically random input.

    Empty values are ignored. The selected value is normalized before it is returned,
    so callers can safely use it in labels, resource names, and telemetry markers.
    """
    values = environment or {}
    explicit = values.get("RHOAI_OBSERVABILITY_RUN_ID", "")
    if explicit.strip():
        return ResolvedRunID(value=normalize_run_id(explicit), source="explicit")

    for variable in CI_RUN_ID_VARIABLES:
        candidate = values.get(variable, "")
        if candidate.strip():
            return ResolvedRunID(value=normalize_run_id(candidate), source="ci")

    generated = str(secrets.randbelow(2**63 - 1) + 1)
    return ResolvedRunID(value=normalize_run_id(generated), source="generated")


def normalize_run_id(value: str) -> str:
    """Normalize arbitrary input to a bounded lowercase marker-safe identifier."""
    normalized = re.sub(pattern=r"[^a-z0-9-]+", repl="-", string=value.casefold())
    normalized = re.sub(pattern=r"-{2,}", repl="-", string=normalized).strip("-")
    if not normalized:
        normalized = "run"
    return normalized[:RUN_ID_MAX_LENGTH].rstrip("-") or "run"
