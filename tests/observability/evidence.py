"""Sanitized machine-readable and human-readable release evidence."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tests.observability.query import RawQueryResult

SENSITIVE_KEY_PATTERN = re.compile(r"(?:api[_-]?key|authorization|bearer|cookie|password|secret|token)", re.IGNORECASE)
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")


@dataclass(frozen=True)
class EvidenceRecord:
    """One sanitized release-contract evidence record."""

    tracking_id: str
    test_identifier: str
    release_stage: str
    component_versions: dict[str, str]
    cluster_run_id: str
    persona: dict[str, object]
    fixture_resources: dict[str, object]
    query: RawQueryResult
    failure_category: str | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a sanitized record."""
        return _sanitize(
            value={
                **asdict(self),
                "query": self.query.to_dict(),
            }
        )


def write_evidence(
    destination: str | Path,
    records: list[EvidenceRecord],
    handoff: dict[str, object] | None = None,
) -> Path:
    """Write release evidence as stable, indented JSON with a final newline."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0.0",
        "records": [record.to_dict() for record in records],
        "handoff": _sanitize(handoff or {}),
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_failure_log(destination: str | Path, records: list[EvidenceRecord]) -> Path:
    """Write concise human-readable failure lines without secret-bearing payloads."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        query = record.query
        lines.append(
            _sanitize(
                value=(
                    f"{record.test_identifier}: disposition={query.expected_disposition} "
                    f"http_status={query.http_status} prometheus_status={query.prometheus_status} "
                    f"error_type={query.error_type} error={'[REDACTED]' if query.error else None}"
                )
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def write_preflight_evidence(destination: str | Path, report: dict[str, object]) -> Path:
    """Write a sanitized preflight report before any resource mutation."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(report), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _sanitize(value: Any, key: str = "") -> Any:
    if SENSITIVE_KEY_PATTERN.search(key):
        return "[REDACTED]"
    if isinstance(value, dict):
        return {str(item_key): _sanitize(value=item_value, key=str(item_key)) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_sanitize(item, key) for item in value]
    if isinstance(value, tuple):
        return [_sanitize(item, key) for item in value]
    if isinstance(value, str):
        return BEARER_PATTERN.sub(repl="Bearer [REDACTED]", string=value)
    return value
