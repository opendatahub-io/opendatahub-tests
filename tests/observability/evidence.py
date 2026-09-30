"""Sanitized machine-readable and human-readable release evidence."""

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tests.observability.query import RawQueryResult

SENSITIVE_KEY_PATTERN = re.compile(r"(?:api[_-]?key|authorization|bearer|cookie|password|secret|token)", re.IGNORECASE)
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[^\s,;]+")
BASIC_PATTERN = re.compile(r"(?i)\bBasic\s+[^\s,;]+")
JWT_PATTERN = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*")
KEY_VALUE_PATTERN = re.compile(
    r"(?i)\b((?:[a-z0-9]+[_-])*(?:api[_-]?key|authorization|bearer|cookie|password|secret|token))"
    r"\s*[=:]\s*[^\s,;&]+"
)


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
            f"{_sanitize_log_field(record.test_identifier)}: "
            f"disposition={_sanitize_log_field(query.expected_disposition)} "
            f"http_status={_sanitize_log_field(query.http_status)} "
            f"prometheus_status={_sanitize_log_field(query.prometheus_status)} "
            f"error_type={_sanitize_log_field(query.error_type)} "
            f"error={_sanitize_log_field('[REDACTED]' if query.error else None)}"
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
        redacted = BEARER_PATTERN.sub(repl="Bearer [REDACTED]", string=value)
        redacted = BASIC_PATTERN.sub(repl="Basic [REDACTED]", string=redacted)
        redacted = JWT_PATTERN.sub(repl="[REDACTED]", string=redacted)
        return KEY_VALUE_PATTERN.sub(repl=r"\1=[REDACTED]", string=redacted)
    return value


def _sanitize_log_field(value: object) -> str:
    """Redact a failure-log field and escape line-breaking control characters."""
    sanitized = str(_sanitize(value=value))
    return sanitized.replace("\r", r"\r").replace("\n", r"\n")
