"""Query and parse MaaS identity fields in Loki usage logs."""

import json
from time import monotonic, sleep, time_ns
from typing import Any

import requests

from tests.ai_gateway.models_as_a_service.observability.constants import LOKI_QUERY_TENANT, LOKI_QUERY_URL

POLL_INTERVAL_SECONDS = 10
LOG_WAIT_SECONDS = 180


def extract_log_fields(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Read stream labels, JSON line fields, and Loki structured metadata.

    Args:
        payload: Successful Loki query_range JSON response.

    Returns:
        One merged field mapping for each log entry.
    """
    records: list[dict[str, Any]] = []
    for stream in payload["data"]["result"]:
        stream_labels = stream.get("stream", {})
        for entry in stream.get("values", []):
            fields: dict[str, Any] = dict(stream_labels)
            if len(entry) > 1:
                try:
                    line = json.loads(entry[1])
                except TypeError, ValueError:
                    line = None
                if isinstance(line, dict):
                    fields.update(line)
            if len(entry) > 2 and isinstance(entry[2], dict):
                fields.update(entry[2])
            records.append(fields)
    return records


def wait_for_usage_log(
    http_session: requests.Session,
    admin_token: str,
    subscription: str,
    key_name: str,
    start_ns: int,
) -> list[dict[str, Any]]:
    """Poll Loki until traffic from one unique API key arrives.

    Args:
        http_session: HTTP session used by the shared Loki query helpers.
        admin_token: OpenShift token authorized for Loki application logs.
        subscription: Test MaaSSubscription name.
        key_name: Test API key name.
        start_ns: Loki range start in Unix nanoseconds.

    Returns:
        Matching log records.

    Raises:
        AssertionError: If the log does not arrive within the ingestion window.
    """
    selector = f'{{service_name="models-as-a-service",subscription="{subscription}"}} | key_name="{key_name}"'
    deadline = monotonic() + LOG_WAIT_SECONDS
    while monotonic() < deadline:
        response = http_session.get(
            url=f"{LOKI_QUERY_URL}_range",
            headers={"Authorization": f"Bearer {admin_token}", "X-Scope-OrgID": LOKI_QUERY_TENANT},
            params={"query": selector, "start": start_ns, "end": time_ns(), "limit": 100},
            timeout=30,
        )
        assert response.status_code == 200, f"Loki query failed with HTTP {response.status_code}"
        payload: Any = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise TypeError("Loki query returned an unexpected response shape")
        if payload.get("status") != "success" or payload["data"].get("resultType") != "streams":
            raise AssertionError("Loki query did not return successful log streams")
        records = extract_log_fields(payload=payload)
        if records:
            return records
        sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"No MaaS usage log arrived for API key name {key_name!r}")
