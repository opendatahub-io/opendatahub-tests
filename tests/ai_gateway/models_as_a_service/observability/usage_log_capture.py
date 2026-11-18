"""Read MaaS usage logs through the product Loki gateway."""

import json
import ssl
from dataclasses import dataclass, field
from http.client import HTTPSConnection
from time import monotonic, sleep, time_ns
from typing import Any
from urllib.parse import urlencode

LOKI_QUERY_PATH = "/api/logs/v1/application/loki/api/v1/query_range"
POLL_INTERVAL_SECONDS = 10
LOG_WAIT_SECONDS = 180


@dataclass(frozen=True)
class LokiUsageLogClient:
    """Query the application tenant over a loopback Kubernetes port-forward."""

    port: int
    tls_context: ssl.SSLContext = field(repr=False)
    bearer_token: str = field(repr=False)

    def query(self, subscription: str, key_name: str, start_ns: int) -> list[dict[str, str]]:
        """Return only log fields for one test API key within the request window.

        Args:
            subscription: Unique MaaSSubscription name used by the test.
            key_name: Unique API key name used for the traffic phase.
            start_ns: Loki range start in Unix nanoseconds.

        Returns:
            Parsed stream labels and structured log fields for matching records.
        """
        selector = f'{{service_name="models-as-a-service",subscription="{subscription}"}} | key_name="{key_name}"'
        params = urlencode(query={"query": selector, "start": start_ns, "end": time_ns(), "limit": 100})
        connection = HTTPSConnection(host="127.0.0.1", port=self.port, context=self.tls_context, timeout=30)
        try:
            connection.request(
                method="GET",
                url=f"{LOKI_QUERY_PATH}?{params}",
                headers={"Authorization": f"Bearer {self.bearer_token}", "Accept": "application/json"},
            )
            response = connection.getresponse()
            body = response.read()
        finally:
            connection.close()
        if response.status != 200:
            raise AssertionError(f"Loki query failed with HTTP {response.status}")
        try:
            payload: Any = json.loads(body)
        except (TypeError, ValueError) as error:
            raise AssertionError("Loki query returned invalid JSON") from error
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), dict):
            raise TypeError("Loki query returned an unexpected response shape")
        if payload.get("status") != "success" or payload["data"].get("resultType") != "streams":
            raise AssertionError("Loki query did not return successful log streams")
        return extract_log_fields(payload=payload)


def extract_log_fields(payload: dict[str, Any]) -> list[dict[str, str]]:
    """Read stream labels, JSON line fields, and Loki structured metadata.

    Args:
        payload: Successful Loki query_range JSON response.

    Returns:
        One merged field mapping for each log entry.
    """
    records: list[dict[str, str]] = []
    for stream in payload["data"]["result"]:
        stream_labels = stream.get("stream", {})
        for entry in stream.get("values", []):
            fields = {str(key): str(value) for key, value in stream_labels.items()}
            if len(entry) > 1:
                try:
                    line = json.loads(entry[1])
                except TypeError, ValueError:
                    line = None
                if isinstance(line, dict):
                    fields.update({str(key): str(value) for key, value in line.items()})
            if len(entry) > 2 and isinstance(entry[2], dict):
                fields.update({str(key): str(value) for key, value in entry[2].items()})
            records.append(fields)
    return records


def wait_for_usage_log(
    client: LokiUsageLogClient, subscription: str, key_name: str, start_ns: int
) -> list[dict[str, str]]:
    """Poll until traffic from one unique API key reaches Loki.

    Args:
        client: Authenticated Loki gateway client.
        subscription: Test subscription name.
        key_name: Test API key name.
        start_ns: Loki range start in Unix nanoseconds.

    Returns:
        Matching log records.

    Raises:
        AssertionError: If the log does not arrive within the ingestion window.
    """
    deadline = monotonic() + LOG_WAIT_SECONDS
    while monotonic() < deadline:
        records = client.query(subscription=subscription, key_name=key_name, start_ns=start_ns)
        if records:
            return records
        sleep(POLL_INTERVAL_SECONDS)
    raise AssertionError(f"No MaaS usage log arrived for API key name {key_name!r}")
