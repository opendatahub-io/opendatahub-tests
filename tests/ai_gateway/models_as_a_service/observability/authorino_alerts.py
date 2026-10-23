"""Bounded queries for the Authorino alert E2E test."""

from typing import Any

import requests
from ocp_utilities.monitoring import Prometheus

AUTHORINO_COUNTER_NAME_WARNING = (
    "PromQL info: metric might not be a counter, name does not end in _total/_sum/_count/_bucket: "
    '"auth_server_authconfig_response_status"'
)


def query_alert_metrics(prometheus: Prometheus, query: str) -> list[dict[str, Any]]:
    """Query Thanos, rejecting API errors and partial results.

    Args:
        prometheus: Authenticated cluster Thanos client.
        query: Instant PromQL expression.

    Returns:
        Instant-vector samples, including an empty vector for an inactive alert.
    """
    # query_sampler discards response warnings and cannot set partial_response=false.
    # Empty successful vectors are valid for inactive alerts; partial data is not.
    response = requests.get(
        url=f"{prometheus.api_url}/api/v1/query",
        headers=prometheus.headers,
        verify=prometheus.verify_ssl,
        params={"query": query, "partial_response": "false"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    assert result.get("status") == "success", "Thanos query failed"
    # Authorino exposes this counter without the conventional _total suffix.
    unexpected_warnings = [
        warning for warning in result.get("warnings", []) if warning != AUTHORINO_COUNTER_NAME_WARNING
    ]
    assert not unexpected_warnings, "Thanos returned unexpected warnings or incomplete query results"
    assert result["data"]["resultType"] == "vector", "Expected an instant vector"
    return result["data"]["result"]
