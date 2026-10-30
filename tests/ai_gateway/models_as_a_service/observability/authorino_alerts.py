"""Bounded queries for the Authorino alert E2E test."""

from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class AuthorinoMetricsClient:
    """Product Thanos connection through an authenticated Kubernetes port-forward."""

    api_url: str
    verify_ssl: str
    monitoring_namespace: str


def query_alert_metrics(prometheus: AuthorinoMetricsClient, query: str) -> list[dict[str, Any]]:
    """Query Thanos, rejecting API errors and partial results.

    Args:
        prometheus: DSCI-managed Thanos client.
        query: Instant PromQL expression.

    Returns:
        Instant-vector samples, including an empty vector for an inactive alert.
    """
    # query_sampler discards response warnings and cannot set partial_response=false.
    # Empty successful vectors are valid for inactive alerts; partial data is not.
    try:
        response = requests.get(
            url=f"{prometheus.api_url}/api/v1/query",
            verify=prometheus.verify_ssl,
            params={"query": query, "partial_response": "false"},
            timeout=30,
        )
    except requests.exceptions.SSLError as error:
        raise AssertionError("Monitoring TLS verification failed") from error
    response.raise_for_status()
    result = response.json()
    assert result.get("status") == "success", "Thanos query failed"
    assert not result.get("warnings"), "Thanos returned unexpected warnings or incomplete query results"
    assert result["data"]["resultType"] == "vector", "Expected an instant vector"
    return result["data"]["result"]


def post_authorino_request(session: requests.Session, url: str) -> requests.Response:
    """Send failure traffic, making TLS failures non-retryable.

    Args:
        session: Session configured with the model CA bundle and no credentials.
        url: Test simulator chat-completions URL.

    Returns:
        HTTP response to be checked for authentication rejection.
    """
    try:
        return session.post(
            url=url,
            json={"messages": [{"role": "user", "content": "alert test"}], "max_tokens": 1},
            timeout=10,
            allow_redirects=False,
        )
    except requests.exceptions.SSLError as error:
        raise AssertionError("Model TLS verification failed") from error
