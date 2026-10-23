"""Query validation independent of the cluster alert lifecycle."""

from unittest.mock import Mock

import pytest

from tests.ai_gateway.models_as_a_service.observability.authorino_alerts import (
    AUTHORINO_COUNTER_NAME_WARNING,
    query_alert_metrics,
)


@pytest.mark.tier1
@pytest.mark.parametrize(
    "status,warnings,expected_error",
    [
        pytest.param("success", [], None, id="test_empty_vector_is_inactive"),
        pytest.param("success", [AUTHORINO_COUNTER_NAME_WARNING], None, id="test_counter_name_info_is_allowed"),
        pytest.param("success", ["partial response"], "unexpected warnings", id="test_partial_results_fail"),
        pytest.param("error", [], "Thanos query failed", id="test_api_error_fails"),
    ],
)
def test_authorino_query_response_validation(
    monkeypatch: pytest.MonkeyPatch,
    status: str,
    warnings: list[str],
    expected_error: str | None,
) -> None:
    """Given a Thanos response, when querying alerts, then errors cannot masquerade as inactive alerts."""
    response = Mock()
    response.json.return_value = {
        "status": status,
        "warnings": warnings,
        "data": {"resultType": "vector", "result": []},
    }
    get = Mock(return_value=response)
    monkeypatch.setattr("tests.ai_gateway.models_as_a_service.observability.authorino_alerts.requests.get", get)
    prometheus = Mock(api_url="https://metrics.example", headers={}, verify_ssl=True)
    if expected_error:
        with pytest.raises(AssertionError, match=expected_error):
            query_alert_metrics(prometheus=prometheus, query="ALERTS")
    else:
        assert query_alert_metrics(prometheus=prometheus, query="ALERTS") == []
    response.raise_for_status.assert_called_once()
    assert get.call_args.kwargs["timeout"] == 30
    assert get.call_args.kwargs["params"]["partial_response"] == "false"
