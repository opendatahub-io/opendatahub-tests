"""Query validation independent of the cluster alert lifecycle."""

from unittest.mock import Mock

import pytest

from tests.ai_gateway.models_as_a_service.observability.authorino_alerts import (
    query_alert_metrics,
)


@pytest.mark.tier1
@pytest.mark.parametrize(
    "status,warnings,expected_error",
    [
        pytest.param("success", [], None, id="test_empty_vector_is_inactive"),
        pytest.param("success", ["counter name warning"], "unexpected warnings", id="test_counter_warning_fails"),
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


@pytest.mark.tier1
@pytest.mark.parametrize(
    "response_status,failure_increase,expected_error",
    [
        pytest.param(401, "5", None, id="test_transport_retries"),
        pytest.param(
            500, "0", "Expected authentication rejection, got HTTP 500", id="test_unexpected_http_status_fails"
        ),
        pytest.param(401, "0", "did not increase Authorino", id="test_no_counter_increase_fails"),
    ],
)
def test_authorino_alert_transport_retries(
    monkeypatch: pytest.MonkeyPatch, response_status: int, failure_increase: str, expected_error: str | None
) -> None:
    """Given transient transport failures, when testing alerts, then retries preserve HTTP and alert assertions."""
    from itertools import count
    from typing import Any

    import requests
    from timeout_sampler import TimeoutSampler

    from tests.ai_gateway.models_as_a_service.observability import test_authorino_alerts as alerts_test

    monkeypatch.setattr(alerts_test, "monotonic", lambda: next(clock))
    clock = count(start=0, step=100)
    rule = Mock(exists=True)
    rule.instance.to_dict.return_value = {
        "spec": {
            "groups": [
                {
                    "rules": [
                        {
                            "alert": alerts_test.ALERT_NAME,
                            "for": "5m",
                            "expr": "auth_server_authconfig_response_status_total",
                        }
                    ]
                }
            ]
        }
    }
    monkeypatch.setattr(alerts_test, "PrometheusRule", Mock(return_value=rule))
    monkeypatch.setattr(
        alerts_test, "Deployment", lambda **kwargs: Mock(exists=kwargs["namespace"] == "kuadrant-system")
    )
    session = Mock()
    session.__enter__ = Mock(return_value=session)
    session.__exit__ = Mock(return_value=False)
    session.post.side_effect = [
        requests.exceptions.ConnectionError("router reset"),
        requests.exceptions.ReadTimeout("router timeout"),
        *[Mock(status_code=response_status) for _ in range(5)],
    ]
    monkeypatch.setattr(alerts_test.requests, "Session", Mock(return_value=session))
    alert_results = iter([
        [],
        requests.exceptions.ConnectionError("query reset"),
        [{"metric": {"alertstate": "pending"}}],
        [{"metric": {"alertstate": "firing"}}],
        requests.exceptions.ReadTimeout("recovery query timeout"),
        [],
        [],
    ])
    rates = iter([requests.exceptions.ReadTimeout("rate query timeout"), [], [], []])
    recovery = False
    recovery_up_failed = False

    def query_metrics(prometheus: Any, query: str) -> list[dict[str, Any]]:
        nonlocal recovery, recovery_up_failed
        if query.startswith("sum(rate("):
            recovery = True
            result = next(rates)
        elif query.startswith("ALERTS"):
            result = [] if response_status != 401 and recovery else next(alert_results)
        elif query.startswith("min(up"):
            if recovery and not recovery_up_failed:
                recovery_up_failed = True
                raise requests.exceptions.ConnectionError("health query reset")
            result = [{"value": [0, "1"]}]
        elif query.startswith("sum(increase("):
            result = [{"value": [0, failure_increase]}]
        else:
            result = [{"value": [0, "1"]}]
        if isinstance(result, Exception):
            raise result
        return result

    def fast_sampler(**kwargs: Any) -> TimeoutSampler:
        kwargs.update(wait_timeout=1, sleep=0, print_log=False)
        return TimeoutSampler(**kwargs)

    monkeypatch.setattr(alerts_test, "query_alert_metrics", query_metrics)
    monkeypatch.setattr(alerts_test, "TimeoutSampler", fast_sampler)
    if expected_error is None:
        alerts_test.test_authorino_authentication_alert_fires(
            admin_client=Mock(),
            authorino_product_prometheus=Mock(verify_ssl=True),
            authorino_model_url="https://model.example/v1/chat/completions",
        )
    else:
        with pytest.raises(AssertionError, match=expected_error):
            alerts_test.test_authorino_authentication_alert_fires(
                admin_client=Mock(),
                authorino_product_prometheus=Mock(verify_ssl=True),
                authorino_model_url="https://model.example/v1/chat/completions",
            )
    assert session.post.call_count == (5 if response_status == 401 else 3)
    assert recovery_up_failed
    assert session.trust_env is False


@pytest.mark.tier1
@pytest.mark.parametrize(
    "endpoint",
    [pytest.param("model", id="test_model_tls_failure"), pytest.param("metrics", id="test_metrics_tls_failure")],
)
def test_authorino_tls_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    """Given invalid TLS, when sampling an endpoint, then fail on the first call instead of retrying."""
    import requests
    from timeout_sampler import TimeoutExpiredError, TimeoutSampler

    from tests.ai_gateway.models_as_a_service.observability.authorino_alerts import post_authorino_request

    request = Mock(side_effect=requests.exceptions.SSLError("certificate verify failed"))
    if endpoint == "model":
        sampler = TimeoutSampler(
            wait_timeout=1,
            sleep=0,
            func=post_authorino_request,
            exceptions_dict={requests.exceptions.ConnectionError: [], requests.exceptions.Timeout: []},
            session=Mock(post=request),
            url="https://model.example/v1/chat/completions",
        )
    else:
        monkeypatch.setattr("tests.ai_gateway.models_as_a_service.observability.authorino_alerts.requests.get", request)
        sampler = TimeoutSampler(
            wait_timeout=1,
            sleep=0,
            func=query_alert_metrics,
            exceptions_dict={requests.exceptions.ConnectionError: [], requests.exceptions.Timeout: []},
            prometheus=Mock(api_url="https://metrics.example", verify_ssl=True),
            query="up",
        )
    with pytest.raises(TimeoutExpiredError, match="TLS verification failed"):
        next(iter(sampler))
    request.assert_called_once()
