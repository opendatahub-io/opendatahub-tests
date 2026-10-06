from unittest.mock import Mock

import pytest

from tests.observability.contract import ContractRecord
from tests.observability.query import QueryRequest, RawQueryClient, build_contract_request

pytestmark = pytest.mark.tier1


def test_raw_query_preserves_prometheus_contract_and_normalizes_series() -> None:
    """Given a successful vector response, retain status, labels, values, and request metadata."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "status": "success",
        "warnings": ["partial response"],
        "data": {
            "resultType": "vector",
            "result": [{"metric": {"namespace": "ns-a", "pod": "model-a"}, "value": [1730000000.12345, "1"]}],
        },
    }
    session = Mock()
    session.request.return_value = response
    client = RawQueryClient(base_url="https://metrics.example", session=session)

    result = client.query(
        request=QueryRequest(
            persona="namespace-admin",
            principal="alice",
            requested_namespace="ns-a",
            fixture_namespace="ns-a",
            datasource="namespace-proxy",
            route="/api/prometheus/api/v1/query",
            promql='up{namespace="ns-a"}',
            query_params={"namespace": "ns-a"},
            expected_disposition="pass",
        ),
        bearer_token="not-written-to-evidence",
    )

    assert result.http_status == 200
    assert result.prometheus_status == "success"
    assert result.result_type == "vector"
    assert result.warnings == ("partial response",)
    assert result.series[0].labels == (("namespace", "ns-a"), ("pod", "model-a"))
    assert result.series[0].values == ((1730000000.123, "1"),)
    assert result.url_path == "/api/prometheus/api/v1/query"
    session.request.assert_called_once()
    assert session.request.call_args.kwargs["headers"]["Authorization"] == "Bearer not-written-to-evidence"
    assert "not-written-to-evidence" not in str(result.to_dict())


def test_raw_query_retains_http_and_prometheus_errors() -> None:
    """Given an HTTP error with a Prometheus error payload, distinguish it from an empty success."""
    response = Mock()
    response.status_code = 400
    response.json.return_value = {
        "status": "error",
        "errorType": "bad_data",
        "error": "invalid parameter",
        "data": {"resultType": "vector", "result": []},
    }
    session = Mock()
    session.request.return_value = response

    result = RawQueryClient(base_url="https://metrics.example", session=session).query(
        request=QueryRequest(
            persona="cluster-admin",
            principal="cluster-admin",
            requested_namespace=None,
            fixture_namespace="ns-a",
            datasource="cluster-thanos",
            route="/api/v1/query",
            promql="bad query",
            query_params={},
            expected_disposition="fail",
        )
    )

    assert result.http_status == 400
    assert result.prometheus_status == "error"
    assert result.error_type == "bad_data"
    assert result.error == "invalid parameter"
    assert result.series == ()


def test_raw_query_keeps_transport_errors_as_unavailable() -> None:
    """Given a transport exception, retain an unavailable disposition instead of raising a false empty result."""
    session = Mock()
    session.request.side_effect = OSError("route unavailable")

    result = RawQueryClient(base_url="https://metrics.example", session=session).query(
        request=QueryRequest(
            persona="cluster-admin",
            principal="cluster-admin",
            requested_namespace=None,
            fixture_namespace="ns-a",
            datasource="cluster-thanos",
            route="/api/v1/query",
            promql="up",
            query_params={},
            expected_disposition="unavailable",
        )
    )

    assert result.http_status is None
    assert result.transport_error == "OSError"
    assert result.expected_disposition == "unavailable"


def test_build_contract_request_injects_namespace_tenancy_and_rendered_promql() -> None:
    """Given a contract query, build a request with rendered variables and the tenancy selector."""
    contract = ContractRecord(
        identifier="query",
        release_stage="GA",
        product_versions={"rhoai": "test"},
        dashboard="cluster",
        panel="panel",
        datasource="tenancy",
        route="/api/v1/query",
        promql='up{namespace="${namespace}"}',
        time_range={},
        expected_http_status=(200,),
        expected_prometheus_status="success",
        expected_result_type="vector",
        minimum_series=0,
        required_labels=(),
        empty_result_valid=True,
        empty_ui_state="No data",
        capability="shipped",
        authorization_response="not-applicable",
        warnings_allowed=False,
    )

    request = build_contract_request(
        contract=contract,
        persona="namespace-admin",
        principal="alice",
        requested_namespace="ns-a",
        fixture_namespace="ns-a",
        variables={"namespace": "ns-a"},
    )

    assert request.query_params == {"namespace": "ns-a"}
    assert request.promql == 'up{namespace="ns-a"}'
