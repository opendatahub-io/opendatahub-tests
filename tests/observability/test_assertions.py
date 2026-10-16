from dataclasses import replace

import pytest

from tests.observability.assertions import (
    QueryContractError,
    assert_authorization_response,
    assert_capability_is_available,
    assert_capability_unavailable,
    assert_namespace_isolation,
    assert_pipeline_health_contract,
    assert_query_contract,
)
from tests.observability.contract import ContractRecord, PipelineHealthRecord
from tests.observability.query import NormalizedSeries, RawQueryResult, SignalQueryResult
from tests.observability.rollout import PipelineStage, RolloutState

pytestmark = pytest.mark.tier1


def _result(labels: tuple[tuple[str, str], ...]) -> RawQueryResult:
    return RawQueryResult(
        persona="namespace-admin",
        principal="namespace-admin",
        requested_namespace="ns-a",
        fixture_namespace="ns-a",
        datasource="namespace-proxy",
        url_path="/api/prometheus/api/v1/query",
        http_method="GET",
        http_status=200,
        response_time_ms=1.0,
        query_params={"namespace": "ns-a"},
        promql="up",
        time_range={},
        prometheus_status="success",
        error_type=None,
        error=None,
        warnings=(),
        result_type="vector",
        series=(NormalizedSeries(labels=labels, values=((1.0, "1"),)),),
        expected_disposition="pass",
    )


def _signal_result() -> SignalQueryResult:
    return SignalQueryResult(
        signal="traces",
        marker="marker-1",
        persona="cluster-admin",
        principal="admin",
        requested_namespace="ns-a",
        fixture_namespace="ns-a",
        url_path="/api/search",
        http_method="GET",
        http_status=200,
        response_time_ms=1.0,
        query_params={},
        request_summary={},
        backend_status="success",
        backend_error_type=None,
        backend_error=None,
        result_count=1,
        labels=(),
        attributes=(("service.name", "model"),),
        edges=(),
        expected_disposition="pass",
        rollout_state="passed",
    )


def _pipeline_health_contract() -> PipelineHealthRecord:
    return PipelineHealthRecord(
        identifier="model-traces-pipeline",
        release_stage="GA",
        signal="traces",
        stage=PipelineStage.USER_RETRIEVAL,
        marker="namespace-workload-pod-window",
        source_query={},
        backend_query={},
        required_labels=(),
        required_attributes=("service.name",),
        required_edges=(),
        propagation_timeout_seconds=240,
        query_window_seconds=240,
        expected_http_status=(200,),
        expected_authorization="cluster-admin",
        minimum_results=1,
        empty_result_valid=False,
        rollout_state=RolloutState.PASSED,
        activation_release="GA",
        reason="activated for assertion testing",
        next_activation_step="none",
    )


def test_namespace_isolation_rejects_foreign_series() -> None:
    """Given an in-scope query response, reject any foreign namespace label."""
    with pytest.raises(QueryContractError, match="foreign namespace"):
        assert_namespace_isolation(
            result=_result(labels=(("namespace", "ns-b"),)),
            allowed_namespaces={"ns-a"},
        )


def test_authorization_assertion_requires_exact_denial_contract() -> None:
    """Given a pending denial contract, refuse to classify an arbitrary response as authorization behavior."""
    with pytest.raises(QueryContractError, match="must be reviewed"):
        assert_authorization_response(result=_result(labels=()), expected="review-required")


def test_success_empty_authorization_contract_checks_empty_series() -> None:
    """Given a successful empty response, accept it only for the explicit success-empty contract."""
    result = _result(labels=())
    result = replace(result, series=())

    assert_authorization_response(result=result, expected="success-empty")


def test_success_filtered_authorization_contract_rejects_empty_series() -> None:
    """Given a successful response with no series, reject it as evidence of filtered authorization."""
    result = replace(_result(labels=()), series=())

    with pytest.raises(QueryContractError, match="at least one filtered series"):
        assert_authorization_response(result=result, expected="success-filtered")


def test_success_filtered_authorization_contract_requires_in_scope_series() -> None:
    """Given filtered data, accept only a non-empty result whose namespace remains in the persona scope."""
    result = _result(labels=(("namespace", "ns-a"),))

    assert_authorization_response(result=result, expected="success-filtered")
    assert_namespace_isolation(result=result, allowed_namespaces={"ns-a"})


def test_isolation_only_authorization_contract_requires_populated_success() -> None:
    """Given a route without user authorization, require populated success before namespace isolation runs."""
    assert_authorization_response(result=_result(labels=(("namespace", "ns-b"),)), expected="isolation-only")

    with pytest.raises(QueryContractError, match="populated response"):
        assert_authorization_response(result=replace(_result(labels=()), series=()), expected="isolation-only")


@pytest.mark.parametrize("status", [403, 404])
def test_denial_authorization_contract_rejects_returned_series(status: int) -> None:
    """Given a denial response containing series, reject it even when the HTTP status is forbidden or not found."""
    result = replace(_result(labels=()), http_status=status)

    with pytest.raises(QueryContractError, match="must not return series"):
        assert_authorization_response(result=result, expected=str(status))


@pytest.mark.parametrize("status", [403, 404])
def test_denial_authorization_contract_accepts_series_free_response(status: int) -> None:
    """Given a series-free denial response, accept the reviewed forbidden or not-found contract."""
    result = replace(_result(labels=()), http_status=status, series=())

    assert_authorization_response(result=result, expected=str(status))


def test_unshipped_capability_cannot_pass_as_positive_data() -> None:
    """Given an explicitly unshipped capability, reject positive assertions and retain its unavailable status."""
    contract = ContractRecord(
        identifier="showback",
        release_stage="GA",
        product_versions={"rhoai": "test"},
        dashboard="maas",
        panel="showback",
        datasource="thanos",
        route="/api/v1/query",
        promql="up",
        time_range={},
        expected_http_status=(200,),
        expected_prometheus_status="success",
        expected_result_type="vector",
        minimum_series=0,
        required_labels=(),
        empty_result_valid=True,
        empty_ui_state="Unavailable",
        capability="not-shipped",
        authorization_response="not-applicable",
        warnings_allowed=False,
    )

    with pytest.raises(QueryContractError, match="not-shipped"):
        assert_capability_is_available(contract=contract)
    assert_capability_unavailable(contract=contract)


def test_query_contract_checks_http_prometheus_result_and_labels() -> None:
    """Given a complete response, validate the full panel contract rather than HTTP status alone."""
    contract = ContractRecord(
        identifier="model-table",
        release_stage="GA",
        product_versions={"rhoai": "test"},
        dashboard="models",
        panel="model-table",
        datasource="thanos",
        route="/api/v1/query",
        promql="up",
        time_range={},
        expected_http_status=(200,),
        expected_prometheus_status="success",
        expected_result_type="vector",
        minimum_series=1,
        required_labels=("namespace",),
        empty_result_valid=False,
        empty_ui_state="No data",
        capability="shipped",
        authorization_response="not-applicable",
        warnings_allowed=False,
    )

    assert_query_contract(result=_result(labels=(("namespace", "ns-a"),)), contract=contract)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param("transport_error", "ConnectTimeout", "transport failed", id="test_transport_error"),
        pytest.param("parse_error", "JSONDecodeError", "could not be parsed", id="test_parse_error"),
        pytest.param("backend_status", "error", "backend response contained an error", id="test_backend_status"),
    ],
)
def test_pipeline_health_contract_rejects_signal_errors_before_result_counts(
    field: str,
    value: str,
    message: str,
) -> None:
    """Given a signal error with an otherwise valid response shape, reject it before evaluating result counts."""
    result = replace(_signal_result(), **{field: value})

    with pytest.raises(QueryContractError, match=message):
        assert_pipeline_health_contract(result=result, contract=_pipeline_health_contract())
