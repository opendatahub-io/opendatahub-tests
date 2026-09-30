from dataclasses import replace

import pytest

from tests.observability.assertions import (
    QueryContractError,
    assert_authorization_response,
    assert_capability_is_available,
    assert_capability_unavailable,
    assert_namespace_isolation,
    assert_query_contract,
)
from tests.observability.contract import ContractRecord
from tests.observability.query import NormalizedSeries, RawQueryResult

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
