"""Assertions shared by release-contract query tests."""

from tests.observability.contract import ContractRecord
from tests.observability.query import RawQueryResult


class QueryContractError(AssertionError):
    """Raised when a raw response does not satisfy its reviewed contract."""


def assert_query_contract(result: RawQueryResult, contract: ContractRecord) -> None:
    """Assert transport, Prometheus, result type, labels, and empty-state behavior."""
    if result.http_status not in contract.expected_http_status:
        raise QueryContractError(
            f"{contract.identifier}: expected HTTP {contract.expected_http_status}, got {result.http_status}"
        )
    if result.prometheus_status != contract.expected_prometheus_status:
        raise QueryContractError(
            f"{contract.identifier}: expected Prometheus status {contract.expected_prometheus_status!r}, "
            f"got {result.prometheus_status!r}"
        )
    if result.result_type != contract.expected_result_type:
        raise QueryContractError(
            f"{contract.identifier}: expected result type {contract.expected_result_type!r}, got {result.result_type!r}"
        )
    if result.warnings and not contract.warnings_allowed:
        raise QueryContractError(f"{contract.identifier}: unexpected Prometheus warnings: {result.warnings}")
    if contract.expected_prometheus_status == "success" and (result.error_type or result.error):
        raise QueryContractError(
            f"{contract.identifier}: Prometheus success contained error fields: "
            f"error_type={result.error_type!r}, error={result.error!r}"
        )
    if len(result.series) < contract.minimum_series:
        raise QueryContractError(
            f"{contract.identifier}: expected at least {contract.minimum_series} series, got {len(result.series)}"
        )
    if not result.series and not contract.empty_result_valid:
        raise QueryContractError(f"{contract.identifier}: empty result is not a contracted UI state")
    if result.series:
        labels = {label for series in result.series for label, _value in series.labels}
        missing_labels = set(contract.required_labels) - labels
        if missing_labels:
            raise QueryContractError(f"{contract.identifier}: missing labels: {sorted(missing_labels)}")


def assert_namespace_isolation(result: RawQueryResult, allowed_namespaces: set[str]) -> None:
    """Reject any returned series carrying a namespace outside the persona scope."""
    if "*" in allowed_namespaces:
        return
    namespace_labels = {"namespace", "k8s_namespace_name", "namespace_name"}
    for series in result.series:
        labels = dict(series.labels)
        returned_namespaces = {labels[label_name] for label_name in namespace_labels if label_name in labels}
        if not returned_namespaces:
            raise QueryContractError(f"{result.datasource}: series has no namespace label for isolation validation")
        foreign_namespaces = returned_namespaces - allowed_namespaces
        if foreign_namespaces:
            raise QueryContractError(
                f"{result.datasource}: foreign namespace series returned: {sorted(foreign_namespaces)}"
            )


def assert_authorization_response(result: RawQueryResult, expected: str) -> None:
    """Assert the exact reviewed denial/filtering contract."""
    if expected == "review-required":
        raise QueryContractError("authorization response must be reviewed before a negative assertion runs")
    if expected in {"403", "404"}:
        if result.http_status != int(expected):
            raise QueryContractError(f"expected HTTP {expected}, got {result.http_status}")
        return
    if expected == "success-empty":
        if result.http_status is None or not 200 <= result.http_status < 300:
            raise QueryContractError(f"expected successful empty response, got HTTP {result.http_status}")
        if result.prometheus_status != "success" or result.series:
            raise QueryContractError("expected Prometheus success with no returned series")
        return
    if expected == "success-filtered":
        if result.http_status is None or not 200 <= result.http_status < 300:
            raise QueryContractError(f"expected successful filtered response, got HTTP {result.http_status}")
        if result.prometheus_status != "success":
            raise QueryContractError(f"expected Prometheus success, got {result.prometheus_status!r}")
        return
    if expected != "not-applicable":
        raise QueryContractError(f"unsupported authorization response contract: {expected}")


def assert_capability_is_available(contract: ContractRecord) -> None:
    """Prevent unavailable or environment-blocked capabilities from passing as product data."""
    if contract.capability == "not-shipped":
        raise QueryContractError(f"{contract.identifier}: capability is explicitly not-shipped")
    if contract.capability == "environment-blocked":
        raise QueryContractError(f"{contract.identifier}: capability is environment-blocked")


def assert_capability_unavailable(contract: ContractRecord) -> None:
    """Assert that a capability declaration is explicit rather than inferred from an empty query."""
    if contract.capability not in {"not-shipped", "environment-blocked"}:
        raise QueryContractError(f"{contract.identifier}: capability is shipped, not unavailable")
