from pathlib import Path

import pytest

from tests.observability.contract import (
    ContractValidationError,
    ensure_authorization_reviewed,
    load_release_contract,
    render_promql,
)

pytestmark = pytest.mark.tier1

CONTRACT_PATH = Path(__file__).parent / "contracts" / "release_contract.yaml"


def test_release_contract_contains_required_matrix_entries() -> None:
    """Given the reviewed contract, verify all release-panel categories are represented."""
    contract = load_release_contract(source=CONTRACT_PATH)

    identifiers = {record.identifier for record in contract.records}

    assert contract.version
    assert contract.release_stage in {"EA1", "EA2", "GA"}
    assert contract.record("showback").capability == "not-shipped"
    assert {
        "cluster-system-health",
        "cluster-gpu-utilization",
        "models-model-table",
        "accelerator-gpu-utilization",
        "gpu-aas-regression",
        "inference-vllm-series",
        "maas-authorized-hits",
        "token-breakdown",
        "showback",
    } <= identifiers


def test_render_promql_replaces_fixture_variables_without_leaking_placeholders() -> None:
    """Given a contract query, render fixture variables and reject unresolved values."""
    contract = load_release_contract(source=CONTRACT_PATH)
    record = next(item for item in contract.records if item.identifier == "models-model-table")

    rendered = render_promql(
        promql=record.promql,
        namespace="observability-a",
        model="model-a",
    )

    assert "observability-a" in rendered
    assert "model-a" in rendered
    assert "${" not in rendered


def test_contract_resolves_product_version_environment_values(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given a release version environment value, replace its contract placeholder without changing other fields."""
    monkeypatch.setenv(name="RHOAI_VERSION", value="3.3.0")

    contract = load_release_contract(source=CONTRACT_PATH)

    assert contract.product_versions["rhoai"] == "3.3.0"


def test_contract_rejects_unreviewed_authorization_response() -> None:
    """Given an unresolved denial contract, prevent namespace assertions from accepting any response."""
    contract = load_release_contract(
        source={
            "contract_version": "1.0.0",
            "release_stage": "GA",
            "product_versions": {"rhoai": "test"},
            "records": [
                {
                    "id": "query",
                    "dashboard": "cluster",
                    "panel": "panel",
                    "datasource": "thanos",
                    "route": "/api/v1/query",
                    "promql": "up",
                    "expected_http_status": [200],
                    "expected_prometheus_status": "success",
                    "expected_result_type": "vector",
                    "minimum_series": 0,
                    "required_labels": [],
                    "empty_result_valid": True,
                    "empty_ui_state": "No data",
                    "capability": "shipped",
                    "authorization_response": "review-required",
                }
            ],
        }
    )

    with pytest.raises(ContractValidationError, match="authorization response"):
        ensure_authorization_reviewed(record=contract.records[0])


def test_contract_rejects_shipped_capability_with_unavailable_empty_state() -> None:
    """Given a shipped panel, reject a contract that treats an unavailable empty result as valid."""
    with pytest.raises(ContractValidationError, match="empty_ui_state"):
        load_release_contract(
            source={
                "contract_version": "1.0.0",
                "release_stage": "GA",
                "product_versions": {"rhoai": "test"},
                "records": [
                    {
                        "id": "query",
                        "dashboard": "cluster",
                        "panel": "panel",
                        "datasource": "thanos",
                        "route": "/api/v1/query",
                        "promql": "up",
                        "expected_http_status": [200],
                        "expected_prometheus_status": "success",
                        "expected_result_type": "vector",
                        "minimum_series": 0,
                        "required_labels": [],
                        "empty_result_valid": True,
                        "empty_ui_state": "Unavailable: not shipped",
                        "capability": "shipped",
                        "authorization_response": "not-applicable",
                    }
                ],
            }
        )
