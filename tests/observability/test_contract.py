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
CONTRACT_DOCUMENTATION_PATH = Path(__file__).parents[2] / "docs" / "observability-release-contract.md"

EXPECTED_PUBLIC_PANEL_CAPABILITIES = {
    ("cluster", "system-health"): "shipped",
    ("cluster", "deployed-models"): "shipped",
    ("cluster", "gpu-utilization"): "shipped",
    ("cluster", "gpu-utilization-by-project"): "shipped",
    ("cluster", "cpu"): "shipped",
    ("cluster", "memory"): "shipped",
    ("cluster", "network"): "shipped",
    ("models", "model-deployment-variable"): "shipped",
    ("models", "model-table"): "shipped",
    ("models", "request-queue"): "shipped",
    ("models", "replicas"): "shipped",
    ("models", "latency"): "shipped",
    ("models", "ttft"): "not-shipped",
    ("models", "token-generation"): "not-shipped",
    ("models", "throughput"): "not-shipped",
    ("models", "response-distribution"): "not-shipped",
    ("models", "vllm-inference"): "shipped",
    ("accelerators", "dcgm-gpu-utilization"): "shipped",
    ("accelerators", "gpu-utilization"): "shipped",
    ("accelerators", "memory-used"): "shipped",
    ("accelerators", "temperature"): "not-shipped",
    ("accelerators", "power"): "not-shipped",
    ("accelerators", "gpu-aas-regression"): "environment-blocked",
    ("maas", "token-usage"): "shipped",
    ("maas", "request-usage"): "shipped",
    ("maas", "rate-limited-requests"): "not-shipped",
    ("maas", "prompt-completion-token-breakdown"): "not-shipped",
    ("maas", "showback"): "not-shipped",
    ("cluster", "namespace-proxy-isolation"): "shipped",
    ("cluster", "tenancy-isolation"): "shipped",
    ("cluster", "data-science-thanos-route"): "shipped",
}


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


def test_public_contract_exposes_stable_dashboard_panels_and_capabilities() -> None:
    """Given the public YAML contract, expose stable dashboard/panel identifiers and capability states."""
    contract = load_release_contract(source=CONTRACT_PATH)

    actual = {(record.dashboard, record.panel): record.capability for record in contract.records}

    assert actual == EXPECTED_PUBLIC_PANEL_CAPABILITIES
    assert all(record.empty_ui_state for record in contract.records)


def test_contract_ref_selection_is_explicit_and_independent_of_dashboard_branch() -> None:
    """Given the consumer contract documentation, require an explicit tests-repository ref for YAML retrieval."""
    documentation = CONTRACT_DOCUMENTATION_PATH.read_text(encoding="utf-8")

    assert "RHOAI_OBSERVABILITY_CONTRACT_REF" in documentation
    assert "DASHBOARD_BRANCH" in documentation
    assert (
        "https://raw.githubusercontent.com/opendatahub-io/opendatahub-tests/<RHOAI_OBSERVABILITY_CONTRACT_REF>/"
        "tests/observability/contracts/release_contract.yaml"
    ) in documentation
    assert "must not silently fall back" in documentation


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


def test_contract_rejects_duplicate_dashboard_panel_mapping() -> None:
    """Given two records for one dashboard/panel, reject an ambiguous public contract."""
    record = {
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
        "authorization_response": "not-applicable",
    }

    with pytest.raises(ContractValidationError, match="duplicate dashboard/panel mapping"):
        load_release_contract(
            source={
                "contract_version": "1.0.0",
                "release_stage": "GA",
                "product_versions": {"rhoai": "test"},
                "records": [record, {**record, "id": "another-query"}],
            }
        )


def test_contract_rejects_unsupported_record_fields() -> None:
    """Given a record field outside the contract schema, reject silently ignored release behavior."""
    with pytest.raises(ContractValidationError, match="unsupported fields"):
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
                        "empty_ui_state": "No data",
                        "capability": "shipped",
                        "authorization_response": "not-applicable",
                        "ui_display_name": "Unsupported",
                    }
                ],
            }
        )


def test_contract_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    """Given duplicate YAML keys, reject the contract instead of accepting parser-dependent data."""
    contract_path = tmp_path / "duplicate.yaml"
    contract_path.write_text(
        data=(
            "contract_version: 1.0.0\n"
            "contract_version: 2.0.0\n"
            "release_stage: GA\n"
            "product_versions: {rhoai: test}\n"
            "records: []\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ContractValidationError, match="duplicate release contract key"):
        load_release_contract(source=contract_path)


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
