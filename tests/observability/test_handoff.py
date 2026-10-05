import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from tests.observability.contract import ReleaseContract, load_release_contract
from tests.observability.fixtures import NamespacePair
from tests.observability.handoff import (
    DashboardHandoff,
    DashboardHandoffRuntime,
    DashboardHandoffValidationError,
    DashboardMetadataDocument,
    build_dashboard_handoff,
    load_dashboard_metadata,
    runtime_from_fixtures,
    validate_dashboard_metadata_mappings,
    write_dashboard_handoff,
)
from tests.observability.personas import Persona
from tests.observability.query import RawQueryResult

pytestmark = pytest.mark.tier1


def test_dashboard_handoff_is_consumer_compatible_and_deterministic(tmp_path: Path) -> None:
    """Given validated contract and runtime inputs, emit stable camelCase handoff data without credentials."""
    handoff = _build_handoff(tmp_path=tmp_path)
    destination = write_dashboard_handoff(destination=tmp_path / "handoff.json", handoff=handoff)

    written = destination.read_text(encoding="utf-8")
    assert written.endswith("\n")
    assert written == destination.read_text(encoding="utf-8")
    payload = json.loads(written)
    assert payload["schemaVersion"] == "1.0.0"
    assert payload["jiraKey"] == "RHOAIENG-96476"
    assert payload["release"] == {
        "dashboardVersion": "2.0.0",
        "imageVersion": "2.0.0",
        "stage": "GA",
    }
    assert payload["fixture"]["namespaceA"] == "observability-a"
    assert payload["fixture"]["seededModelName"] == "seeded-live"
    assert payload["fixture"]["foreignModelNames"] == ["foreign-live"]
    assert payload["personas"][0]["credentialVariable"] == "NAMESPACE_ADMIN_USER"
    assert payload["personas"][0]["namespaceScope"] == "observability-a"
    assert "admin-principal" not in written


def test_dashboard_handoff_maps_capability_and_empty_result_states(tmp_path: Path) -> None:
    """Given mixed canonical capabilities, preserve shipped, valid-empty, not-shipped, and blocked states."""
    payload = cast(dict[str, Any], _build_handoff(tmp_path=tmp_path).to_dict())
    panels = {panel["id"]: panel for dashboard in payload["dashboards"] for panel in dashboard["panels"]}

    assert panels["non-empty"]["expectedState"] == "non-empty"
    assert panels["non-empty"]["emptyUiState"] == "No data"
    assert panels["valid-empty"]["expectedState"] == "valid-empty"
    assert panels["valid-empty"]["emptyUiState"] == "Zero"
    assert panels["not-shipped"]["capability"] == "not-shipped"
    assert panels["not-shipped"]["expectedState"] == "not-shipped"
    assert panels["environment-blocked"]["capability"] == "environment-blocked"
    assert panels["environment-blocked"]["expectedState"] == "environment-blocked"


def test_dashboard_handoff_rejects_unknown_and_missing_ui_mappings(tmp_path: Path) -> None:
    """Given UI metadata that diverges from the canonical YAML records, fail with an actionable mapping error."""
    contract = _contract()
    runtime = _runtime(tmp_path=tmp_path)
    unknown_metadata = load_dashboard_metadata(source=_metadata(include_unknown=True))
    with pytest.raises(DashboardHandoffValidationError, match="unknown UI dashboard/panel mapping"):
        build_dashboard_handoff(contract=contract, metadata=unknown_metadata, runtime=runtime)

    missing_metadata = load_dashboard_metadata(source=_metadata(exclude_panel="environment-blocked"))
    with pytest.raises(DashboardHandoffValidationError, match="missing UI dashboard/panel mapping"):
        build_dashboard_handoff(contract=contract, metadata=missing_metadata, runtime=runtime)


def test_dashboard_metadata_rejects_duplicate_ui_mappings() -> None:
    """Given duplicate UI panel entries, reject the metadata instead of silently choosing one display name."""
    with pytest.raises(DashboardHandoffValidationError, match="duplicate UI panel mapping"):
        load_dashboard_metadata(source=_metadata(duplicate_panel=True))


def test_checked_in_dashboard_metadata_maps_every_canonical_record() -> None:
    """Given the checked-in UI metadata and release contract, reject dashboard or panel drift."""
    contract_path = Path(__file__).parent / "contracts" / "release_contract.yaml"
    metadata_path = Path(__file__).parent / "contracts" / "dashboard_metadata.yaml"

    contract = load_release_contract(source=contract_path)
    metadata = load_dashboard_metadata(source=metadata_path)

    validate_dashboard_metadata_mappings(contract=contract, metadata=metadata)


def test_dashboard_metadata_rejects_duplicate_yaml_keys(tmp_path: Path) -> None:
    """Given duplicate YAML keys, reject ambiguous metadata instead of accepting the last value."""
    metadata_path = tmp_path / "duplicate.yaml"
    metadata_path.write_text(
        data=(
            "schema_version: 1.0.0\n"
            "dashboards:\n"
            "  - name: cluster\n"
            "    name: duplicate-cluster\n"
            "    display_name: Cluster\n"
            "    panels:\n"
            "      - id: panel\n"
            "        display_name: Panel\n"
        ),
        encoding="utf-8",
    )

    with pytest.raises(DashboardHandoffValidationError, match="duplicate dashboard metadata key"):
        load_dashboard_metadata(source=metadata_path)


@pytest.mark.parametrize(
    ("responses", "message"),
    [
        (("review-required",), "review-required"),
        (("403", "success-filtered"), "disagree"),
        (("not-applicable",), "no relevant"),
        (("418",), "cannot map"),
    ],
)
def test_dashboard_handoff_fails_closed_for_authorization_outcomes(
    tmp_path: Path,
    responses: tuple[str, ...],
    message: str,
) -> None:
    """Given unreviewed, conflicting, or unmappable authorization records, refuse to create an isolation claim."""
    if responses == ("418",):
        base_contract = _contract()
        contract = replace(
            base_contract,
            records=tuple(replace(record, authorization_response="418") for record in base_contract.records),
        )
    else:
        raw_contract = _raw_contract()
        records = cast(list[dict[str, Any]], raw_contract["records"])
        for index, response in enumerate(responses):
            records[index]["authorization_response"] = response
        if len(responses) == 1 and responses[0] == "not-applicable":
            for record in records:
                record["authorization_response"] = "not-applicable"
        contract = load_release_contract(source=raw_contract)

    with pytest.raises(DashboardHandoffValidationError, match=message):
        build_dashboard_handoff(contract=contract, metadata=_metadata_document(), runtime=_runtime(tmp_path=tmp_path))


@pytest.mark.parametrize(
    ("response", "expected_outcome"),
    [
        ("403", "forbidden"),
        ("404", "forbidden"),
        ("success-empty", "empty"),
        ("success-filtered", "filtered"),
    ],
)
def test_dashboard_handoff_maps_each_reviewed_authorization_response(
    tmp_path: Path,
    response: str,
    expected_outcome: str,
) -> None:
    """Given one reviewed authorization response, derive only a supported dashboard isolation outcome."""
    contract = _contract()
    contract = replace(
        contract,
        records=tuple(replace(record, authorization_response=response) for record in contract.records),
    )

    handoff = build_dashboard_handoff(
        contract=contract,
        metadata=_metadata_document(),
        runtime=_runtime(tmp_path=tmp_path),
    )

    assert handoff.authorization.unauthorized_namespace_outcome == expected_outcome


def test_dashboard_handoff_rejects_secret_like_runtime_values(tmp_path: Path) -> None:
    """Given a secret-like runtime identifier, reject it rather than writing sensitive data to the handoff."""
    runtime = _runtime(tmp_path=tmp_path, run_id="Bearer raw-secret-token")

    with pytest.raises(DashboardHandoffValidationError, match="secret-like"):
        build_dashboard_handoff(contract=_contract(), metadata=_metadata_document(), runtime=runtime)


def test_dashboard_handoff_rejects_inconsistent_persona_namespace_scope(tmp_path: Path) -> None:
    """Given a dashboard scope that disagrees with authorization namespaces, refuse contradictory handoff data."""
    persona = replace(_personas()[0], namespace_scope="observability-b")
    runtime = replace(_runtime(tmp_path=tmp_path), personas=(persona,))

    with pytest.raises(DashboardHandoffValidationError, match="namespace_scope is inconsistent"):
        build_dashboard_handoff(contract=_contract(), metadata=_metadata_document(), runtime=runtime)


def test_runtime_context_uses_live_fixture_names_and_source_gate(tmp_path: Path) -> None:
    """Given live fixture objects and source readiness, derive values without hardcoding names."""
    namespace_a = SimpleNamespace(name="namespace-from-fixture")
    namespace_b = SimpleNamespace(name="foreign-namespace-from-fixture")
    models = [
        SimpleNamespace(name="seeded-model-from-fixture", namespace="namespace-from-fixture"),
        SimpleNamespace(name="foreign-model-from-fixture", namespace="foreign-namespace-from-fixture"),
    ]
    source_result = SimpleNamespace(http_status=200, prometheus_status="success", series=("series",))

    runtime = runtime_from_fixtures(
        namespaces=NamespacePair(namespace_a=cast(Any, namespace_a), namespace_b=cast(Any, namespace_b)),
        models=models,
        source_metrics={"source-readiness": cast(RawQueryResult, source_result)},
        readiness_signal="source-readiness",
        personas=_personas(),
        evidence_directory=tmp_path,
        run_id="run-from-fixture",
    )

    assert runtime.namespace_a == "namespace-from-fixture"
    assert runtime.namespace_b == "foreign-namespace-from-fixture"
    assert runtime.seeded_model_name == "seeded-model-from-fixture"
    assert runtime.foreign_model_names == ("foreign-model-from-fixture",)
    assert runtime.source_telemetry_ready is True
    assert runtime.readiness_signal == "source-readiness"


def _build_handoff(tmp_path: Path) -> DashboardHandoff:
    return build_dashboard_handoff(
        contract=_contract(),
        metadata=_metadata_document(),
        runtime=_runtime(tmp_path=tmp_path),
    )


def _contract() -> ReleaseContract:
    return load_release_contract(source=_raw_contract())


def _raw_contract() -> dict[str, object]:
    record_defaults = {
        "dashboard": "cluster",
        "datasource": "cluster-thanos",
        "route": "/api/v1/query",
        "promql": "up",
        "expected_http_status": [200],
        "expected_prometheus_status": "success",
        "expected_result_type": "vector",
        "minimum_series": 1,
        "required_labels": ["namespace"],
        "authorization_response": "success-filtered",
    }
    records = []
    for identifier, panel, capability, empty_result_valid in (
        ("non-empty-record", "non-empty", "shipped", False),
        ("valid-empty-record", "valid-empty", "shipped", True),
        ("not-shipped-record", "not-shipped", "not-shipped", True),
        ("environment-blocked-record", "environment-blocked", "environment-blocked", True),
    ):
        records.append({
            **record_defaults,
            "id": identifier,
            "panel": panel,
            "empty_result_valid": empty_result_valid,
            "empty_ui_state": "No data" if not empty_result_valid else "Zero",
            "capability": capability,
        })
    return {
        "contract_version": "1.0.0",
        "release_stage": "GA",
        "product_versions": {"dashboard": "2.0.0"},
        "records": records,
    }


def _metadata_document() -> DashboardMetadataDocument:
    return load_dashboard_metadata(source=_metadata())


def _metadata(
    *,
    include_unknown: bool = False,
    exclude_panel: str | None = None,
    duplicate_panel: bool = False,
) -> dict[str, object]:
    panels = [
        {"id": "non-empty", "display_name": "Non-empty"},
        {"id": "valid-empty", "display_name": "Valid empty"},
        {"id": "not-shipped", "display_name": "Not shipped"},
        {"id": "environment-blocked", "display_name": "Environment blocked"},
    ]
    if exclude_panel:
        panels = [panel for panel in panels if panel["id"] != exclude_panel]
    if include_unknown:
        panels.append({"id": "unknown", "display_name": "Unknown"})
    if duplicate_panel:
        panels.append({"id": "non-empty", "display_name": "Duplicate"})
    return {
        "schema_version": "1.0.0",
        "dashboards": [
            {
                "name": "cluster",
                "display_name": "Cluster",
                "panels": panels,
            }
        ],
    }


def _runtime(*, tmp_path: Path, run_id: str = "run-1") -> DashboardHandoffRuntime:
    return DashboardHandoffRuntime(
        namespace_a="observability-a",
        namespace_b="observability-b",
        seeded_model_name="seeded-live",
        foreign_model_names=("foreign-live",),
        source_telemetry_ready=True,
        readiness_signal="source-readiness",
        personas=_personas(),
        evidence_directory=tmp_path,
        run_id=run_id,
    )


def _personas() -> tuple[Persona, ...]:
    return (
        Persona(
            name="namespace-admin",
            principal="admin-principal",
            groups=("system:authenticated",),
            namespaces=("observability-a",),
            credential_variable="NAMESPACE_ADMIN_USER",
            namespace_scope="observability-a",
            unauthorized_namespace_scope="observability-b",
            visible_dashboard_names=("cluster",),
            hidden_dashboard_names=(),
            load_shipped_dashboards=True,
            model_dashboard_name="cluster",
        ),
    )
