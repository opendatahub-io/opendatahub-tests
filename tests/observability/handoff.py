"""Typed dashboard handoff generation for the integrated observability contract."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode

from tests.observability.contract import ContractRecord, ReleaseContract
from tests.observability.evidence import sanitize_evidence_value, write_text_atomically
from tests.observability.fixtures import NamespacePair, source_metric_available
from tests.observability.personas import Persona
from tests.observability.query import RawQueryResult

HANDOFF_SCHEMA_VERSION = "1.0.0"
UI_METADATA_SCHEMA_VERSION = "1.0.0"
TRACKING_ID = "RHOAIENG-96476"
AUTHORIZATION_OUTCOMES = {
    "403": "forbidden",
    "404": "forbidden",
    "success-empty": "empty",
    "success-filtered": "filtered",
}
CAPABILITY_ORDER = ("not-shipped", "environment-blocked", "shipped")


class DashboardHandoffValidationError(ValueError):
    """Raised when the dashboard handoff cannot be generated safely."""


@dataclass(frozen=True)
class ModelSelectorMetadata:
    """UI-only model selector labels, validated independently of the release contract."""

    variable_name: str
    display_name: str
    namespace_variable_name: str


@dataclass(frozen=True)
class PanelMetadata:
    """UI-only metadata for one canonical dashboard panel."""

    identifier: str
    display_name: str


@dataclass(frozen=True)
class DashboardMetadata:
    """UI metadata that does not duplicate contract behavior."""

    name: str
    display_name: str
    panels: tuple[PanelMetadata, ...]
    model_selector: ModelSelectorMetadata | None = None


@dataclass(frozen=True)
class DashboardMetadataDocument:
    """Complete dashboard UI metadata input."""

    schema_version: str
    dashboards: tuple[DashboardMetadata, ...]


@dataclass(frozen=True)
class DashboardHandoffRuntime:
    """Sanitized runtime values captured while fixture resources are alive."""

    namespace_a: str
    namespace_b: str
    seeded_model_name: str
    foreign_model_names: tuple[str, ...]
    source_telemetry_ready: bool
    readiness_signal: str | None
    personas: tuple[Persona, ...]
    evidence_directory: Path
    run_id: str


@dataclass(frozen=True)
class HandoffRelease:
    """Release identity exposed to the dashboard consumer."""

    stage: str
    dashboard_version: str
    image_version: str

    def to_dict(self) -> dict[str, str]:
        """Serialize release identity using consumer-compatible names."""
        return {
            "stage": self.stage,
            "dashboardVersion": self.dashboard_version,
            "imageVersion": self.image_version,
        }


@dataclass(frozen=True)
class HandoffFixture:
    """Live namespace, model, and source-readiness values."""

    namespace_a: str
    namespace_b: str
    seeded_model_name: str
    foreign_model_names: tuple[str, ...]
    source_telemetry_ready: bool
    readiness_signal: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize fixture values using consumer-compatible names."""
        return {
            "namespaceA": self.namespace_a,
            "namespaceB": self.namespace_b,
            "seededModelName": self.seeded_model_name,
            "foreignModelNames": list(self.foreign_model_names),
            "sourceTelemetryReady": self.source_telemetry_ready,
            "readinessSignal": self.readiness_signal,
        }


@dataclass(frozen=True)
class HandoffAuthorization:
    """Reviewed namespace-isolation outcome exposed to the dashboard suite."""

    unauthorized_namespace_outcome: str
    foreign_data_must_not_render: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize the reviewed authorization outcome."""
        return {
            "unauthorizedNamespaceOutcome": self.unauthorized_namespace_outcome,
            "foreignDataMustNotRender": self.foreign_data_must_not_render,
        }


@dataclass(frozen=True)
class HandoffPanel:
    """Dashboard panel capability and expected UI state derived from one contract record."""

    identifier: str
    display_name: str
    capability: str
    expected_state: str
    empty_ui_state: str

    def to_dict(self) -> dict[str, str]:
        """Serialize panel fields using consumer-compatible names."""
        return {
            "id": self.identifier,
            "displayName": self.display_name,
            "capability": self.capability,
            "expectedState": self.expected_state,
            "emptyUiState": self.empty_ui_state,
        }


@dataclass(frozen=True)
class HandoffDashboard:
    """One dashboard and its canonical-record-derived panels."""

    name: str
    display_name: str
    capability: str
    panels: tuple[HandoffPanel, ...]
    model_selector: ModelSelectorMetadata | None

    def to_dict(self) -> dict[str, object]:
        """Serialize dashboard metadata and panels."""
        payload: dict[str, object] = {
            "name": self.name,
            "displayName": self.display_name,
            "capability": self.capability,
            "panels": [panel.to_dict() for panel in self.panels],
        }
        if self.model_selector is not None:
            payload["modelSelector"] = {
                "variableName": self.model_selector.variable_name,
                "displayName": self.model_selector.display_name,
                "namespaceVariableName": self.model_selector.namespace_variable_name,
            }
        return payload


@dataclass(frozen=True)
class HandoffPersona:
    """Sanitized persona configuration for the dashboard browser suite."""

    identifier: str
    credential_variable: str
    namespace_scope: str
    unauthorized_namespace_scope: str
    visible_dashboard_names: tuple[str, ...]
    hidden_dashboard_names: tuple[str, ...]
    load_shipped_dashboards: bool
    model_dashboard_name: str | None

    def to_dict(self) -> dict[str, object]:
        """Serialize persona configuration without principals or credentials."""
        return {
            "id": self.identifier,
            "credentialVariable": self.credential_variable,
            "namespaceScope": self.namespace_scope,
            "unauthorizedNamespaceScope": self.unauthorized_namespace_scope,
            "visibleDashboardNames": list(self.visible_dashboard_names),
            "hiddenDashboardNames": list(self.hidden_dashboard_names),
            "loadShippedDashboards": self.load_shipped_dashboards,
            "modelDashboardName": self.model_dashboard_name,
        }


@dataclass(frozen=True)
class HandoffEvidence:
    """Evidence location passed to the dashboard consumer."""

    directory: str
    run_id: str

    def to_dict(self) -> dict[str, str]:
        """Serialize evidence location using consumer-compatible names."""
        return {"directory": self.directory, "runId": self.run_id}


@dataclass(frozen=True)
class DashboardHandoff:
    """Complete deterministic dashboard handoff document."""

    release: HandoffRelease
    fixture: HandoffFixture
    authorization: HandoffAuthorization
    dashboards: tuple[HandoffDashboard, ...]
    personas: tuple[HandoffPersona, ...]
    evidence: HandoffEvidence
    schema_version: str = HANDOFF_SCHEMA_VERSION
    jira_key: str = TRACKING_ID

    def to_dict(self) -> dict[str, object]:
        """Serialize the handoff with the dashboard consumer's camelCase schema."""
        return {
            "schemaVersion": self.schema_version,
            "jiraKey": self.jira_key,
            "release": self.release.to_dict(),
            "fixture": self.fixture.to_dict(),
            "authorization": self.authorization.to_dict(),
            "dashboards": [dashboard.to_dict() for dashboard in self.dashboards],
            "personas": [persona.to_dict() for persona in self.personas],
            "evidence": self.evidence.to_dict(),
        }


def load_dashboard_metadata(source: str | Path | Mapping[str, Any]) -> DashboardMetadataDocument:
    """Load and validate versioned UI metadata without accepting contract behavior fields."""
    raw: Any
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as metadata_file:
            try:
                raw = yaml.load(metadata_file, Loader=_UniqueKeyLoader)
            except DashboardHandoffValidationError:
                raise
            except yaml.YAMLError as error:
                raise DashboardHandoffValidationError(f"dashboard metadata YAML is invalid: {error}") from error
    else:
        raw = source
    if not isinstance(raw, dict):
        raise DashboardHandoffValidationError("dashboard metadata must be a mapping")
    _reject_unknown_keys(source=raw, allowed={"schema_version", "dashboards"}, context="dashboard metadata")
    schema_version = _required_string(source=raw, key="schema_version", context="dashboard metadata")
    if schema_version != UI_METADATA_SCHEMA_VERSION:
        raise DashboardHandoffValidationError(f"dashboard metadata schema_version must be {UI_METADATA_SCHEMA_VERSION}")
    raw_dashboards = raw.get("dashboards")
    if not isinstance(raw_dashboards, list) or not raw_dashboards:
        raise DashboardHandoffValidationError("dashboard metadata dashboards must be a non-empty list")

    dashboards: list[DashboardMetadata] = []
    names: set[str] = set()
    for index, raw_dashboard in enumerate(raw_dashboards):
        context = f"dashboard metadata dashboards[{index}]"
        if not isinstance(raw_dashboard, dict):
            raise DashboardHandoffValidationError(f"{context} must be a mapping")
        _reject_unknown_keys(
            source=raw_dashboard,
            allowed={"name", "display_name", "panels", "model_selector"},
            context=context,
        )
        name = _required_string(source=raw_dashboard, key="name", context=context)
        if name in names:
            raise DashboardHandoffValidationError(f"duplicate UI dashboard mapping: {name}")
        names.add(name)
        raw_panels = raw_dashboard.get("panels")
        if not isinstance(raw_panels, list) or not raw_panels:
            raise DashboardHandoffValidationError(f"{context}.panels must be a non-empty list")
        panels: list[PanelMetadata] = []
        panel_names: set[str] = set()
        for panel_index, raw_panel in enumerate(raw_panels):
            panel_context = f"{context}.panels[{panel_index}]"
            if not isinstance(raw_panel, dict):
                raise DashboardHandoffValidationError(f"{panel_context} must be a mapping")
            _reject_unknown_keys(source=raw_panel, allowed={"id", "display_name"}, context=panel_context)
            identifier = _required_string(source=raw_panel, key="id", context=panel_context)
            if identifier in panel_names:
                raise DashboardHandoffValidationError(f"duplicate UI panel mapping: {name}/{identifier}")
            panel_names.add(identifier)
            panels.append(
                PanelMetadata(
                    identifier=identifier,
                    display_name=_required_string(source=raw_panel, key="display_name", context=panel_context),
                )
            )
        raw_selector = raw_dashboard.get("model_selector")
        selector = None
        if raw_selector is not None:
            selector_context = f"{context}.model_selector"
            if not isinstance(raw_selector, dict):
                raise DashboardHandoffValidationError(f"{selector_context} must be a mapping")
            _reject_unknown_keys(
                source=raw_selector,
                allowed={"variable_name", "display_name", "namespace_variable_name"},
                context=selector_context,
            )
            selector = ModelSelectorMetadata(
                variable_name=_required_string(source=raw_selector, key="variable_name", context=selector_context),
                display_name=_required_string(source=raw_selector, key="display_name", context=selector_context),
                namespace_variable_name=_required_string(
                    source=raw_selector,
                    key="namespace_variable_name",
                    context=selector_context,
                ),
            )
        dashboards.append(
            DashboardMetadata(
                name=name,
                display_name=_required_string(source=raw_dashboard, key="display_name", context=context),
                panels=tuple(panels),
                model_selector=selector,
            )
        )
    return DashboardMetadataDocument(schema_version=schema_version, dashboards=tuple(dashboards))


def runtime_from_fixtures(
    *,
    namespaces: NamespacePair,
    models: Sequence[object],
    source_metrics: Mapping[str, RawQueryResult],
    readiness_signal: str | None,
    personas: Sequence[Persona],
    evidence_directory: Path,
    run_id: str,
) -> DashboardHandoffRuntime:
    """Build handoff runtime context from live fixture objects and the source readiness gate."""
    namespace_a, namespace_b = namespaces.names
    model_names_by_namespace: dict[str, str] = {}
    for model in models:
        model_name = getattr(model, "name", None)
        model_namespace = getattr(model, "namespace", None)
        if not isinstance(model_name, str) or not model_name.strip():
            raise DashboardHandoffValidationError("created model fixtures must expose a non-empty name")
        if not isinstance(model_namespace, str) or not model_namespace.strip():
            raise DashboardHandoffValidationError("created model fixtures must expose name and namespace")
        if model_namespace in model_names_by_namespace:
            raise DashboardHandoffValidationError(f"multiple model fixtures found in namespace {model_namespace}")
        model_names_by_namespace[model_namespace] = model_name
    if namespace_a not in model_names_by_namespace:
        raise DashboardHandoffValidationError(f"no seeded model fixture found in namespace {namespace_a}")
    if namespace_b not in model_names_by_namespace:
        raise DashboardHandoffValidationError(f"no foreign model fixture found in namespace {namespace_b}")

    invalid_source_metrics = [
        identifier for identifier, result in source_metrics.items() if not source_metric_available(result=result)
    ]
    if invalid_source_metrics:
        raise DashboardHandoffValidationError(
            "source telemetry readiness gate returned unavailable records: " + ", ".join(sorted(invalid_source_metrics))
        )
    if source_metrics and readiness_signal not in source_metrics:
        raise DashboardHandoffValidationError(
            f"readiness_signal {readiness_signal!r} is not present in source telemetry results"
        )
    if not source_metrics and readiness_signal is not None:
        raise DashboardHandoffValidationError("readiness_signal must be empty when source telemetry results are empty")
    return DashboardHandoffRuntime(
        namespace_a=namespace_a,
        namespace_b=namespace_b,
        seeded_model_name=model_names_by_namespace[namespace_a],
        foreign_model_names=tuple(
            sorted(model_name for namespace, model_name in model_names_by_namespace.items() if namespace != namespace_a)
        ),
        source_telemetry_ready=bool(source_metrics),
        readiness_signal=readiness_signal,
        personas=tuple(personas),
        evidence_directory=evidence_directory,
        run_id=run_id,
    )


def build_dashboard_handoff(
    *,
    contract: ReleaseContract,
    metadata: DashboardMetadataDocument,
    runtime: DashboardHandoffRuntime,
) -> DashboardHandoff:
    """Build a deterministic dashboard handoff from validated contract and runtime inputs."""
    _validate_runtime(runtime=runtime)
    records_by_panel = validate_dashboard_handoff_configuration(contract=contract, metadata=metadata)
    handoff_personas = validate_persona_handoff_configuration(
        personas=runtime.personas,
        dashboard_names={dashboard.name for dashboard in metadata.dashboards},
    )
    authorization_outcome = _authorization_outcome(contract=contract)
    dashboards: list[HandoffDashboard] = []
    for dashboard_metadata in sorted(metadata.dashboards, key=lambda item: item.name):
        panels: list[HandoffPanel] = []
        records: list[ContractRecord] = []
        for panel_metadata in sorted(dashboard_metadata.panels, key=lambda item: item.identifier):
            record = records_by_panel[(dashboard_metadata.name, panel_metadata.identifier)]
            records.append(record)
            panels.append(
                HandoffPanel(
                    identifier=panel_metadata.identifier,
                    display_name=panel_metadata.display_name,
                    capability=record.capability,
                    expected_state=_expected_state(record=record),
                    empty_ui_state=record.empty_ui_state,
                )
            )
        dashboards.append(
            HandoffDashboard(
                name=dashboard_metadata.name,
                display_name=dashboard_metadata.display_name,
                capability=_dashboard_capability(records=records),
                panels=tuple(panels),
                model_selector=dashboard_metadata.model_selector,
            )
        )

    dashboard_version = _required_product_version(contract=contract, name="dashboard")
    handoff = DashboardHandoff(
        release=HandoffRelease(
            stage=contract.release_stage,
            dashboard_version=dashboard_version,
            image_version=dashboard_version,
        ),
        fixture=HandoffFixture(
            namespace_a=runtime.namespace_a,
            namespace_b=runtime.namespace_b,
            seeded_model_name=runtime.seeded_model_name,
            foreign_model_names=tuple(sorted(runtime.foreign_model_names)),
            source_telemetry_ready=runtime.source_telemetry_ready,
            readiness_signal=runtime.readiness_signal,
        ),
        authorization=HandoffAuthorization(
            unauthorized_namespace_outcome=authorization_outcome,
            foreign_data_must_not_render=True,
        ),
        dashboards=tuple(dashboards),
        personas=handoff_personas,
        evidence=HandoffEvidence(directory=str(runtime.evidence_directory), run_id=runtime.run_id),
    )
    _validate_public_handoff(value=handoff.to_dict())
    return handoff


def write_dashboard_handoff(destination: str | Path, handoff: DashboardHandoff) -> Path:
    """Write a deterministic dashboard handoff with sorted keys, indentation, and a final newline."""
    path = Path(destination)
    payload = handoff.to_dict()
    _validate_public_handoff(value=payload)
    write_text_atomically(path=path, content=json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def validate_dashboard_handoff_configuration(
    *,
    contract: ReleaseContract,
    metadata: DashboardMetadataDocument,
) -> dict[tuple[str, str], ContractRecord]:
    """Validate static handoff inputs before any cluster resources are created."""
    records_by_panel = validate_dashboard_metadata_mappings(contract=contract, metadata=metadata)
    _required_product_version(contract=contract, name="dashboard")
    return records_by_panel


def validate_dashboard_metadata_mappings(
    *,
    contract: ReleaseContract,
    metadata: DashboardMetadataDocument,
) -> dict[tuple[str, str], ContractRecord]:
    """Validate the one-to-one UI metadata mapping against the canonical contract."""
    return _validate_ui_mappings(contract=contract, metadata=metadata)


def validate_persona_handoff_configuration(
    *,
    personas: Sequence[Persona],
    dashboard_names: set[str],
) -> tuple[HandoffPersona, ...]:
    """Validate persona fields required by the dashboard consumer before resource creation."""
    return tuple(
        sorted(
            (_build_persona_handoff(persona=persona, dashboard_names=dashboard_names) for persona in personas),
            key=lambda persona: persona.identifier,
        )
    )


def _validate_ui_mappings(
    *,
    contract: ReleaseContract,
    metadata: DashboardMetadataDocument,
) -> dict[tuple[str, str], ContractRecord]:
    canonical: dict[tuple[str, str], ContractRecord] = {}
    for record in contract.records:
        key = (record.dashboard, record.panel)
        if key in canonical:
            raise DashboardHandoffValidationError(
                f"canonical contract has duplicate dashboard/panel mapping: {record.dashboard}/{record.panel}"
            )
        canonical[key] = record
    metadata_keys = {
        (dashboard.name, panel.identifier) for dashboard in metadata.dashboards for panel in dashboard.panels
    }
    unknown = sorted(metadata_keys - canonical.keys())
    missing = sorted(canonical.keys() - metadata_keys)
    if unknown:
        raise DashboardHandoffValidationError(
            "unknown UI dashboard/panel mapping: " + ", ".join(_format_key(key) for key in unknown)
        )
    if missing:
        raise DashboardHandoffValidationError(
            "missing UI dashboard/panel mapping: " + ", ".join(_format_key(key) for key in missing)
        )
    return canonical


def _authorization_outcome(contract: ReleaseContract) -> str:
    responses = {
        record.authorization_response
        for record in contract.records
        if record.authorization_response != "not-applicable"
    }
    if not responses:
        raise DashboardHandoffValidationError("contract has no relevant authorization response records")
    if "review-required" in responses:
        raise DashboardHandoffValidationError(
            "authorization handoff cannot be generated while a record is review-required"
        )
    if len(responses) != 1:
        raise DashboardHandoffValidationError(
            "relevant authorization records disagree: " + ", ".join(sorted(responses))
        )
    response = next(iter(responses))
    try:
        return AUTHORIZATION_OUTCOMES[response]
    except KeyError as error:
        raise DashboardHandoffValidationError(
            f"authorization response {response!r} cannot map to forbidden, empty, or filtered"
        ) from error


def _expected_state(record: ContractRecord) -> str:
    if record.capability == "shipped":
        return "valid-empty" if record.empty_result_valid else "non-empty"
    return record.capability


def _dashboard_capability(records: Sequence[ContractRecord]) -> str:
    return max((record.capability for record in records), key=CAPABILITY_ORDER.index)


def _build_persona_handoff(*, persona: Persona, dashboard_names: set[str]) -> HandoffPersona:
    required_values = {
        "credential_variable": persona.credential_variable,
        "namespace_scope": persona.namespace_scope,
        "unauthorized_namespace_scope": persona.unauthorized_namespace_scope,
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        raise DashboardHandoffValidationError(
            f"persona {persona.name!r} is missing handoff configuration: {', '.join(missing)}"
        )
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", str(persona.credential_variable)):
        raise DashboardHandoffValidationError(
            f"persona {persona.name!r} credential_variable must be an environment variable name"
        )
    if "*" not in persona.namespaces and persona.namespace_scope not in persona.namespaces:
        raise DashboardHandoffValidationError(
            f"persona {persona.name!r} namespace_scope is inconsistent with namespaces"
        )
    if "*" not in persona.namespaces and persona.unauthorized_namespace_scope in persona.namespaces:
        raise DashboardHandoffValidationError(
            f"persona {persona.name!r} unauthorized_namespace_scope is within namespaces"
        )
    if persona.namespace_scope == persona.unauthorized_namespace_scope:
        raise DashboardHandoffValidationError(
            f"persona {persona.name!r} has identical authorized and unauthorized namespace scopes"
        )
    if persona.visible_dashboard_names is None or persona.hidden_dashboard_names is None:
        raise DashboardHandoffValidationError(f"persona {persona.name!r} is missing dashboard visibility metadata")
    if persona.load_shipped_dashboards is None:
        raise DashboardHandoffValidationError(f"persona {persona.name!r} is missing load_shipped_dashboards")
    visible = tuple(sorted(set(persona.visible_dashboard_names)))
    hidden = tuple(sorted(set(persona.hidden_dashboard_names)))
    unknown = (set(visible) | set(hidden)) - dashboard_names
    if unknown:
        raise DashboardHandoffValidationError(
            f"persona {persona.name!r} references unknown dashboards: {', '.join(sorted(unknown))}"
        )
    overlap = set(visible) & set(hidden)
    if overlap:
        raise DashboardHandoffValidationError(
            f"persona {persona.name!r} marks dashboards both visible and hidden: {', '.join(sorted(overlap))}"
        )
    if persona.model_dashboard_name is not None and persona.model_dashboard_name not in dashboard_names:
        raise DashboardHandoffValidationError(
            f"persona {persona.name!r} references unknown model dashboard: {persona.model_dashboard_name}"
        )
    return HandoffPersona(
        identifier=persona.name,
        credential_variable=str(persona.credential_variable),
        namespace_scope=str(persona.namespace_scope),
        unauthorized_namespace_scope=str(persona.unauthorized_namespace_scope),
        visible_dashboard_names=visible,
        hidden_dashboard_names=hidden,
        load_shipped_dashboards=persona.load_shipped_dashboards,
        model_dashboard_name=persona.model_dashboard_name,
    )


def _validate_runtime(runtime: DashboardHandoffRuntime) -> None:
    for name, value in (
        ("namespace_a", runtime.namespace_a),
        ("namespace_b", runtime.namespace_b),
        ("seeded_model_name", runtime.seeded_model_name),
        ("run_id", runtime.run_id),
    ):
        if not value or not value.strip():
            raise DashboardHandoffValidationError(f"runtime {name} must be a non-empty value")
    if runtime.namespace_a == runtime.namespace_b:
        raise DashboardHandoffValidationError("runtime namespace_a and namespace_b must be different")
    if not runtime.foreign_model_names:
        raise DashboardHandoffValidationError("runtime foreign_model_names must not be empty")
    if len(set(runtime.foreign_model_names)) != len(runtime.foreign_model_names):
        raise DashboardHandoffValidationError("runtime foreign_model_names must be unique")
    if runtime.seeded_model_name in runtime.foreign_model_names:
        raise DashboardHandoffValidationError("runtime seeded_model_name must not be a foreign model name")
    if runtime.source_telemetry_ready and not runtime.readiness_signal:
        raise DashboardHandoffValidationError("ready source telemetry must include readiness_signal")
    if not runtime.source_telemetry_ready and runtime.readiness_signal:
        raise DashboardHandoffValidationError("blocked source telemetry must not include readiness_signal")
    if not runtime.personas:
        raise DashboardHandoffValidationError("runtime personas must not be empty")
    persona_names = [persona.name for persona in runtime.personas]
    if not all(persona_names) or len(set(persona_names)) != len(persona_names):
        raise DashboardHandoffValidationError("runtime persona names must be non-empty and unique")


def _required_product_version(*, contract: ReleaseContract, name: str) -> str:
    version = contract.product_versions.get(name)
    if not version or version.startswith("${"):
        raise DashboardHandoffValidationError(
            f"contract product_versions.{name} must be resolved before handoff generation"
        )
    return version


def _validate_public_handoff(*, value: object) -> None:
    """Reuse the evidence sanitizer without redacting the public authorization object."""
    _validate_public_strings(value=value)


def _validate_public_strings(value: object) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _validate_public_strings(value=item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _validate_public_strings(value=item)
    elif isinstance(value, str) and sanitize_evidence_value(value=value) != value:
        raise DashboardHandoffValidationError("dashboard handoff contains a secret-like value")


def _required_string(source: Mapping[str, Any], key: str, context: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DashboardHandoffValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _reject_unknown_keys(source: Mapping[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(source) - allowed
    if unknown:
        raise DashboardHandoffValidationError(f"{context} contains unsupported fields: {', '.join(sorted(unknown))}")


def _format_key(key: tuple[str, str]) -> str:
    return f"{key[0]}/{key[1]}"


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys instead of silently overwriting them."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
    if not isinstance(node, MappingNode):
        raise DashboardHandoffValidationError("dashboard metadata root must be a YAML mapping")
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise DashboardHandoffValidationError(f"duplicate dashboard metadata key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    tag=yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    constructor=_construct_unique_mapping,
)
