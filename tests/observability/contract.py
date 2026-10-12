"""Versioned release-contract loading and PromQL rendering."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any

import yaml
from yaml.nodes import MappingNode

from tests.observability.rollout import PipelineStage, RolloutState

CAPABILITY_STATUSES = {"shipped", "not-shipped", "environment-blocked"}
RELEASE_STAGES = {"EA1", "EA2", "GA"}
AUTHORIZATION_RESPONSES = {
    "not-applicable",
    "403",
    "404",
    "success-empty",
    "success-filtered",
    "isolation-only",
    "review-required",
}
PIPELINE_SIGNALS = {"metrics", "traces", "logs", "correlation"}
CONTRACT_KEYS = {
    "contract_version",
    "release_stage",
    "product_versions",
    "default_time_range",
    "pipeline_health",
    "records",
}
RECORD_KEYS = {
    "id",
    "release_stage",
    "dashboard",
    "panel",
    "datasource",
    "route",
    "promql",
    "time_range",
    "expected_http_status",
    "expected_prometheus_status",
    "expected_result_type",
    "minimum_series",
    "required_labels",
    "empty_result_valid",
    "empty_ui_state",
    "capability",
    "authorization_response",
    "warnings_allowed",
}
PIPELINE_HEALTH_KEYS = {"marker", "defaults", "records"}
PIPELINE_RECORD_KEYS = {
    "id",
    "release_stage",
    "signal",
    "stage",
    "marker",
    "source_query",
    "backend_query",
    "required_labels",
    "required_attributes",
    "required_edges",
    "propagation_timeout_seconds",
    "query_window_seconds",
    "expected_http_status",
    "expected_authorization",
    "minimum_results",
    "empty_result_valid",
    "rollout_state",
    "activation_release",
    "reason",
    "next_activation_step",
}


class ContractValidationError(ValueError):
    """Raised when a release contract is incomplete or ambiguous."""


@dataclass(frozen=True)
class ContractRecord:
    """A single dashboard panel query contract."""

    identifier: str
    release_stage: str
    product_versions: dict[str, str]
    dashboard: str
    panel: str
    datasource: str
    route: str
    promql: str
    time_range: dict[str, str | int | float]
    expected_http_status: tuple[int, ...]
    expected_prometheus_status: str
    expected_result_type: str
    minimum_series: int
    required_labels: tuple[str, ...]
    empty_result_valid: bool
    empty_ui_state: str
    capability: str
    authorization_response: str
    warnings_allowed: bool


@dataclass(frozen=True)
class PipelineHealthRecord:
    """A signal-specific, non-PromQL pipeline-health contract."""

    identifier: str
    release_stage: str
    signal: str
    stage: PipelineStage
    marker: str
    source_query: dict[str, Any]
    backend_query: dict[str, Any]
    required_labels: tuple[str, ...]
    required_attributes: tuple[str, ...]
    required_edges: tuple[str, ...]
    propagation_timeout_seconds: int
    query_window_seconds: int
    expected_http_status: tuple[int, ...]
    expected_authorization: str
    minimum_results: int
    empty_result_valid: bool
    rollout_state: RolloutState
    activation_release: str
    reason: str
    next_activation_step: str


@dataclass(frozen=True)
class PipelineHealthContract:
    """Marker defaults and reviewed pipeline-health records."""

    marker: dict[str, Any]
    defaults: dict[str, Any]
    records: tuple[PipelineHealthRecord, ...]


@dataclass(frozen=True)
class ReleaseContract:
    """The complete versioned release contract."""

    version: str
    release_stage: str
    product_versions: dict[str, str]
    records: tuple[ContractRecord, ...]
    pipeline_health: PipelineHealthContract = field(
        default_factory=lambda: PipelineHealthContract(marker={}, defaults={}, records=())
    )

    def record(self, identifier: str) -> ContractRecord:
        """Return a record by identifier."""
        for record in self.records:
            if record.identifier == identifier:
                return record
        raise KeyError(identifier)

    def pipeline_record(self, identifier: str) -> PipelineHealthRecord:
        """Return a pipeline-health record by identifier."""
        for record in self.pipeline_health.records:
            if record.identifier == identifier:
                return record
        raise KeyError(identifier)


def load_release_contract(source: str | Path | dict[str, Any]) -> ReleaseContract:
    """Load and validate a YAML contract from a path or mapping."""
    raw: Any
    if isinstance(source, (str, Path)):
        with Path(source).open(encoding="utf-8") as contract_file:
            try:
                raw = yaml.load(contract_file, Loader=_UniqueKeyLoader)
            except ContractValidationError:
                raise
            except yaml.YAMLError as error:
                raise ContractValidationError(f"release contract YAML is invalid: {error}") from error
    else:
        raw = source

    if not isinstance(raw, dict):
        raise ContractValidationError("release contract must be a mapping")
    _reject_unknown_keys(source=raw, allowed=CONTRACT_KEYS, context="contract")

    version = _required_string(source=raw, key="contract_version", context="contract")
    release_stage = _required_string(source=raw, key="release_stage", context="contract")
    if release_stage not in RELEASE_STAGES:
        raise ContractValidationError(f"release_stage must be one of {sorted(RELEASE_STAGES)}")

    product_versions = {
        name: _resolve_product_version(value=version)
        for name, version in _string_mapping(value=raw.get("product_versions"), key="product_versions").items()
    }
    default_time_range = _parameter_mapping(value=raw.get("default_time_range", {}), key="default_time_range")
    pipeline_health = _parse_pipeline_health(raw=raw.get("pipeline_health", {}), release_stage=release_stage)
    raw_records = raw.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        raise ContractValidationError("records must be a non-empty list")

    records: list[ContractRecord] = []
    identifiers: set[str] = set()
    dashboard_panels: set[tuple[str, str]] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ContractValidationError(f"records[{index}] must be a mapping")
        record = _parse_record(
            raw=raw_record,
            release_stage=release_stage,
            product_versions=product_versions,
            default_time_range=default_time_range,
            index=index,
        )
        if record.identifier in identifiers:
            raise ContractValidationError(f"duplicate record identifier: {record.identifier}")
        identifiers.add(record.identifier)
        dashboard_panel = (record.dashboard, record.panel)
        if dashboard_panel in dashboard_panels:
            raise ContractValidationError(f"duplicate dashboard/panel mapping: {record.dashboard}/{record.panel}")
        dashboard_panels.add(dashboard_panel)
        records.append(record)

    return ReleaseContract(
        version=version,
        release_stage=release_stage,
        product_versions=product_versions,
        records=tuple(records),
        pipeline_health=pipeline_health,
    )


def render_promql(promql: str, **variables: str) -> str:
    """Render contract variables while escaping PromQL string values."""
    escaped_variables = {name: _escape_promql_value(value=value) for name, value in variables.items()}
    try:
        rendered = Template(template=promql).substitute(**escaped_variables)
    except (KeyError, ValueError) as error:
        raise ContractValidationError(f"unresolved PromQL variable: {error}") from error
    if "${" in rendered:
        raise ContractValidationError(f"unresolved PromQL variable in query: {promql}")
    return rendered


def _parse_record(
    raw: dict[str, Any],
    release_stage: str,
    product_versions: dict[str, str],
    default_time_range: dict[str, str | int | float],
    index: int,
) -> ContractRecord:
    _reject_unknown_keys(source=raw, allowed=RECORD_KEYS, context=f"records[{index}]")
    identifier = _required_string(source=raw, key="id", context=f"records[{index}]")
    record_stage = raw.get("release_stage", release_stage)
    if not isinstance(record_stage, str) or record_stage not in RELEASE_STAGES:
        raise ContractValidationError(f"records[{index}].release_stage must be one of {sorted(RELEASE_STAGES)}")

    expected_status = raw.get("expected_http_status")
    if (
        not isinstance(expected_status, list)
        or not expected_status
        or not all(isinstance(status, int) and 100 <= status <= 599 for status in expected_status)
    ):
        raise ContractValidationError(f"records[{index}].expected_http_status must be a list of HTTP statuses")
    if len(set(expected_status)) != len(expected_status):
        raise ContractValidationError(f"records[{index}].expected_http_status must not contain duplicates")

    minimum_series = raw.get("minimum_series")
    if not isinstance(minimum_series, int) or minimum_series < 0:
        raise ContractValidationError(f"records[{index}].minimum_series must be a non-negative integer")

    required_labels = raw.get("required_labels")
    if not isinstance(required_labels, list) or not all(isinstance(label, str) and label for label in required_labels):
        raise ContractValidationError(f"records[{index}].required_labels must be a list of non-empty strings")
    if len(set(required_labels)) != len(required_labels):
        raise ContractValidationError(f"records[{index}].required_labels must not contain duplicates")

    capability = _required_string(source=raw, key="capability", context=f"records[{index}]")
    if capability not in CAPABILITY_STATUSES:
        raise ContractValidationError(f"records[{index}].capability is not a supported status")

    authorization_response = _required_string(
        source=raw,
        key="authorization_response",
        context=f"records[{index}]",
    )
    if authorization_response not in AUTHORIZATION_RESPONSES:
        raise ContractValidationError(f"records[{index}].authorization response is not reviewed")

    empty_result_valid = raw.get("empty_result_valid")
    if not isinstance(empty_result_valid, bool):
        raise ContractValidationError(f"records[{index}].empty_result_valid must be boolean")
    empty_ui_state = _required_string(source=raw, key="empty_ui_state", context=f"records[{index}]")
    if (
        capability == "shipped"
        and empty_result_valid
        and any(marker in empty_ui_state.casefold() for marker in ("not shipped", "unavailable"))
    ):
        raise ContractValidationError(
            f"records[{index}].empty_ui_state cannot describe an unavailable shipped capability"
        )

    return ContractRecord(
        identifier=identifier,
        release_stage=record_stage,
        product_versions=product_versions,
        dashboard=_required_string(source=raw, key="dashboard", context=f"records[{index}]"),
        panel=_required_string(source=raw, key="panel", context=f"records[{index}]"),
        datasource=_required_string(source=raw, key="datasource", context=f"records[{index}]"),
        route=_required_string(source=raw, key="route", context=f"records[{index}]"),
        promql=_required_string(source=raw, key="promql", context=f"records[{index}]"),
        time_range=_parameter_mapping(
            value=raw.get("time_range", default_time_range),
            key=f"records[{index}].time_range",
        ),
        expected_http_status=tuple(expected_status),
        expected_prometheus_status=_required_string(
            source=raw,
            key="expected_prometheus_status",
            context=f"records[{index}]",
        ),
        expected_result_type=_required_string(
            source=raw,
            key="expected_result_type",
            context=f"records[{index}]",
        ),
        minimum_series=minimum_series,
        required_labels=tuple(required_labels),
        empty_result_valid=empty_result_valid,
        empty_ui_state=empty_ui_state,
        capability=capability,
        authorization_response=authorization_response,
        warnings_allowed=_optional_boolean(source=raw, key="warnings_allowed", context=f"records[{index}]"),
    )


def _parse_pipeline_health(raw: Any, release_stage: str) -> PipelineHealthContract:
    """Parse optional signal contracts without changing the existing metrics schema."""
    if raw in ({}, None):
        return PipelineHealthContract(marker={}, defaults={}, records=())
    if not isinstance(raw, dict):
        raise ContractValidationError("pipeline_health must be a mapping")
    _reject_unknown_keys(source=raw, allowed=PIPELINE_HEALTH_KEYS, context="pipeline_health")
    marker = _mapping(value=raw.get("marker", {}), key="pipeline_health.marker")
    defaults = _mapping(value=raw.get("defaults", {}), key="pipeline_health.defaults")
    raw_records = raw.get("records", [])
    if not isinstance(raw_records, list):
        raise ContractValidationError("pipeline_health.records must be a list")

    records: list[PipelineHealthRecord] = []
    identifiers: set[str] = set()
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, dict):
            raise ContractValidationError(f"pipeline_health.records[{index}] must be a mapping")
        record = _parse_pipeline_record(raw=raw_record, release_stage=release_stage, defaults=defaults, index=index)
        if record.identifier in identifiers:
            raise ContractValidationError(f"duplicate pipeline-health record identifier: {record.identifier}")
        identifiers.add(record.identifier)
        records.append(record)
    return PipelineHealthContract(marker=marker, defaults=defaults, records=tuple(records))


def _parse_pipeline_record(
    raw: dict[str, Any],
    release_stage: str,
    defaults: dict[str, Any],
    index: int,
) -> PipelineHealthRecord:
    _reject_unknown_keys(source=raw, allowed=PIPELINE_RECORD_KEYS, context=f"pipeline_health.records[{index}]")
    context = f"pipeline_health.records[{index}]"
    record_stage = raw.get("release_stage", release_stage)
    if not isinstance(record_stage, str) or record_stage not in RELEASE_STAGES:
        raise ContractValidationError(f"{context}.release_stage must be one of {sorted(RELEASE_STAGES)}")
    signal = _required_string(source=raw, key="signal", context=context)
    if signal not in PIPELINE_SIGNALS:
        raise ContractValidationError(f"{context}.signal must be one of {sorted(PIPELINE_SIGNALS)}")
    try:
        stage = PipelineStage(value=_required_string(source=raw, key="stage", context=context))
    except ValueError as error:
        raise ContractValidationError(f"{context}.stage is not a supported pipeline stage") from error

    rollout_value = _required_string(source=raw, key="rollout_state", context=context)
    try:
        rollout_state = RolloutState(value=rollout_value)
    except ValueError as error:
        raise ContractValidationError(f"{context}.rollout_state is not a supported rollout state") from error

    required_labels = _string_list(source=raw, key="required_labels", context=context)
    required_attributes = _string_list(source=raw, key="required_attributes", context=context)
    required_edges = _string_list(source=raw, key="required_edges", context=context)
    expected_status = raw.get("expected_http_status")
    if (
        not isinstance(expected_status, list)
        or not expected_status
        or not all(isinstance(status, int) and 100 <= status <= 599 for status in expected_status)
    ):
        raise ContractValidationError(f"{context}.expected_http_status must be a list of HTTP statuses")
    if len(set(expected_status)) != len(expected_status):
        raise ContractValidationError(f"{context}.expected_http_status must not contain duplicates")

    minimum_results = raw.get("minimum_results")
    if not isinstance(minimum_results, int) or minimum_results < 0:
        raise ContractValidationError(f"{context}.minimum_results must be a non-negative integer")
    empty_result_valid = raw.get("empty_result_valid")
    if not isinstance(empty_result_valid, bool):
        raise ContractValidationError(f"{context}.empty_result_valid must be boolean")

    propagation_timeout = _positive_int(
        value=raw.get("propagation_timeout_seconds", defaults.get("propagation_timeout_seconds", 240)),
        key=f"{context}.propagation_timeout_seconds",
    )
    query_window = _positive_int(
        value=raw.get("query_window_seconds", defaults.get("query_window_seconds", 240)),
        key=f"{context}.query_window_seconds",
    )
    return PipelineHealthRecord(
        identifier=_required_string(source=raw, key="id", context=context),
        release_stage=record_stage,
        signal=signal,
        stage=stage,
        marker=_required_string(source=raw, key="marker", context=context),
        source_query=_mapping(value=raw.get("source_query"), key=f"{context}.source_query"),
        backend_query=_mapping(value=raw.get("backend_query"), key=f"{context}.backend_query"),
        required_labels=required_labels,
        required_attributes=required_attributes,
        required_edges=required_edges,
        propagation_timeout_seconds=propagation_timeout,
        query_window_seconds=query_window,
        expected_http_status=tuple(expected_status),
        expected_authorization=_required_string(source=raw, key="expected_authorization", context=context),
        minimum_results=minimum_results,
        empty_result_valid=empty_result_valid,
        rollout_state=rollout_state,
        activation_release=_required_string(source=raw, key="activation_release", context=context),
        reason=_required_string(source=raw, key="reason", context=context),
        next_activation_step=_required_string(source=raw, key="next_activation_step", context=context),
    )


def _required_string(source: dict[str, Any], key: str, context: str) -> str:
    value = source.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ContractValidationError(f"{context}.{key} must be a non-empty string")
    return value


def _string_mapping(value: Any, key: str) -> dict[str, str]:
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(item, str) for name, item in value.items()
    ):
        raise ContractValidationError(f"{key} must be a string mapping")
    return dict(value)


def _parameter_mapping(value: Any, key: str) -> dict[str, str | int | float]:
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(item, (str, int, float)) for name, item in value.items()
    ):
        raise ContractValidationError(f"{key} must be a scalar parameter mapping")
    return dict(value)


def _mapping(value: Any, key: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        raise ContractValidationError(f"{key} must be a mapping")
    return dict(value)


def _string_list(source: dict[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = source.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ContractValidationError(f"{context}.{key} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ContractValidationError(f"{context}.{key} must not contain duplicates")
    return tuple(value)


def _positive_int(value: Any, key: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractValidationError(f"{key} must be a positive integer")
    return value


def _optional_boolean(source: dict[str, Any], key: str, context: str) -> bool:
    value = source.get(key, False)
    if not isinstance(value, bool):
        raise ContractValidationError(f"{context}.{key} must be boolean")
    return value


def _escape_promql_value(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractValidationError("PromQL variables must be non-empty strings")
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _resolve_product_version(value: str) -> str:
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], value)
    return value


def _reject_unknown_keys(source: dict[str, Any], allowed: set[str], context: str) -> None:
    unknown = set(source) - allowed
    if unknown:
        raise ContractValidationError(f"{context} contains unsupported fields: {', '.join(sorted(unknown))}")


class _UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys instead of silently overwriting them."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: MappingNode, deep: bool = False) -> dict[Any, Any]:
    if not isinstance(node, MappingNode):
        raise ContractValidationError("release contract root must be a YAML mapping")
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ContractValidationError("release contract mapping keys must be strings")
        if key in mapping:
            raise ContractValidationError(f"duplicate release contract key: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(
    tag=yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    constructor=_construct_unique_mapping,
)


def ensure_authorization_reviewed(record: ContractRecord) -> None:
    """Refuse a negative authorization assertion until its response is reviewed."""
    if record.authorization_response == "review-required":
        raise ContractValidationError(
            f"{record.identifier}: authorization response must be reviewed before a negative assertion runs"
        )
