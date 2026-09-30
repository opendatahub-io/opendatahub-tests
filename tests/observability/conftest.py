"""Fixtures for the integrated observability release-contract suite."""

import json
import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest
from pytest_testconfig import config as py_config

from tests.observability.authorization import (
    SubjectAccessReviewResult,
    authenticate_persona_tokens,
    check_persona_access,
)
from tests.observability.contract import ContractRecord, ReleaseContract, load_release_contract
from tests.observability.evidence import EvidenceRecord, write_evidence, write_failure_log, write_preflight_evidence
from tests.observability.fixtures import (
    NamespacePair,
    create_model_pair,
    create_namespace_pair,
    resource_evidence,
    wait_for_source_metric,
)
from tests.observability.personas import (
    Persona,
    PersonaValidationError,
    select_persona_tokens,
    validate_personas,
    validate_token_identities,
)
from tests.observability.preflight import (
    PreflightCheck,
    PreflightReport,
    authorization_preflight_checks,
    evaluate_preflight,
    parse_preflight_present,
)
from tests.observability.query import RawQueryClient, RawQueryResult, build_contract_request
from utilities.path_utils import resolve_repo_path

if TYPE_CHECKING:
    from kubernetes.dynamic import DynamicClient

CONTRACT_PATH = Path(__file__).parent / "contracts" / "release_contract.yaml"
REQUIRED_PREFLIGHT_CHECKS = {
    "release-stage",
    "component-versions",
    "observability-feature",
    "dashboard-resources",
    "perses-crds",
    "datasource-routes",
    "gpu-operator",
    "gpu-capacity",
    "gpu-workload",
    "maas-stack",
    "persona-authentication",
    "subject-access-review",
}
SOURCE_RECORD_IDENTIFIERS = {
    "accelerator-dcgm-source",
    "inference-vllm-series",
    "maas-authorized-hits",
    "maas-authorized-calls",
}


@pytest.fixture(scope="session")
def release_contract() -> ReleaseContract:
    """Load the reviewed contract input selected for this release run."""
    configured_path = os.environ.get("RHOAI_OBSERVABILITY_CONTRACT")
    path = resolve_repo_path(configured_path) if configured_path else CONTRACT_PATH
    return load_release_contract(source=path)


@pytest.fixture(scope="session")
def release_evidence_directory() -> Path:
    """Return the CI evidence directory without placing credentials in its name or contents."""
    configured_path = os.environ.get("RHOAI_OBSERVABILITY_EVIDENCE_DIR")
    if configured_path:
        return Path(configured_path)
    return Path(str(py_config.get("tmp_base_dir", "/tmp"))) / "observability-evidence"


@pytest.fixture(scope="session")
def release_preflight(
    release_contract: ReleaseContract,
    release_evidence_directory: Path,
) -> PreflightReport:
    """Evaluate release-run preflight input before resource fixtures can mutate the cluster."""
    checks = _preflight_checks(release_contract=release_contract)
    report = evaluate_preflight(checks=checks)
    write_preflight_evidence(
        destination=release_evidence_directory / "preflight.json",
        report={
            "tracking_id": "RHOAIENG-96476",
            "contract_version": release_contract.version,
            "release_stage": release_contract.release_stage,
            "product_versions": release_contract.product_versions,
            **report.to_dict(),
        },
    )
    return report


@pytest.fixture(scope="session")
def observability_evidence(
    release_contract: ReleaseContract,
    release_evidence_directory: Path,
) -> Generator[Callable[..., None]]:
    """Collect sanitized query evidence and write the release handoff after all tests finish."""
    records: list[EvidenceRecord] = []
    cluster_run_id = os.environ.get("RHOAI_OBSERVABILITY_RUN_ID", "local")

    def add_record(
        *,
        test_identifier: str,
        contract_record: ContractRecord,
        persona: Persona,
        fixture_resources: dict[str, object],
        query: RawQueryResult,
        failure_category: str | None = None,
    ) -> None:
        records.append(
            EvidenceRecord(
                tracking_id="RHOAIENG-96476",
                test_identifier=test_identifier,
                release_stage=contract_record.release_stage,
                component_versions=contract_record.product_versions,
                cluster_run_id=cluster_run_id,
                persona={
                    "name": persona.name,
                    "principal": persona.principal,
                    "groups": list(persona.groups),
                    "namespaces": list(persona.namespaces),
                },
                fixture_resources=fixture_resources,
                query=query,
                failure_category=failure_category,
            )
        )

    yield add_record
    write_evidence(
        destination=release_evidence_directory / "observability-evidence.json",
        records=records,
        handoff={
            "tracking_id": "RHOAIENG-96476",
            "cluster_run_id": cluster_run_id,
            "contract_version": release_contract.version,
            "record_count": len(records),
        },
    )
    write_failure_log(
        destination=release_evidence_directory / "observability-failures.log",
        records=[record for record in records if record.failure_category],
    )


@pytest.fixture(scope="class")
def observability_namespaces(
    admin_client: DynamicClient,
    teardown_resources: bool,
    release_preflight: PreflightReport,
) -> Generator[NamespacePair]:
    """Create the two namespace-isolation fixtures only after a ready preflight."""
    if release_preflight.disposition.value != "ready":
        pytest.fail(f"RHOAIENG-96476 preflight is {release_preflight.disposition.value}: {release_preflight.to_dict()}")
    with create_namespace_pair(admin_client=admin_client, teardown=teardown_resources) as namespaces:
        yield namespaces


@pytest.fixture(scope="class")
def observability_models(
    admin_client: DynamicClient,
    observability_namespaces: NamespacePair,
    release_contract: ReleaseContract,
    teardown_resources: bool,
) -> Generator[list[Any]]:
    """Create model-a and model-b from release-runner-selected serving configuration."""
    required_values = {
        "runtime_template": os.environ.get("RHOAI_OBSERVABILITY_RUNTIME_TEMPLATE"),
        "model_format": os.environ.get("RHOAI_OBSERVABILITY_MODEL_FORMAT"),
        "storage_uri": os.environ.get("RHOAI_OBSERVABILITY_STORAGE_URI"),
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        pytest.fail(f"[blocked] missing model fixture configuration: {', '.join(missing)}")
    configured_resources = os.environ.get("RHOAI_OBSERVABILITY_MODEL_RESOURCES")
    resources: dict[str, Any] | None = None
    if configured_resources:
        try:
            raw_resources: Any = json.loads(configured_resources)
        except json.JSONDecodeError as error:
            pytest.fail(f"[failed] model resource JSON is invalid: {type(error).__name__}")
        if not isinstance(raw_resources, dict):
            pytest.fail("[failed] model resource input must be a mapping")
        resources = raw_resources
    gpu_required = any(
        record.identifier.startswith("accelerator-") and record.capability == "shipped"
        for record in release_contract.records
    )
    if gpu_required and resources is None:
        pytest.fail("[blocked] set RHOAI_OBSERVABILITY_MODEL_RESOURCES for the shipped GPU contract")
    with create_model_pair(
        admin_client=admin_client,
        namespaces=observability_namespaces,
        runtime_template=str(required_values["runtime_template"]),
        model_format=str(required_values["model_format"]),
        storage_uri=str(required_values["storage_uri"]),
        deployment_mode=os.environ.get("RHOAI_OBSERVABILITY_DEPLOYMENT_MODE"),
        enable_auth=os.environ.get("RHOAI_OBSERVABILITY_ENABLE_AUTH", "false").lower() == "true",
        external_route=os.environ.get("RHOAI_OBSERVABILITY_EXTERNAL_ROUTE", "false").lower() == "true",
        resources=resources,
        teardown=teardown_resources,
    ) as models:
        yield models


@pytest.fixture(scope="class")
def observability_fixture_resources(
    observability_namespaces: NamespacePair,
    observability_models: list[Any],
    observability_inference_traffic: int,
    observability_maas_traffic: dict[str, int],
    observability_source_metrics: dict[str, RawQueryResult],
) -> dict[str, object]:
    """Return stable resource identity metadata for machine-readable release evidence."""
    return {
        "namespaces": [
            resource_evidence(observability_namespaces.namespace_a).to_dict(),
            resource_evidence(observability_namespaces.namespace_b).to_dict(),
        ],
        "models": [resource_evidence(model).to_dict() for model in observability_models],
        "maas_traffic": observability_maas_traffic,
    }


@pytest.fixture(scope="class")
def observability_inference_traffic(observability_models: list[Any]) -> int:
    """Send fixed inference traffic through each fixture model before telemetry is queried."""
    configured = os.environ.get("RHOAI_OBSERVABILITY_INFERENCE_CONFIG")
    if not configured:
        pytest.fail("[blocked] set RHOAI_OBSERVABILITY_INFERENCE_CONFIG for deterministic model traffic")
    try:
        inference_config: Any = json.loads(configured)
    except json.JSONDecodeError as error:
        pytest.fail(f"[failed] inference configuration JSON is invalid: {type(error).__name__}")
    if not isinstance(inference_config, dict):
        pytest.fail("[failed] inference configuration input must be a mapping")

    from tests.model_serving.model_server.utils import verify_inference_response

    inference_input: Any = None
    configured_input = os.environ.get("RHOAI_OBSERVABILITY_INFERENCE_INPUT")
    if configured_input:
        try:
            inference_input = json.loads(configured_input)
        except json.JSONDecodeError as error:
            pytest.fail(f"[failed] inference input JSON is invalid: {type(error).__name__}")
    use_default_query = inference_input is None
    for model in observability_models:
        verify_inference_response(
            inference_service=model,
            inference_config=inference_config,
            inference_type=os.environ.get("RHOAI_OBSERVABILITY_INFERENCE_TYPE", "infer"),
            protocol=os.environ.get("RHOAI_OBSERVABILITY_INFERENCE_PROTOCOL", "rest"),
            model_name=model.name,
            inference_input=inference_input,
            use_default_query=use_default_query,
            token=os.environ.get("RHOAI_OBSERVABILITY_INFERENCE_TOKEN"),
            insecure=os.environ.get("RHOAI_OBSERVABILITY_INSECURE", "false").lower() == "true",
            inference_timeout=120,
        )
    return len(observability_models)


@pytest.fixture(scope="class")
def observability_maas_traffic(release_contract: ReleaseContract) -> dict[str, int]:
    """Generate fixed MaaS success and, when shipped, rate-limited requests without retaining tokens."""
    required_values = {
        "base_url": os.environ.get("RHOAI_OBSERVABILITY_MAAS_BASE_URL"),
        "token": os.environ.get("RHOAI_OBSERVABILITY_MAAS_TOKEN"),
        "model": os.environ.get("RHOAI_OBSERVABILITY_MAAS_MODEL"),
    }
    missing = [name for name, value in required_values.items() if not value]
    if missing:
        pytest.fail(f"[blocked] missing MaaS traffic configuration: {', '.join(missing)}")
    try:
        request_count = int(os.environ.get("RHOAI_OBSERVABILITY_MAAS_REQUEST_COUNT", "3"))
    except ValueError:
        pytest.fail("[failed] RHOAI_OBSERVABILITY_MAAS_REQUEST_COUNT must be an integer")
    if request_count < 1:
        pytest.fail("[failed] RHOAI_OBSERVABILITY_MAAS_REQUEST_COUNT must be positive")

    import requests

    from tests.ai_gateway.models_as_a_service.utils import build_maas_headers, verify_chat_completions

    session = requests.Session()
    session.verify = os.environ.get("RHOAI_OBSERVABILITY_MAAS_CA_BUNDLE") or True
    statuses: list[int] = []
    try:
        for _request_index in range(request_count):
            response = verify_chat_completions(
                request_session_http=session,
                model_url=f"{str(required_values['base_url']).rstrip('/')}/v1/chat/completions",
                headers=build_maas_headers(token=str(required_values["token"])),
                models_list=[{"id": str(required_values["model"])}],
                expected_status_codes=(200, 429),
                log_prefix="RHOAIENG-96476 observability traffic",
            )
            statuses.append(response.status_code)
    finally:
        session.close()

    from tests.observability.fixtures import summarize_traffic

    summary = summarize_traffic(statuses=statuses)
    limited_record = release_contract.record(identifier="maas-limited-calls")
    if limited_record.capability == "shipped" and summary.rate_limited_requests == 0:
        pytest.fail("[failed] shipped MaaS limited_calls contract produced no HTTP 429 responses")
    if summary.successful_requests == 0:
        pytest.fail("[failed] MaaS traffic produced no successful requests")
    return summary.to_dict()


@pytest.fixture(scope="class")
def observability_source_metrics(
    release_contract: ReleaseContract,
    observability_namespaces: NamespacePair,
    observability_models: list[Any],
    observability_inference_traffic: int,
    observability_maas_traffic: dict[str, int],
    observability_query_clients: dict[str, RawQueryClient],
    observability_personas: tuple[Persona, ...],
    observability_persona_tokens: dict[str, str],
) -> dict[str, RawQueryResult]:
    """Verify source telemetry for shipped workload records before dashboard assertions run."""
    del observability_models, observability_inference_traffic, observability_maas_traffic
    admin = next(persona for persona in observability_personas if persona.name == "cluster-admin")
    namespace = observability_namespaces.namespace_a.name or ""
    results: dict[str, RawQueryResult] = {}
    for record in release_contract.records:
        if record.identifier not in SOURCE_RECORD_IDENTIFIERS or record.capability != "shipped":
            continue
        client = observability_query_clients.get(record.datasource)
        if client is None:
            pytest.fail(f"[failed] no datasource route configured for source record {record.identifier}")
        request = build_contract_request(
            contract=record,
            persona=admin.name,
            principal=admin.principal,
            requested_namespace=namespace,
            fixture_namespace=namespace,
            variables={"namespace": namespace, "model": "model-a"},
        )
        try:
            results[record.identifier] = wait_for_source_metric(
                query=lambda request=request, client=client: client.query(
                    request=request,
                    bearer_token=observability_persona_tokens[admin.name],
                )
            )
        except AssertionError as error:
            pytest.fail(f"[failed] source telemetry unavailable for {record.identifier}: {error}")
    return results


@pytest.fixture(scope="session")
def observability_query_clients() -> dict[str, RawQueryClient]:
    """Build one raw client per dashboard datasource from release-runner route bases."""
    configured = os.environ.get("RHOAI_OBSERVABILITY_DATASOURCE_URLS")
    if not configured:
        pytest.fail("[blocked] set RHOAI_OBSERVABILITY_DATASOURCE_URLS to sanitized datasource base URLs")
    try:
        raw_urls: Any = json.loads(configured)
    except json.JSONDecodeError as error:
        pytest.fail(f"[failed] datasource URL JSON is invalid: {type(error).__name__}")
    if not isinstance(raw_urls, dict) or not all(
        isinstance(name, str) and isinstance(url, str) for name, url in raw_urls.items()
    ):
        pytest.fail("[failed] datasource URL input must be a string mapping")
    ca_bundle = os.environ.get("RHOAI_OBSERVABILITY_CA_BUNDLE")
    return {
        name: RawQueryClient(
            base_url=url,
            verify=ca_bundle or True,
        )
        for name, url in raw_urls.items()
    }


@pytest.fixture(scope="session")
def observability_personas() -> tuple[Persona, ...]:
    """Load independently authenticated persona identity and scope metadata from the release runner."""
    configured = os.environ.get("RHOAI_OBSERVABILITY_PERSONAS")
    if not configured:
        pytest.fail("[blocked] set RHOAI_OBSERVABILITY_PERSONAS with sanitized principal and scope metadata")
    try:
        raw_personas: Any = json.loads(configured)
    except json.JSONDecodeError as error:
        pytest.fail(f"[failed] persona JSON is invalid: {type(error).__name__}")
    if not isinstance(raw_personas, list):
        pytest.fail("[failed] persona input must be a list")
    personas = []
    for item in raw_personas:
        if not isinstance(item, dict):
            pytest.fail("[failed] persona entries must be mappings")
        groups = item.get("groups", [])
        namespaces = item.get("namespaces", [])
        if not isinstance(groups, list) or not isinstance(namespaces, list):
            pytest.fail("[failed] persona groups and namespaces must be lists")
        personas.append(
            Persona(
                name=str(item.get("name", "")),
                principal=str(item.get("principal", "")),
                groups=tuple(str(group) for group in groups),
                namespaces=tuple(str(namespace) for namespace in namespaces),
            )
        )
    try:
        return validate_personas(personas=personas)
    except ValueError as error:
        pytest.fail(f"[failed] persona validation failed: {error}")


@pytest.fixture(scope="session")
def observability_persona_tokens(
    admin_client: DynamicClient,
    observability_personas: tuple[Persona, ...],
) -> dict[str, str]:
    """Load and authenticate bearer tokens injected by the release runner without placing them in evidence."""
    configured = os.environ.get("RHOAI_OBSERVABILITY_PERSONA_TOKENS")
    if not configured:
        pytest.fail("[blocked] set RHOAI_OBSERVABILITY_PERSONA_TOKENS through the CI secret provider")
    try:
        raw_tokens: Any = json.loads(configured)
    except json.JSONDecodeError as error:
        pytest.fail(f"[failed] persona token mapping is invalid: {type(error).__name__}")
    if not isinstance(raw_tokens, dict):
        pytest.fail("[failed] persona token mapping must be a string mapping")
    tokens = select_persona_tokens(raw_tokens=raw_tokens, personas=observability_personas)
    missing = [persona.name for persona in observability_personas if not tokens.get(persona.name)]
    if missing:
        pytest.fail(f"[failed] missing independent authentication tokens for: {', '.join(missing)}")
    if len(set(tokens.values())) != len(tokens):
        pytest.fail("[failed] persona authentication tokens must be independent")
    try:
        identities = authenticate_persona_tokens(admin_client=admin_client, tokens=tokens)
        validate_token_identities(personas=observability_personas, identities=identities)
    except PersonaValidationError as error:
        pytest.fail(f"[failed] persona token identity validation failed: {error}")
    except ValueError as error:
        pytest.fail(f"[failed] persona token authentication failed: {error}")
    return tokens


@pytest.fixture(scope="class")
def observability_sar_baseline(
    admin_client: DynamicClient,
    observability_namespaces: NamespacePair,
    observability_personas: tuple[Persona, ...],
) -> tuple[SubjectAccessReviewResult, ...]:
    """Capture SAR outcomes for both fixture namespaces before route queries run."""
    resource = os.environ.get("RHOAI_OBSERVABILITY_SAR_RESOURCE")
    if not resource:
        pytest.fail("[blocked] set RHOAI_OBSERVABILITY_SAR_RESOURCE to the reviewed metrics resource")
    api_group = os.environ.get("RHOAI_OBSERVABILITY_SAR_API_GROUP", "")
    api_version = os.environ.get("RHOAI_OBSERVABILITY_SAR_API_VERSION", "v1")
    namespaces = observability_namespaces.names
    results = []
    for persona in observability_personas:
        for namespace in namespaces:
            results.append(
                check_persona_access(
                    admin_client=admin_client,
                    persona=persona,
                    verb="get",
                    resource=resource,
                    namespace=namespace,
                    api_group=api_group,
                    api_version=api_version,
                )
            )
    return tuple(results)


def _preflight_checks(release_contract: ReleaseContract) -> list[PreflightCheck]:
    authorization_checks = authorization_preflight_checks(
        records=[(record.identifier, record.authorization_response) for record in release_contract.records]
    )
    version_checks = [
        PreflightCheck(
            name=f"component-version:{name}",
            present=not version.startswith("${"),
            category="product",
            detail=f"set the release-runner value for {name}" if version.startswith("${") else "",
        )
        for name, version in release_contract.product_versions.items()
    ]
    configured = os.environ.get("RHOAI_OBSERVABILITY_PREFLIGHT")
    if not configured:
        return [
            PreflightCheck(
                name="release-run-preflight-inputs",
                present=False,
                category="environment",
                detail="Set RHOAI_OBSERVABILITY_PREFLIGHT to the release runner's sanitized prerequisite JSON",
            ),
            PreflightCheck(name="contract", present=bool(release_contract.records), category="product"),
            *version_checks,
            *authorization_checks,
        ]

    try:
        raw_checks: Any = json.loads(configured)
    except json.JSONDecodeError as error:
        return [
            PreflightCheck(
                name="release-run-preflight-inputs", present=False, category="product", detail=type(error).__name__
            ),
            *authorization_checks,
        ]
    if not isinstance(raw_checks, list):
        return [
            PreflightCheck(
                name="release-run-preflight-inputs",
                present=False,
                category="product",
                detail="preflight JSON must be a list",
            ),
            *authorization_checks,
        ]
    checks = []
    for item in raw_checks:
        if not isinstance(item, dict):
            checks.append(PreflightCheck(name="release-run-preflight-inputs", present=False, category="product"))
            continue
        name = str(item.get("name", "unnamed"))
        try:
            present = parse_preflight_present(value=item.get("present", False))
        except (TypeError, ValueError) as error:
            checks.append(
                PreflightCheck(
                    name=name,
                    present=False,
                    category="product",
                    detail=str(error),
                )
            )
            continue
        checks.append(
            PreflightCheck(
                name=name,
                present=present,
                category=str(item.get("category", "product")),
                detail=str(item.get("detail", "")),
            )
        )
    configured_names = {check.name for check in checks}
    checks.extend(
        PreflightCheck(
            name=name,
            present=False,
            category="environment",
            detail="required release preflight check was not supplied",
        )
        for name in REQUIRED_PREFLIGHT_CHECKS - configured_names
    )
    checks.extend(version_checks)
    checks.extend(authorization_checks)
    return checks
