import base64
import json
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Any, TypedDict

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.pod import Pod
from ocp_resources.replica_set import ReplicaSet
from ocp_resources.resource import NamespacedResource
from ocp_resources.secret import Secret
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.maas_api_key.utils import get_api_key
from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
    build_aitenant_test_context,
    wait_for_maas_resource_phase,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import (
    TENANT_ISOLATION_MODEL_NAME,
    assert_tenant_cross_tenant_inference_rejected,
    assert_tenant_inference_status,
    isolation_bootstrap_gateway_context,
    label_namespace_gateway_access,
    maas_api_base_url_for_gateway,
    maas_api_deployment_name_for_aitenant,
    make_tenant_model_accessible,
    provision_tenant_model,
    tenant_gateway_external_route,
    tenant_gateway_inference_url,
    tenant_isolation_auth_policy_name,
    tenant_isolation_subscription_name,
    verify_maas_api_deployment_for_aitenant,
    verify_maas_api_httproute_attached_to_gateway,
    verify_tenant_gateway_auth_policy_callback_url,
    wait_for_tenant_gateway_maas_api_reachable,
    wait_for_tenant_inference_response,
)
from tests.ai_gateway.models_as_a_service.utils import (
    MaaSTenantResource,
    aitenant_from_spec,
    assert_api_key_created_ok,
    bootstrap_gateway_ref,
    build_aitenant_spec,
    build_maas_headers,
    create_api_key,
    deploy_and_verify_aitenant_ready,
    get_httproute,
    get_maas_models_response,
    verify_maas_tenant_config_ready,
    wait_for_httproute,
)
from utilities.constants import ApiGroups
from utilities.general import generate_random_name
from utilities.logger import RedactedString
from utilities.resources.aitenant import AITenant
from utilities.resources.destination_rule import DestinationRule
from utilities.resources.external_model import ExternalModel
from utilities.resources.http_route import HTTPRoute
from utilities.resources.maastenantconfig import MaasTenantConfig
from utilities.resources.service_entry import ServiceEntry

LOGGER = structlog.get_logger(name=__name__)

MAAS_UPGRADE_BASELINE_CM_NAME = "maas-upgrade-test-baseline"
MAAS_UPGRADE_BASELINE_CM_KEY = "maas_baseline"
DEFAULT_AITENANT_NAME = "models-as-a-service"

LEGACY_MIGRATION_NAMESPACE = "upgrade-maas-legacy-em"
LEGACY_MIGRATION_MODEL_NAME = "upgrade-maas-legacy-em"
LEGACY_MIGRATION_SECRET_NAME = f"{LEGACY_MIGRATION_MODEL_NAME}-api-key"
LEGACY_MIGRATION_AUTH_POLICY_NAME = "upgrade-maas-legacy-em-auth"
LEGACY_MIGRATION_SUBSCRIPTION_NAME = "upgrade-maas-legacy-em-sub"
LEGACY_MIGRATION_ENDPOINT = "httpbin.org"
LEGACY_MIGRATION_TARGET_MODEL = "gpt-3.5-turbo"
MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME = "maas-legacy-migration-baseline"
MAAS_LEGACY_MIGRATION_BASELINE_CM_KEY = "legacy_migration_baseline"
LEGACY_EXTERNAL_MODEL_CRD_NAME = f"externalmodels.{ApiGroups.MAAS_IO}"
INFERENCE_EXTERNAL_MODEL_CRD_NAME = f"externalmodels.{ApiGroups.INFERENCE_OPENDATAHUB_IO}"
LEGACY_EXTERNAL_MODEL_API_VERSION = f"{ApiGroups.MAAS_IO}/v1alpha1"
LEGACY_EXTERNAL_MODEL_KIND = "ExternalModel"

MAAS_MULTITENANT_UPGRADE_NAMESPACE = "upgrade-maas-multitenant"
MAAS_MULTITENANT_UPGRADE_BASELINE_CM_NAME = "maas-multitenant-upgrade-baseline"
MAAS_MULTITENANT_UPGRADE_BASELINE_CM_KEY = "multitenant_upgrade_baseline"
MAAS_MT_UPGRADE_API_KEY_SECRET_KEY = "api-key"  # pragma: allowlist secret
MAAS_MT_UPGRADE_API_KEY_ID_SECRET_KEY = "api-key-id"
MULTITENANT_UPGRADE_TENANT_A_TOKENS_PER_MINUTE = 30
MULTITENANT_UPGRADE_TENANT_B_TOKENS_PER_MINUTE = 10000
MULTITENANT_UPGRADE_RATE_LIMIT_BURST_ATTEMPTS = 12
MULTITENANT_UPGRADE_RATE_LIMIT_BURST_MAX_TOKENS = 80
MAAS_API_DB_MIGRATION_FAILURE_LOG_SNIPPETS = (
    "database schema is in dirty state",
    "schema migration failed",
    "migration failed",
)
MAAS_API_DB_MIGRATION_SUCCESS_LOG_SNIPPET = "database schema applied"
MULTITENANT_UPGRADE_RATE_LIMIT_READY_TIMEOUT_SECONDS = 300


class MultitenantUpgradeTenantRecord(TypedDict):
    """Pre-upgrade snapshot for one AITenant in the multitenant upgrade scenario (RHOAIENG-100389)."""

    aitenant_name: str
    tenant_namespace_name: str
    subscription_name: str
    auth_policy_name: str
    subscription_generation: int
    tokens_per_minute: int
    gateway_name: str
    gateway_namespace: str
    api_key_id: str
    api_key_secret_name: str


class MultitenantUpgradeBaseline(TypedDict):
    """Persisted multitenant upgrade state for post-upgrade validation."""

    tenants: list[MultitenantUpgradeTenantRecord]


class MaaSBaseline(TypedDict):
    gateway_name: str
    gateway_namespace: str
    model_ref_name: str
    model_ref_namespace: str
    auth_policy_name: str
    auth_policy_namespace: str
    subscription_name: str
    subscription_namespace: str
    subscription_generation: int
    tenant_name: str
    tenant_namespace: str
    tenant_phase: str


class LegacyMigrationBaseline(TypedDict):
    model_name: str
    model_namespace: str
    legacy_resource_name: str
    auth_policy_name: str
    auth_policy_namespace: str
    subscription_name: str
    subscription_namespace: str


def _config_map_string_data(config_map: ConfigMap) -> dict[str, str]:
    """Return ConfigMap string data as a plain dict."""
    return dict(config_map.instance.data or {})


def _tenant_status_phase(tenant: NamespacedResource) -> str:
    """Read tenant status.phase when present on MaasTenantConfig or legacy Tenant."""
    tenant_status = tenant.instance.status
    if not hasattr(tenant_status, "phase"):
        return ""
    tenant_phase = tenant_status.phase
    return tenant_phase or ""


def _field_value(value: object, field_name: str, default: object = None) -> object:
    """Read a field from a Kubernetes wrapper object or a plain mapping."""
    if isinstance(value, Mapping):
        return value.get(field_name, default)
    return getattr(value, field_name, default)


def capture_maas_baseline(
    gateway: Gateway,
    model_ref: MaaSModelRef,
    auth_policy: MaaSAuthPolicy,
    subscription: MaaSSubscription,
    tenant: MaaSTenantResource,
) -> MaaSBaseline:
    """Snapshot MaaS control plane state before upgrade."""
    baseline: MaaSBaseline = {
        "gateway_name": gateway.name,
        "gateway_namespace": gateway.namespace,
        "model_ref_name": model_ref.name,
        "model_ref_namespace": model_ref.namespace,
        "auth_policy_name": auth_policy.name,
        "auth_policy_namespace": auth_policy.namespace,
        "subscription_name": subscription.name,
        "subscription_namespace": subscription.namespace,
        "subscription_generation": subscription.instance.metadata.generation or 0,
        "tenant_name": tenant.name,
        "tenant_namespace": tenant.namespace,
        "tenant_phase": _tenant_status_phase(tenant=tenant),
    }
    LOGGER.info(f"Captured MaaS upgrade baseline: {baseline}")
    return baseline


def save_maas_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baseline: MaaSBaseline,
) -> ConfigMap:
    """Persist the MaaS baseline snapshot to a ConfigMap for post-upgrade retrieval."""
    serialized_data = {MAAS_UPGRADE_BASELINE_CM_KEY: json.dumps(baseline)}
    config_map = ConfigMap(client=client, name=MAAS_UPGRADE_BASELINE_CM_NAME, namespace=namespace)
    if config_map.exists:
        resource_dict = config_map.instance.to_dict()
        resource_dict.setdefault("data", {}).update(serialized_data)
        config_map.update(resource_dict=resource_dict)
    else:
        config_map = ConfigMap(
            client=client,
            name=MAAS_UPGRADE_BASELINE_CM_NAME,
            namespace=namespace,
            data=serialized_data,
        )
        config_map.deploy()
    LOGGER.info(f"Saved MaaS baseline to ConfigMap {namespace}/{MAAS_UPGRADE_BASELINE_CM_NAME}")
    return config_map


def load_maas_baseline_from_configmap(
    client: DynamicClient,
    namespace: str,
) -> MaaSBaseline:
    """Load the MaaS baseline snapshot from the ConfigMap created during pre-upgrade."""
    config_map = ConfigMap(client=client, name=MAAS_UPGRADE_BASELINE_CM_NAME, namespace=namespace)
    assert config_map.exists, (
        f"MaaS baseline ConfigMap '{MAAS_UPGRADE_BASELINE_CM_NAME}' not found in '{namespace}'. "
        "Ensure pre-upgrade tests ran successfully."
    )
    config_map_data = _config_map_string_data(config_map=config_map)
    assert MAAS_UPGRADE_BASELINE_CM_KEY in config_map_data, (
        f"MaaS baseline ConfigMap '{MAAS_UPGRADE_BASELINE_CM_NAME}' is missing "
        f"the '{MAAS_UPGRADE_BASELINE_CM_KEY}' key."
    )
    raw_baseline = config_map_data[MAAS_UPGRADE_BASELINE_CM_KEY]
    return json.loads(raw_baseline)


def load_maas_api_key_from_secret(
    client: DynamicClient,
    namespace: str,
    secret_name: str,
    secret_key: str,
) -> RedactedString:
    """Load a base64-encoded MaaS API key as a redacted fixture-safe string."""
    secret = Secret(client=client, name=secret_name, namespace=namespace)
    assert secret.exists, (
        f"MaaS API-key Secret '{secret_name}' not found in namespace '{namespace}'. "
        "Ensure pre-upgrade tests created the Secret."
    )
    secret_data = _field_value(value=secret.instance, field_name="data", default={})
    encoded_value = _field_value(value=secret_data, field_name=secret_key)
    assert encoded_value, f"MaaS API-key Secret '{secret_name}' in namespace '{namespace}' has no '{secret_key}' key."
    try:
        decoded_value = base64.b64decode(str(encoded_value), validate=True).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as error:
        raise AssertionError(
            f"MaaS API-key Secret '{secret_name}' key '{secret_key}' is not valid base64 UTF-8 data"
        ) from error
    return RedactedString(value=decoded_value)


def verify_maas_model_ref_exists(model_ref: MaaSModelRef) -> None:
    """Assert that the MaaSModelRef exists after upgrade."""
    assert model_ref.exists, (
        f"MaaSModelRef '{model_ref.name}' not found in namespace '{model_ref.namespace}' after upgrade."
    )


def verify_maas_auth_policy_exists(auth_policy: MaaSAuthPolicy) -> None:
    """Assert that the MaaSAuthPolicy exists after upgrade."""
    assert auth_policy.exists, (
        f"MaaSAuthPolicy '{auth_policy.name}' not found in namespace '{auth_policy.namespace}' after upgrade."
    )


def verify_maas_subscription_ready(subscription: MaaSSubscription) -> None:
    """Assert that the MaaSSubscription exists after upgrade.

    The subscription may not reach Ready=True without a backing LLMInferenceService,
    which is out of scope for upgrade tests. The goal is to verify CR survival across upgrade.
    """
    assert subscription.exists, (
        f"MaaSSubscription '{subscription.name}' not found in namespace '{subscription.namespace}' after upgrade."
    )


def verify_maas_subscription_not_mutated(
    subscription: MaaSSubscription,
    baseline: MaaSBaseline,
) -> None:
    """Assert that MaaSSubscription generation matches the pre-upgrade baseline."""
    current_generation = subscription.instance.metadata.generation or 0
    expected_generation = baseline["subscription_generation"]
    assert current_generation == expected_generation, (
        f"MaaSSubscription '{subscription.name}' was mutated during upgrade: "
        f"expected generation {expected_generation}, got {current_generation}."
    )


def cluster_has_legacy_external_model_crd(admin_client: DynamicClient) -> bool:
    """Return True when the legacy maas.opendatahub.io ExternalModel CRD is installed."""
    legacy_external_model_crd = CustomResourceDefinition(client=admin_client, name=LEGACY_EXTERNAL_MODEL_CRD_NAME)
    return bool(legacy_external_model_crd.exists)


def cluster_has_inference_external_model_crd(admin_client: DynamicClient) -> bool:
    """Return True when the inference.opendatahub.io ExternalModel CRD is installed."""
    inference_external_model_crd = CustomResourceDefinition(client=admin_client, name=INFERENCE_EXTERNAL_MODEL_CRD_NAME)
    return bool(inference_external_model_crd.exists)


def legacy_external_model_resource_name_candidates(model_name: str) -> list[str]:
    """Return possible legacy ExternalModel networking resource names.

    RHOAI 3.4 maas-controller names children after the model directly. 3.5+ uses
    ``maas-{model_name}`` via modelnaming.ExternalModelResourceName.
    """
    prefixed_name = f"maas-{model_name}"
    if prefixed_name == model_name:
        return [model_name]
    return [prefixed_name, model_name]


def legacy_external_model_resource_name(model_name: str) -> str:
    """Return the preferred legacy ExternalModel networking resource name (3.5+)."""
    return f"maas-{model_name}"


def resolve_legacy_maas_networking_resource_name(
    client: DynamicClient,
    model_name: str,
    namespace: str,
) -> str | None:
    """Return the legacy networking resource name present in the namespace, if any."""
    for candidate in legacy_external_model_resource_name_candidates(model_name=model_name):
        if legacy_maas_networking_present(client=client, resource_name=candidate, namespace=namespace):
            return candidate
    return None


def get_legacy_maas_service(
    client: DynamicClient,
    resource_name: str,
    namespace: str,
) -> Service | None:
    """Look up the legacy maas-* Service, returning None when it is absent."""
    try:
        service = Service(client=client, name=resource_name, namespace=namespace)
        if service.exists:
            return service
    except NotFoundError, ResourceNotFoundError:
        LOGGER.debug(f"Service {namespace}/{resource_name} not found")
    return None


def get_legacy_maas_service_entry(
    client: DynamicClient,
    resource_name: str,
    namespace: str,
) -> ServiceEntry | None:
    """Look up the legacy maas-* ServiceEntry, returning None when it is absent."""
    try:
        service_entry = ServiceEntry(client=client, name=resource_name, namespace=namespace)
        if service_entry.exists:
            return service_entry
    except NotFoundError, ResourceNotFoundError:
        LOGGER.debug(f"ServiceEntry {namespace}/{resource_name} not found")
    return None


def get_legacy_maas_destination_rule(
    client: DynamicClient,
    resource_name: str,
    namespace: str,
) -> DestinationRule | None:
    """Look up the legacy maas-* DestinationRule, returning None when it is absent."""
    try:
        destination_rule = DestinationRule(client=client, name=resource_name, namespace=namespace)
        if destination_rule.exists:
            return destination_rule
    except NotFoundError, ResourceNotFoundError:
        LOGGER.debug(f"DestinationRule {namespace}/{resource_name} not found")
    return None


def legacy_maas_networking_present(
    client: DynamicClient,
    resource_name: str,
    namespace: str,
) -> bool:
    """Return True when all legacy maas-* networking children exist."""
    destination_rule = get_legacy_maas_destination_rule(
        client=client,
        resource_name=resource_name,
        namespace=namespace,
    )
    return (
        get_httproute(client=client, name=resource_name, namespace=namespace) is not None
        and get_legacy_maas_service(client=client, resource_name=resource_name, namespace=namespace) is not None
        and get_legacy_maas_service_entry(client=client, resource_name=resource_name, namespace=namespace) is not None
        and destination_rule is not None
    )


def legacy_maas_networking_absent(
    client: DynamicClient,
    resource_name: str,
    namespace: str,
) -> bool:
    """Return True when all legacy maas-* networking children are absent."""
    return (
        get_httproute(client=client, name=resource_name, namespace=namespace) is None
        and get_legacy_maas_service(client=client, resource_name=resource_name, namespace=namespace) is None
        and get_legacy_maas_service_entry(client=client, resource_name=resource_name, namespace=namespace) is None
        and get_legacy_maas_destination_rule(client=client, resource_name=resource_name, namespace=namespace) is None
    )


def wait_for_legacy_maas_networking_present(
    client: DynamicClient,
    model_name: str,
    namespace: str,
    timeout: int = 300,
) -> str:
    """Poll until legacy networking children exist and return the resolved resource name."""
    candidates = legacy_external_model_resource_name_candidates(model_name=model_name)

    def networking_present() -> str | None:
        for candidate in candidates:
            if legacy_maas_networking_present(client=client, resource_name=candidate, namespace=namespace):
                return candidate
        return None

    for _sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=networking_present,
    ):
        if resolved_name := networking_present():
            return resolved_name

    raise TimeoutError(
        f"Legacy networking for model '{model_name}' not fully present in '{namespace}' "
        f"(checked {candidates}) within {timeout}s"
    )


def wait_for_legacy_maas_prefixed_networking_deleted(
    client: DynamicClient,
    resource_name: str,
    namespace: str,
    timeout: int = 300,
) -> None:
    """Poll until maas-prefixed legacy networking is removed (PR #1417 teardown target).

    Args:
        client: Kubernetes dynamic client.
        resource_name: Legacy networking resource name (typically ``maas-{model}``).
        namespace: Namespace containing the legacy networking children.
        timeout: Maximum wait time in seconds.

    Raises:
        TimeoutError: If legacy networking children for ``resource_name`` are still present.
    """
    for _sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=legacy_maas_networking_absent,
        client=client,
        resource_name=resource_name,
        namespace=namespace,
    ):
        if legacy_maas_networking_absent(client=client, resource_name=resource_name, namespace=namespace):
            return

    raise TimeoutError(f"Legacy networking '{namespace}/{resource_name}' still present after {timeout}s")


def owner_ref_is_legacy_external_model(owner_ref: Any, model_name: str) -> bool:
    """Return True when an ownerReference points at the legacy maas ExternalModel."""
    if isinstance(owner_ref, dict):
        api_version = owner_ref.get("apiVersion", "")
        kind = owner_ref.get("kind", "")
        name = owner_ref.get("name", "")
    else:
        api_version = owner_ref.apiVersion
        kind = owner_ref.kind
        name = owner_ref.name
    return (
        api_version == LEGACY_EXTERNAL_MODEL_API_VERSION and kind == LEGACY_EXTERNAL_MODEL_KIND and name == model_name
    )


def verify_no_legacy_owned_httproutes(
    client: DynamicClient,
    namespace: str,
    model_name: str,
) -> None:
    """Assert no HTTPRoute is owned by the legacy maas.opendatahub.io ExternalModel.

    Inference may reuse the model name for its HTTPRoute after supersede; ownership distinguishes
    legacy reconciler routes from inference routes.

    Args:
        client: Kubernetes dynamic client.
        namespace: Namespace to inspect.
        model_name: Legacy ExternalModel name used as the owner reference target.

    Raises:
        AssertionError: When any HTTPRoute is still owned by the legacy ExternalModel.
    """
    legacy_owned_routes: list[str] = []
    for http_route in HTTPRoute.get(client=client, namespace=namespace):
        owner_references = http_route.instance.metadata.ownerReferences or []
        if any(
            owner_ref_is_legacy_external_model(owner_ref=owner_ref, model_name=model_name)
            for owner_ref in owner_references
        ):
            legacy_owned_routes.append(http_route.name)

    assert not legacy_owned_routes, (
        f"Legacy-owned HTTPRoutes still present in '{namespace}' after supersede: {legacy_owned_routes}"
    )


def inference_external_model_for_baseline(
    client: DynamicClient,
    baseline: LegacyMigrationBaseline,
) -> ExternalModel:
    """Return the inference ExternalModel CR referenced by a legacy migration baseline."""
    return ExternalModel(
        client=client,
        name=baseline["model_name"],
        namespace=baseline["model_namespace"],
    )


def verify_inference_external_model_exists(external_model: ExternalModel) -> None:
    """Assert that the inference ExternalModel exists after upgrade migration."""
    assert external_model.exists, (
        f"Inference ExternalModel '{external_model.name}' not found in namespace "
        f"'{external_model.namespace}' after upgrade migration."
    )


def get_inference_http_route_name(external_model: ExternalModel) -> str:
    """Read status.httpRouteName from an inference ExternalModel when programmed."""
    external_model_status = external_model.instance.status
    if external_model_status is None:
        return ""
    if hasattr(external_model_status, "httpRouteName"):
        return external_model_status.httpRouteName or ""
    return ""


def wait_for_inference_external_model_programmed(
    external_model: ExternalModel,
    timeout: int = 300,
) -> str:
    """Poll until the inference ExternalModel reports a programmed HTTPRoute name."""
    for _sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=5,
        func=get_inference_http_route_name,
        external_model=external_model,
    ):
        http_route_name = get_inference_http_route_name(external_model=external_model)
        if http_route_name:
            return http_route_name

    raise TimeoutError(
        f"Inference ExternalModel '{external_model.namespace}/{external_model.name}' "
        f"did not report status.httpRouteName within {timeout}s"
    )


def verify_inference_external_model_programmed(external_model: ExternalModel) -> str:
    """Assert the inference ExternalModel exists and reports a programmed HTTPRoute name."""
    verify_inference_external_model_exists(external_model=external_model)
    return wait_for_inference_external_model_programmed(external_model=external_model)


def wait_for_inference_external_model_httproute(
    client: DynamicClient,
    external_model: ExternalModel,
    namespace: str,
    timeout: int = 300,
) -> str:
    """Poll until the inference ExternalModel is programmed and its HTTPRoute exists."""
    http_route_name = wait_for_inference_external_model_programmed(
        external_model=external_model,
        timeout=timeout,
    )
    wait_for_httproute(
        client=client,
        name=http_route_name,
        namespace=namespace,
        timeout=timeout,
    )
    return http_route_name


def capture_legacy_migration_baseline(
    client: DynamicClient,
    model_name: str,
    model_namespace: str,
    auth_policy: MaaSAuthPolicy,
    subscription: MaaSSubscription,
) -> LegacyMigrationBaseline:
    """Snapshot legacy external model migration state before upgrade."""
    legacy_resource_name = resolve_legacy_maas_networking_resource_name(
        client=client,
        model_name=model_name,
        namespace=model_namespace,
    ) or legacy_external_model_resource_name(model_name=model_name)
    baseline: LegacyMigrationBaseline = {
        "model_name": model_name,
        "model_namespace": model_namespace,
        "legacy_resource_name": legacy_resource_name,
        "auth_policy_name": auth_policy.name,
        "auth_policy_namespace": auth_policy.namespace,
        "subscription_name": subscription.name,
        "subscription_namespace": subscription.namespace,
    }
    LOGGER.info(f"Captured legacy external model migration baseline: {baseline}")
    return baseline


def save_legacy_migration_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baseline: LegacyMigrationBaseline,
) -> ConfigMap:
    """Persist the legacy migration baseline snapshot to a ConfigMap."""
    serialized_data = {MAAS_LEGACY_MIGRATION_BASELINE_CM_KEY: json.dumps(baseline)}
    config_map = ConfigMap(client=client, name=MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME, namespace=namespace)
    if config_map.exists:
        resource_dict = config_map.instance.to_dict()
        resource_dict.setdefault("data", {}).update(serialized_data)
        config_map.update(resource_dict=resource_dict)
    else:
        config_map = ConfigMap(
            client=client,
            name=MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME,
            namespace=namespace,
            data=serialized_data,
        )
        config_map.deploy()
    LOGGER.info(f"Saved legacy migration baseline to ConfigMap {namespace}/{MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME}")
    return config_map


def load_legacy_migration_baseline_from_configmap(
    client: DynamicClient,
    namespace: str,
) -> LegacyMigrationBaseline:
    """Load the legacy migration baseline snapshot created during pre-upgrade."""
    config_map = ConfigMap(
        client=client,
        name=MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME,
        namespace=namespace,
    )
    assert config_map.exists, (
        f"Legacy migration baseline ConfigMap '{MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME}' not found in '{namespace}'. "
        "Ensure pre-upgrade tests ran successfully."
    )
    config_map_data = _config_map_string_data(config_map=config_map)
    assert MAAS_LEGACY_MIGRATION_BASELINE_CM_KEY in config_map_data, (
        f"Legacy migration baseline ConfigMap '{MAAS_LEGACY_MIGRATION_BASELINE_CM_NAME}' is missing "
        f"the '{MAAS_LEGACY_MIGRATION_BASELINE_CM_KEY}' key."
    )
    raw_baseline = config_map_data[MAAS_LEGACY_MIGRATION_BASELINE_CM_KEY]
    return json.loads(raw_baseline)


def multitenant_upgrade_tenant_aitenant_names() -> tuple[str, str]:
    """Return stable AITenant names for the two-tenant upgrade stack."""
    suffix = generate_random_name()
    return (
        f"e2e-mt-upg-a-{suffix}",
        f"e2e-mt-upg-b-{suffix}",
    )


def capture_multitenant_upgrade_baseline(
    tenant_records: list[MultitenantUpgradeTenantRecord],
) -> MultitenantUpgradeBaseline:
    """Build a baseline snapshot from provisioned tenant records."""
    baseline: MultitenantUpgradeBaseline = {"tenants": tenant_records}
    LOGGER.info(f"Captured multitenant MaaS upgrade baseline with {len(tenant_records)} tenant(s)")
    return baseline


def save_multitenant_upgrade_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baseline: MultitenantUpgradeBaseline,
) -> ConfigMap:
    """Persist multitenant upgrade baseline to a ConfigMap."""
    serialized_data = {MAAS_MULTITENANT_UPGRADE_BASELINE_CM_KEY: json.dumps(baseline)}
    config_map = ConfigMap(
        client=client,
        name=MAAS_MULTITENANT_UPGRADE_BASELINE_CM_NAME,
        namespace=namespace,
    )
    if config_map.exists:
        resource_dict = config_map.instance.to_dict()
        resource_dict.setdefault("data", {}).update(serialized_data)
        config_map.update(resource_dict=resource_dict)
    else:
        config_map = ConfigMap(
            client=client,
            name=MAAS_MULTITENANT_UPGRADE_BASELINE_CM_NAME,
            namespace=namespace,
            data=serialized_data,
        )
        config_map.deploy()
    LOGGER.info(
        f"Saved multitenant upgrade baseline to ConfigMap {namespace}/{MAAS_MULTITENANT_UPGRADE_BASELINE_CM_NAME}"
    )
    return config_map


def load_multitenant_upgrade_baseline_from_configmap(
    client: DynamicClient,
    namespace: str,
) -> MultitenantUpgradeBaseline:
    """Load multitenant upgrade baseline written during pre-upgrade."""
    config_map = ConfigMap(
        client=client,
        name=MAAS_MULTITENANT_UPGRADE_BASELINE_CM_NAME,
        namespace=namespace,
    )
    assert config_map.exists, (
        f"Multitenant upgrade baseline ConfigMap '{MAAS_MULTITENANT_UPGRADE_BASELINE_CM_NAME}' "
        f"not found in '{namespace}'. Ensure pre-upgrade tests ran successfully."
    )
    config_map_data = _config_map_string_data(config_map=config_map)
    assert MAAS_MULTITENANT_UPGRADE_BASELINE_CM_KEY in config_map_data, (
        f"Multitenant upgrade baseline ConfigMap missing key '{MAAS_MULTITENANT_UPGRADE_BASELINE_CM_KEY}'."
    )
    return json.loads(config_map_data[MAAS_MULTITENANT_UPGRADE_BASELINE_CM_KEY])


def load_persisted_multitenant_upgrade_api_key_from_secret(
    client: DynamicClient,
    namespace: str,
    secret_name: str,
) -> RedactedString:
    """Load a persisted multitenant upgrade API key plaintext from a Secret."""
    return load_maas_api_key_from_secret(
        client=client,
        namespace=namespace,
        secret_name=secret_name,
        secret_key=MAAS_MT_UPGRADE_API_KEY_SECRET_KEY,
    )


def load_multitenant_upgrade_api_key_id_from_secret(
    client: DynamicClient,
    namespace: str,
    secret_name: str,
) -> str:
    """Load the MaaS API key id persisted alongside the plaintext in a multitenant upgrade Secret."""
    secret = Secret(client=client, name=secret_name, namespace=namespace)
    assert secret.exists, f"Multitenant upgrade API key Secret '{secret_name}' not found in namespace '{namespace}'."
    secret_data = _field_value(value=secret.instance, field_name="data", default={})
    encoded_key_id = _field_value(value=secret_data, field_name=MAAS_MT_UPGRADE_API_KEY_ID_SECRET_KEY)
    assert encoded_key_id, (
        f"Multitenant upgrade API key Secret '{namespace}/{secret_name}' is missing "
        f"'{MAAS_MT_UPGRADE_API_KEY_ID_SECRET_KEY}'. Delete the Secret and re-run pre-upgrade provisioning."
    )
    try:
        return base64.b64decode(str(encoded_key_id), validate=True).decode("utf-8")
    except (UnicodeDecodeError, ValueError) as error:
        raise AssertionError(
            f"Multitenant upgrade API key Secret '{secret_name}' key '{MAAS_MT_UPGRADE_API_KEY_ID_SECRET_KEY}' "
            "is not valid base64 UTF-8 data"
        ) from error


def assert_multitenant_upgrade_api_key_secrets_absent(
    client: DynamicClient,
    namespace: str,
    secret_names: tuple[str, str],
) -> None:
    """Fail fast when fixed-name API key Secrets exist without a persisted baseline ConfigMap."""
    existing_secret_names = [
        secret_name
        for secret_name in secret_names
        if Secret(client=client, name=secret_name, namespace=namespace).exists
    ]
    assert not existing_secret_names, (
        f"Multitenant upgrade API key Secret(s) {existing_secret_names} already exist in '{namespace}' "
        f"without baseline ConfigMap '{MAAS_MULTITENANT_UPGRADE_BASELINE_CM_NAME}'. "
        "Delete those Secrets before re-provisioning, or restore the baseline ConfigMap to re-run pre-upgrade."
    )


def _persist_multitenant_upgrade_api_key_secret(
    admin_client: DynamicClient,
    persist_namespace: str,
    secret_name: str,
    plaintext_key: str,
    key_id: str,
    teardown: bool,
) -> None:
    """Create or update the multitenant upgrade API key Secret (idempotent for fixed secret names)."""
    string_data = {
        MAAS_MT_UPGRADE_API_KEY_SECRET_KEY: plaintext_key,
        MAAS_MT_UPGRADE_API_KEY_ID_SECRET_KEY: key_id,
    }
    secret = Secret(
        client=admin_client,
        name=secret_name,
        namespace=persist_namespace,
        type="Opaque",
        teardown=teardown,
        wait_for_resource=True,
    )
    if secret.exists:
        resource_dict = secret.instance.to_dict()
        resource_dict.setdefault("stringData", {}).update(string_data)
        secret.update(resource_dict=resource_dict)
    else:
        secret_with_data = Secret(
            client=admin_client,
            name=secret_name,
            namespace=persist_namespace,
            type="Opaque",
            string_data=string_data,
            teardown=teardown,
            wait_for_resource=True,
        )
        secret_with_data.deploy()


@contextmanager
def provision_multitenant_upgrade_aitenant(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    aitenant_name: str,
    maas_api_infra_namespace: str,
    maas_host: str,
    maas_free_group: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
    request_session_http: requests.Session,
    teardown: bool,
    tokens_per_minute: int,
) -> Generator[MultitenantUpgradeTenantRecord, Any, Any]:
    """Provision one Ready AITenant with model, subscription, and reachable per-tenant maas-api."""
    aitenant_spec = build_aitenant_spec(aitenant_name=aitenant_name)
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=aitenant_name,
        aitenant_spec=aitenant_spec,
    )
    auth_policy_name = tenant_isolation_auth_policy_name(aitenant_name=aitenant_name)
    subscription_name = tenant_isolation_subscription_name(aitenant_name=aitenant_name)

    with (
        isolation_bootstrap_gateway_context(
            admin_client=admin_client,
            gateway_name=gateway_name,
            api_namespace=maas_api_infra_namespace,
            gateway_namespace=gateway_namespace,
            teardown=teardown,
        ),
        tenant_gateway_external_route(
            admin_client=admin_client,
            gateway_name=gateway_name,
            maas_host=maas_host,
            teardown=teardown,
        ),
        aitenant_from_spec(
            admin_client=admin_client,
            aitenant_name=aitenant_name,
            cr_namespace=aitenant_infra_namespace,
            aitenant_spec=aitenant_spec,
            teardown=teardown,
        ) as aitenant,
    ):
        deploy_and_verify_aitenant_ready(aitenant=aitenant)
        test_context = build_aitenant_test_context(aitenant=aitenant)
        tenant_namespace_name = test_context["tenant_namespace_name"]
        with (
            provision_tenant_model(
                admin_client=admin_client,
                model_name=TENANT_ISOLATION_MODEL_NAME,
                tenant_namespace_name=tenant_namespace_name,
                gateway_name=gateway_name,
                gateway_namespace=gateway_namespace,
                aws_access_key=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_s3_region=models_s3_bucket_region,
                aws_s3_bucket=models_s3_bucket_name,
                aws_s3_endpoint=models_s3_bucket_endpoint,
                teardown=teardown,
            ),
            make_tenant_model_accessible(
                admin_client=admin_client,
                model_name=TENANT_ISOLATION_MODEL_NAME,
                model_namespace=tenant_namespace_name,
                tenant_namespace_name=tenant_namespace_name,
                auth_policy_name=auth_policy_name,
                subscription_name=subscription_name,
                owner_group_name="system:authenticated",
                free_group_name=maas_free_group,
                tokens_per_minute=tokens_per_minute,
                teardown=teardown,
            ) as subscription,
        ):
            subscription_generation = subscription.instance.metadata.generation or 0
            label_namespace_gateway_access(
                admin_client=admin_client,
                namespace_name=tenant_namespace_name,
                gateway_name=gateway_name,
            )
            verify_maas_api_deployment_for_aitenant(
                admin_client=admin_client,
                api_namespace=maas_api_infra_namespace,
                aitenant_name=aitenant_name,
                tenant_namespace_name=tenant_namespace_name,
            )
            verify_maas_api_httproute_attached_to_gateway(
                admin_client=admin_client,
                api_namespace=maas_api_infra_namespace,
                aitenant_name=aitenant_name,
                tenant_namespace_name=tenant_namespace_name,
                gateway_name=gateway_name,
                gateway_namespace=gateway_namespace,
            )
            verify_tenant_gateway_auth_policy_callback_url(
                admin_client=admin_client,
                gateway_name=gateway_name,
                gateway_namespace=gateway_namespace,
                aitenant_name=aitenant_name,
                api_namespace=maas_api_infra_namespace,
            )
            wait_for_tenant_gateway_maas_api_reachable(
                request_session_http=request_session_http,
                gateway_name=gateway_name,
                maas_host=maas_host,
            )
            yield MultitenantUpgradeTenantRecord(
                aitenant_name=aitenant_name,
                tenant_namespace_name=tenant_namespace_name,
                subscription_name=subscription_name,
                auth_policy_name=auth_policy_name,
                subscription_generation=subscription_generation,
                tokens_per_minute=tokens_per_minute,
                gateway_name=gateway_name,
                gateway_namespace=gateway_namespace,
                api_key_id="",
                api_key_secret_name="",
            )


def create_and_persist_tenant_api_key(
    admin_client: DynamicClient,
    persist_namespace: str,
    base_url: str,
    ocp_user_token: str,
    request_session_http: requests.Session,
    subscription_name: str,
    secret_name: str,
    teardown: bool,
) -> MultitenantUpgradeTenantRecord:
    """Create a tenant API key, persist plaintext to a Secret, and return key metadata."""
    existing_secret = Secret(client=admin_client, name=secret_name, namespace=persist_namespace)
    if existing_secret.exists:
        key_id = load_multitenant_upgrade_api_key_id_from_secret(
            client=admin_client,
            namespace=persist_namespace,
            secret_name=secret_name,
        )
        load_persisted_multitenant_upgrade_api_key_from_secret(
            client=admin_client,
            namespace=persist_namespace,
            secret_name=secret_name,
        )
        LOGGER.info(f"Reusing multitenant upgrade API key id={key_id} from Secret {persist_namespace}/{secret_name}")
        return MultitenantUpgradeTenantRecord(
            aitenant_name="",
            tenant_namespace_name="",
            subscription_name=subscription_name,
            auth_policy_name="",
            subscription_generation=0,
            tokens_per_minute=0,
            gateway_name="",
            gateway_namespace="",
            api_key_id=key_id,
            api_key_secret_name=secret_name,
        )

    api_key_name = f"e2e-mt-upg-key-{generate_random_name()}"
    create_response, create_body = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_user_token,
        request_session_http=request_session_http,
        api_key_name=api_key_name,
        subscription=subscription_name,
    )
    assert_api_key_created_ok(
        resp=create_response,
        body=create_body,
        required_fields=("id", "key"),  # pragma: allowlist secret
    )
    key_id = str(create_body["id"])
    plaintext_key = str(create_body["key"])
    _persist_multitenant_upgrade_api_key_secret(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        secret_name=secret_name,
        plaintext_key=plaintext_key,
        key_id=key_id,
        teardown=teardown,
    )
    LOGGER.info(f"Persisted multitenant upgrade API key id={key_id} to Secret {persist_namespace}/{secret_name}")
    return MultitenantUpgradeTenantRecord(
        aitenant_name="",
        tenant_namespace_name="",
        subscription_name=subscription_name,
        auth_policy_name="",
        subscription_generation=0,
        tokens_per_minute=0,
        gateway_name="",
        gateway_namespace="",
        api_key_id=key_id,
        api_key_secret_name=secret_name,
    )


def tenant_base_url_for_record(
    maas_host: str,
    tenant_record: MultitenantUpgradeTenantRecord,
) -> str:
    """Return maas-api base URL for a tenant record."""
    return maas_api_base_url_for_gateway(
        gateway_name=tenant_record["gateway_name"],
        maas_host=maas_host,
        scheme="https",
    )


def tenant_inference_url_for_record(
    maas_host: str,
    tenant_record: MultitenantUpgradeTenantRecord,
) -> str:
    """Return external inference URL for the tenant-local model."""
    return tenant_gateway_inference_url(
        gateway_name=tenant_record["gateway_name"],
        maas_host=maas_host,
        tenant_namespace_name=tenant_record["tenant_namespace_name"],
        model_name=TENANT_ISOLATION_MODEL_NAME,
    )


def verify_api_key_absent_on_foreign_tenant_gateway(
    request_session_http: requests.Session,
    foreign_base_url: str,
    ocp_user_token: str,
    foreign_key_id: str,
) -> None:
    """Assert an API key from another tenant is not visible on this tenant's maas-api."""
    get_response, _ = get_api_key(
        request_session_http=request_session_http,
        base_url=foreign_base_url,
        key_id=foreign_key_id,
        ocp_user_token=ocp_user_token,
    )
    assert get_response.status_code == 404, (
        f"Expected 404 when fetching foreign key id={foreign_key_id} via {foreign_base_url}, "
        f"got {get_response.status_code}: {get_response.text[:200]}"
    )


def aitenant_for_multitenant_upgrade_record(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    tenant_record: MultitenantUpgradeTenantRecord,
) -> AITenant:
    """Return an AITenant handle for a multitenant upgrade baseline tenant record."""
    return AITenant(
        client=admin_client,
        name=tenant_record["aitenant_name"],
        namespace=aitenant_infra_namespace,
        ensure_exists=True,
    )


def multitenant_upgrade_inference_chat_payload() -> dict[str, Any]:
    """Return a small chat payload for tenant-local isolation model inference."""
    return {
        "model": TENANT_ISOLATION_MODEL_NAME,
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 16,
    }


def multitenant_upgrade_rate_limit_burst_payload() -> dict[str, Any]:
    """Return a chat payload sized to consume subscription token quota quickly."""
    return {
        "model": TENANT_ISOLATION_MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": "Repeat the word token many times separated by spaces. No other text.",
            },
        ],
        "max_tokens": MULTITENANT_UPGRADE_RATE_LIMIT_BURST_MAX_TOKENS,
    }


def multitenant_upgrade_api_key_headers(
    admin_client: DynamicClient,
    persist_namespace: str,
    tenant_record: MultitenantUpgradeTenantRecord,
) -> dict[str, str]:
    """Build Authorization headers from the persisted pre-upgrade API key Secret."""
    api_key = load_persisted_multitenant_upgrade_api_key_from_secret(
        client=admin_client,
        namespace=persist_namespace,
        secret_name=tenant_record["api_key_secret_name"],
    )
    return build_maas_headers(token=str(api_key))


def assert_multitenant_upgrade_own_inference_succeeds(
    admin_client: DynamicClient,
    persist_namespace: str,
    maas_host: str,
    request_session_http: requests.Session,
    tenant_record: MultitenantUpgradeTenantRecord,
    chat_payload: dict[str, Any],
) -> None:
    """Assert inference on the tenant's own gateway returns 200."""
    headers = multitenant_upgrade_api_key_headers(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        tenant_record=tenant_record,
    )
    inference_url = tenant_inference_url_for_record(maas_host=maas_host, tenant_record=tenant_record)
    assert_tenant_inference_status(
        session=request_session_http,
        inference_url=inference_url,
        headers=headers,
        payload=chat_payload,
        expected_status=200,
    )


def assert_multitenant_upgrade_cross_tenant_inference_rejected(
    admin_client: DynamicClient,
    persist_namespace: str,
    maas_host: str,
    request_session_http: requests.Session,
    owner_tenant_record: MultitenantUpgradeTenantRecord,
    foreign_tenant_record: MultitenantUpgradeTenantRecord,
    chat_payload: dict[str, Any],
) -> None:
    """Assert the owner's API key is rejected on another tenant's inference gateway."""
    headers = multitenant_upgrade_api_key_headers(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        tenant_record=owner_tenant_record,
    )
    foreign_inference_url = tenant_inference_url_for_record(
        maas_host=maas_host,
        tenant_record=foreign_tenant_record,
    )
    assert_tenant_cross_tenant_inference_rejected(
        session=request_session_http,
        inference_url=foreign_inference_url,
        headers=headers,
        payload=chat_payload,
    )


def assert_multitenant_upgrade_inference_isolation_for_pair(
    admin_client: DynamicClient,
    persist_namespace: str,
    maas_host: str,
    request_session_http: requests.Session,
    owner_tenant_record: MultitenantUpgradeTenantRecord,
    peer_tenant_record: MultitenantUpgradeTenantRecord,
    chat_payload: dict[str, Any],
) -> None:
    """Assert own-gateway success and cross-tenant rejection for one tenant's persisted key."""
    assert_multitenant_upgrade_own_inference_succeeds(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        maas_host=maas_host,
        request_session_http=request_session_http,
        tenant_record=owner_tenant_record,
        chat_payload=chat_payload,
    )
    assert_multitenant_upgrade_cross_tenant_inference_rejected(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        maas_host=maas_host,
        request_session_http=request_session_http,
        owner_tenant_record=owner_tenant_record,
        foreign_tenant_record=peer_tenant_record,
        chat_payload=chat_payload,
    )


def verify_multitenant_upgrade_cross_tenant_api_key_get_isolation(
    request_session_http: requests.Session,
    maas_host: str,
    ocp_user_token: str,
    tenant_records: list[MultitenantUpgradeTenantRecord],
) -> None:
    """Assert each tenant's API key id is not visible via the other tenant's maas-api."""
    assert len(tenant_records) == 2, f"Expected 2 tenants, got {len(tenant_records)}"
    first_record, second_record = tenant_records[0], tenant_records[1]
    verify_api_key_absent_on_foreign_tenant_gateway(
        request_session_http=request_session_http,
        foreign_base_url=tenant_base_url_for_record(maas_host=maas_host, tenant_record=second_record),
        ocp_user_token=ocp_user_token,
        foreign_key_id=first_record["api_key_id"],
    )
    verify_api_key_absent_on_foreign_tenant_gateway(
        request_session_http=request_session_http,
        foreign_base_url=tenant_base_url_for_record(maas_host=maas_host, tenant_record=first_record),
        ocp_user_token=ocp_user_token,
        foreign_key_id=second_record["api_key_id"],
    )


def verify_multitenant_upgrade_models_list_probe(
    admin_client: DynamicClient,
    persist_namespace: str,
    maas_host: str,
    request_session_http: requests.Session,
    tenant_record: MultitenantUpgradeTenantRecord,
) -> None:
    """Assert GET /v1/models succeeds with the tenant's persisted API key."""
    headers = multitenant_upgrade_api_key_headers(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        tenant_record=tenant_record,
    )
    base_url = tenant_base_url_for_record(maas_host=maas_host, tenant_record=tenant_record)
    get_maas_models_response(
        session=request_session_http,
        base_url=base_url,
        headers=headers,
    )


def verify_multitenant_upgrade_tenant_governance_survived(
    admin_client: DynamicClient,
    tenant_record: MultitenantUpgradeTenantRecord,
) -> None:
    """Assert tenant MaaSModelRef, MaaSAuthPolicy, and MaaSSubscription survived upgrade unchanged."""
    tenant_namespace = tenant_record["tenant_namespace_name"]
    model_ref = MaaSModelRef(
        client=admin_client,
        name=TENANT_ISOLATION_MODEL_NAME,
        namespace=tenant_namespace,
    )
    verify_maas_model_ref_exists(model_ref=model_ref)
    auth_policy = MaaSAuthPolicy(
        client=admin_client,
        name=tenant_record["auth_policy_name"],
        namespace=tenant_namespace,
    )
    verify_maas_auth_policy_exists(auth_policy=auth_policy)
    subscription = MaaSSubscription(
        client=admin_client,
        name=tenant_record["subscription_name"],
        namespace=tenant_namespace,
    )
    verify_maas_subscription_ready(subscription=subscription)
    current_generation = subscription.instance.metadata.generation or 0
    expected_generation = tenant_record["subscription_generation"]
    assert current_generation == expected_generation, (
        f"MaaSSubscription '{subscription.name}' in '{tenant_namespace}' was mutated during upgrade: "
        f"expected generation {expected_generation}, got {current_generation}."
    )


def verify_multitenant_upgrade_maastenantconfig_active(
    admin_client: DynamicClient,
    tenant_record: MultitenantUpgradeTenantRecord,
) -> None:
    """Assert bootstrapped MaasTenantConfig in the tenant namespace is Ready and phase Active."""
    tenant_namespace = tenant_record["tenant_namespace_name"]
    maas_tenant_config = MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace,
        ensure_exists=True,
    )
    verify_maas_tenant_config_ready(maas_tenant_config=maas_tenant_config)
    wait_for_maas_resource_phase(
        resource=maas_tenant_config,
        expected_phases=("Active",),
    )


def _replica_set_names_for_deployment(
    admin_client: DynamicClient,
    namespace: str,
    deployment_name: str,
) -> set[str]:
    """Return ReplicaSet names owned by the named Deployment."""
    matching_names: set[str] = set()
    for replica_set in ReplicaSet.get(client=admin_client, namespace=namespace):
        owner_references = replica_set.instance.metadata.ownerReferences or []
        for owner_reference in owner_references:
            if owner_reference.kind == "Deployment" and owner_reference.name == deployment_name:
                matching_names.add(replica_set.name)
    return matching_names


def _running_pods_for_deployment(
    admin_client: DynamicClient,
    namespace: str,
    deployment_name: str,
) -> list[Pod]:
    """Return Running Pods owned by ReplicaSets of the named Deployment."""
    deployment = Deployment(
        client=admin_client,
        name=deployment_name,
        namespace=namespace,
        ensure_exists=True,
    )
    replica_set_names = _replica_set_names_for_deployment(
        admin_client=admin_client,
        namespace=namespace,
        deployment_name=deployment_name,
    )
    match_labels = deployment.instance.spec.selector.matchLabels
    label_selector = ",".join(f"{label_key}={label_value}" for label_key, label_value in match_labels.items())
    all_pods = list(
        Pod.get(
            client=admin_client,
            namespace=namespace,
            label_selector=label_selector,
        )
    )
    running_pods: list[Pod] = []
    for pod in all_pods:
        if pod.instance.status.phase != "Running":
            continue
        owner_references = pod.instance.metadata.ownerReferences or []
        for owner_reference in owner_references:
            if owner_reference.kind == "ReplicaSet" and owner_reference.name in replica_set_names:
                running_pods.append(pod)
                break
    return running_pods


def verify_multitenant_upgrade_maas_api_db_migration_logs_clean(
    admin_client: DynamicClient,
    maas_api_infra_namespace: str,
    tenant_records: list[MultitenantUpgradeTenantRecord],
) -> None:
    """Assert per-tenant maas-api Pods applied golang-migrate schema and report no dirty state."""
    for tenant_record in tenant_records:
        deployment_name = maas_api_deployment_name_for_aitenant(aitenant_name=tenant_record["aitenant_name"])
        running_pods = _running_pods_for_deployment(
            admin_client=admin_client,
            namespace=maas_api_infra_namespace,
            deployment_name=deployment_name,
        )
        assert running_pods, (
            f"Expected at least one Running Pod for Deployment '{maas_api_infra_namespace}/{deployment_name}' "
            f"after upgrade"
        )
        for pod in running_pods:
            log_text = pod.log().lower()
            for failure_snippet in MAAS_API_DB_MIGRATION_FAILURE_LOG_SNIPPETS:
                assert failure_snippet not in log_text, (
                    f"Pod '{pod.name}' log suggests maas-api DB migration failure "
                    f"({failure_snippet!r}): {log_text[-400:]}"
                )
            assert MAAS_API_DB_MIGRATION_SUCCESS_LOG_SNIPPET in log_text, (
                f"Pod '{pod.name}' log missing golang-migrate success marker "
                f"'{MAAS_API_DB_MIGRATION_SUCCESS_LOG_SNIPPET}': {log_text[-400:]}"
            )
            LOGGER.info(
                f"maas-api Pod '{pod.name}' for tenant '{tenant_record['aitenant_name']}' "
                f"reports {MAAS_API_DB_MIGRATION_SUCCESS_LOG_SNIPPET!r} with no dirty-state markers"
            )


def wait_for_multitenant_upgrade_subscription_rate_limits_live(
    admin_client: DynamicClient,
    persist_namespace: str,
    maas_host: str,
    request_session_http: requests.Session,
    rate_limited_tenant_record: MultitenantUpgradeTenantRecord,
    peer_tenant_record: MultitenantUpgradeTenantRecord,
) -> None:
    """Wait until peer inference succeeds and the low-quota tenant accepts inference (TRLP enforced, not deny-all)."""
    chat_payload = multitenant_upgrade_inference_chat_payload()
    assert_multitenant_upgrade_own_inference_succeeds(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        maas_host=maas_host,
        request_session_http=request_session_http,
        tenant_record=peer_tenant_record,
        chat_payload=chat_payload,
    )
    rate_limited_headers = multitenant_upgrade_api_key_headers(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        tenant_record=rate_limited_tenant_record,
    )
    rate_limited_url = tenant_inference_url_for_record(
        maas_host=maas_host,
        tenant_record=rate_limited_tenant_record,
    )
    try:
        wait_for_tenant_inference_response(
            session=request_session_http,
            inference_url=rate_limited_url,
            headers=rate_limited_headers,
            payload=chat_payload,
            expected_status=200,
            wait_timeout=MULTITENANT_UPGRADE_RATE_LIMIT_READY_TIMEOUT_SECONDS,
        )
    except TimeoutExpiredError as error:
        raise TimeoutExpiredError(
            f"Low-quota tenant '{rate_limited_tenant_record['aitenant_name']}' inference did not return 200 "
            f"within {MULTITENANT_UPGRADE_RATE_LIMIT_READY_TIMEOUT_SECONDS}s — "
            f"subscription TRLP may not be live yet"
        ) from error


def assert_multitenant_upgrade_rate_limit_isolation(
    admin_client: DynamicClient,
    persist_namespace: str,
    maas_host: str,
    request_session_http: requests.Session,
    rate_limited_tenant_record: MultitenantUpgradeTenantRecord,
    peer_tenant_record: MultitenantUpgradeTenantRecord,
) -> None:
    """Burst inference on the low-quota tenant until limited, then assert the peer tenant still succeeds."""
    wait_for_multitenant_upgrade_subscription_rate_limits_live(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        maas_host=maas_host,
        request_session_http=request_session_http,
        rate_limited_tenant_record=rate_limited_tenant_record,
        peer_tenant_record=peer_tenant_record,
    )
    rate_limited_headers = multitenant_upgrade_api_key_headers(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        tenant_record=rate_limited_tenant_record,
    )
    rate_limited_url = tenant_inference_url_for_record(
        maas_host=maas_host,
        tenant_record=rate_limited_tenant_record,
    )
    burst_payload = multitenant_upgrade_rate_limit_burst_payload()
    saw_rate_limited = False
    for attempt_index in range(MULTITENANT_UPGRADE_RATE_LIMIT_BURST_ATTEMPTS):
        response = request_session_http.post(
            url=rate_limited_url,
            headers=rate_limited_headers,
            json=burst_payload,
            timeout=60,
        )
        LOGGER.info(
            f"Rate-limit burst attempt {attempt_index + 1}/{MULTITENANT_UPGRADE_RATE_LIMIT_BURST_ATTEMPTS} "
            f"on {rate_limited_tenant_record['aitenant_name']}: status={response.status_code}"
        )
        if response.status_code == 429:
            saw_rate_limited = True
            break
        assert response.status_code == 200, (
            f"Unexpected status {response.status_code} while exhausting tenant "
            f"{rate_limited_tenant_record['aitenant_name']} quota: {response.text[:200]}"
        )
    assert saw_rate_limited, (
        f"Expected HTTP 429 after bursting tenant '{rate_limited_tenant_record['aitenant_name']}' "
        f"with tokens_per_minute={rate_limited_tenant_record['tokens_per_minute']} "
        f"within {MULTITENANT_UPGRADE_RATE_LIMIT_BURST_ATTEMPTS} attempts"
    )
    assert_multitenant_upgrade_own_inference_succeeds(
        admin_client=admin_client,
        persist_namespace=persist_namespace,
        maas_host=maas_host,
        request_session_http=request_session_http,
        tenant_record=peer_tenant_record,
        chat_payload=multitenant_upgrade_inference_chat_payload(),
    )
