import json
from typing import TypedDict

import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription
from ocp_resources.resource import NamespacedResource
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

from tests.model_serving.maas_billing.utils import MaaSTenantResource
from utilities.constants import ApiGroups
from utilities.resources.destination_rule import DestinationRule
from utilities.resources.http_route import HTTPRoute
from utilities.resources.service_entry import ServiceEntry

LOGGER = structlog.get_logger(name=__name__)

MAAS_UPGRADE_BASELINE_CM_NAME = "maas-upgrade-test-baseline"
MAAS_UPGRADE_BASELINE_CM_KEY = "maas_baseline"

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


def _tenant_status_phase(tenant: NamespacedResource) -> str:
    """Read tenant status.phase when present on the legacy Tenant CR."""
    tenant_status = tenant.instance.status
    if not hasattr(tenant_status, "phase"):
        return ""
    tenant_phase = tenant_status.phase
    return tenant_phase or ""


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


def cluster_has_legacy_external_model_crd(admin_client: DynamicClient) -> bool:
    """Return True when the legacy maas.opendatahub.io ExternalModel CRD is installed."""
    legacy_external_model_crd = CustomResourceDefinition(client=admin_client, name=LEGACY_EXTERNAL_MODEL_CRD_NAME)
    return bool(legacy_external_model_crd.exists)


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


def get_httproute(
    client: DynamicClient,
    name: str,
    namespace: str,
) -> HTTPRoute | None:
    """Look up an HTTPRoute by name/namespace. Returns the resource wrapper or None."""
    try:
        route = HTTPRoute(client=client, name=name, namespace=namespace)
        if route.exists:
            return route
    except NotFoundError, ResourceNotFoundError:
        LOGGER.debug(f"HTTPRoute {namespace}/{name} not found")
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

    for _ in TimeoutSampler(
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
