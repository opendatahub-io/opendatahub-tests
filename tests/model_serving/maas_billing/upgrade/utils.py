import json
from typing import TypedDict

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.gateway_gateway_networking_k8s_io import Gateway
from ocp_resources.maas_auth_policy import MaaSAuthPolicy
from ocp_resources.maas_model_ref import MaaSModelRef
from ocp_resources.maas_subscription import MaaSSubscription

from utilities.resources.tenant import Tenant

LOGGER = structlog.get_logger(name=__name__)

MAAS_UPGRADE_BASELINE_CM_NAME = "maas-upgrade-test-baseline"
MAAS_UPGRADE_BASELINE_CM_KEY = "maas_baseline"


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


def capture_maas_baseline(
    *,
    gateway: Gateway,
    model_ref: MaaSModelRef,
    auth_policy: MaaSAuthPolicy,
    subscription: MaaSSubscription,
    tenant: Tenant,
) -> MaaSBaseline:
    """Snapshot MaaS control plane state before upgrade.

    Args:
        gateway: MaaS Gateway resource.
        model_ref: MaaSModelRef created for upgrade testing.
        auth_policy: MaaSAuthPolicy created for upgrade testing.
        subscription: MaaSSubscription created for upgrade testing.
        tenant: default-tenant Tenant CR bootstrapped by MaaS in 3.4.

    Returns:
        MaaSBaseline snapshot dict for post-upgrade comparison.
    """
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
        "tenant_phase": getattr(tenant.instance.status, "phase", "") or "",
    }
    LOGGER.info(f"Captured MaaS upgrade baseline: {baseline}")
    return baseline


def save_maas_baseline_to_configmap(
    *,
    client: DynamicClient,
    namespace: str,
    baseline: MaaSBaseline,
) -> ConfigMap:
    """Persist the MaaS baseline snapshot to a ConfigMap for post-upgrade retrieval.

    Args:
        client: DynamicClient instance.
        namespace: Namespace where the ConfigMap is created.
        baseline: Baseline snapshot to persist.

    Returns:
        The created or updated ConfigMap.
    """
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
    *,
    client: DynamicClient,
    namespace: str,
) -> MaaSBaseline:
    """Load the MaaS baseline snapshot from the ConfigMap created during pre-upgrade.

    Args:
        client: DynamicClient instance.
        namespace: Namespace where the ConfigMap was saved.

    Returns:
        MaaSBaseline snapshot dict.

    Raises:
        AssertionError: If the ConfigMap or baseline key is missing.
    """
    config_map = ConfigMap(client=client, name=MAAS_UPGRADE_BASELINE_CM_NAME, namespace=namespace)
    assert config_map.exists, (
        f"MaaS baseline ConfigMap '{MAAS_UPGRADE_BASELINE_CM_NAME}' not found in '{namespace}'. "
        "Ensure pre-upgrade tests ran successfully."
    )
    cm_data = config_map.instance.data or {}
    raw_baseline = cm_data.get(MAAS_UPGRADE_BASELINE_CM_KEY)
    assert raw_baseline, (
        f"MaaS baseline ConfigMap '{MAAS_UPGRADE_BASELINE_CM_NAME}' is missing "
        f"the '{MAAS_UPGRADE_BASELINE_CM_KEY}' key."
    )
    return json.loads(raw_baseline)


def verify_maas_gateway_programmed(gateway: Gateway) -> None:
    """Assert that the MaaS Gateway exists and has reached Programmed=True.

    Args:
        gateway: Gateway resource to check.

    Raises:
        AssertionError: If the gateway does not exist.
    """
    assert gateway.exists, f"MaaS Gateway '{gateway.name}' not found in namespace '{gateway.namespace}' after upgrade."
    gateway.wait_for_condition(condition="Programmed", status="True", timeout=300)


def verify_maas_model_ref_exists(model_ref: MaaSModelRef) -> None:
    """Assert that the MaaSModelRef exists after upgrade.

    Args:
        model_ref: MaaSModelRef resource to check.

    Raises:
        AssertionError: If the resource does not exist.
    """
    assert model_ref.exists, (
        f"MaaSModelRef '{model_ref.name}' not found in namespace '{model_ref.namespace}' after upgrade."
    )


def verify_maas_auth_policy_exists(auth_policy: MaaSAuthPolicy) -> None:
    """Assert that the MaaSAuthPolicy exists after upgrade.

    Args:
        auth_policy: MaaSAuthPolicy resource to check.

    Raises:
        AssertionError: If the resource does not exist.
    """
    assert auth_policy.exists, (
        f"MaaSAuthPolicy '{auth_policy.name}' not found in namespace '{auth_policy.namespace}' after upgrade."
    )


def verify_maas_subscription_ready(subscription: MaaSSubscription) -> None:
    """Assert that the MaaSSubscription exists after upgrade.

    The subscription may not reach Ready=True without a backing LLMInferenceService,
    which is out of scope for upgrade tests. The goal is to verify CR survival across upgrade.

    Args:
        subscription: MaaSSubscription resource to check.

    Raises:
        AssertionError: If the resource does not exist.
    """
    assert subscription.exists, (
        f"MaaSSubscription '{subscription.name}' not found in namespace '{subscription.namespace}' after upgrade."
    )


def verify_maas_subscription_not_mutated(
    subscription: MaaSSubscription,
    baseline: MaaSBaseline,
) -> None:
    """Assert that MaaSSubscription generation matches the pre-upgrade baseline.

    Args:
        subscription: MaaSSubscription resource to check.
        baseline: Pre-upgrade baseline snapshot.

    Raises:
        AssertionError: If the subscription generation increased (indicating unwanted mutation).
    """
    current_generation = subscription.instance.metadata.generation or 0
    expected_generation = baseline["subscription_generation"]
    assert current_generation == expected_generation, (
        f"MaaSSubscription '{subscription.name}' was mutated during upgrade: "
        f"expected generation {expected_generation}, got {current_generation}."
    )


def verify_maas_tenant_ready(tenant: Tenant) -> None:
    """Assert that the default-tenant Tenant CR exists and has Ready=True.

    The Tenant is the best leading indicator of controller health post-upgrade —
    if it regresses to a non-Ready state every other MaaS assertion will reflect
    the same root cause.

    Args:
        tenant: Tenant CR to check.

    Raises:
        AssertionError: If the Tenant does not exist.
    """
    assert tenant.exists, f"Tenant '{tenant.name}' not found in namespace '{tenant.namespace}' after upgrade."
    tenant.wait_for_condition(condition="Ready", status="True", timeout=300)
