"""Verification helpers for ai-gateway-controller component health tests."""

from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role import ClusterRole
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutExpiredError

from tests.ai_gateway.models_as_a_service.component_health.constants import (
    AI_GATEWAY_CONTROLLER_AITENANT_RESOURCE,
    AI_GATEWAY_CONTROLLER_AITENANT_WATCH_VERBS,
    AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_BINDING_NAME,
    AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_NAME,
    AI_GATEWAY_CONTROLLER_DEPLOYMENT_AVAILABLE_TIMEOUT,
    AI_GATEWAY_CONTROLLER_DEPLOYMENT_NAME,
    AI_GATEWAY_CONTROLLER_HEALTH_PROBE_PORT,
    AI_GATEWAY_CONTROLLER_IMAGE_CONFIGMAP_KEY,
    AI_GATEWAY_CONTROLLER_LIVENESS_PROBE_PATH,
    AI_GATEWAY_CONTROLLER_MANAGER_CONTAINER_NAME,
    AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME,
    AI_GATEWAY_CONTROLLER_READINESS_PROBE_PATH,
    AI_GATEWAY_CONTROLLER_SERVICE_ACCOUNT_NAME,
    AIGATEWAY_CR_CONDITION_TIMEOUT,
    AIGATEWAY_CR_NAME,
    MODELS_AS_A_SERVICE_READY_CONDITION,
    PRAXIS_EXTPROC_IMAGE_CONFIGMAP_KEY,
    RELATED_IMAGE_ODH_PRAXIS_EXTPROC_ENV_NAME,
)
from tests.ai_gateway.models_as_a_service.utils import dsc_uses_aigateway_maas_schema
from utilities.constants import ApiGroups
from utilities.general import wait_for_pods_running
from utilities.resources.aigateway import AIGateway

LOGGER = structlog.get_logger(name=__name__)


def get_ai_gateway_controller_deployment(admin_client: DynamicClient) -> Deployment:
    """Return the ai-gateway-controller Deployment, asserting it exists."""
    controller_deployment = Deployment(
        client=admin_client,
        name=AI_GATEWAY_CONTROLLER_DEPLOYMENT_NAME,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
    assert controller_deployment.exists, (
        f"Deployment '{controller_deployment.namespace}/{AI_GATEWAY_CONTROLLER_DEPLOYMENT_NAME}' not found - "
        "expected ai-gateway-operator to deploy ai-gateway-controller when modelsAsAService is Managed"
    )
    return controller_deployment


def verify_ai_gateway_controller_deployment_available(admin_client: DynamicClient) -> None:
    """Assert the ai-gateway-controller Deployment reaches Available=True."""
    controller_deployment = get_ai_gateway_controller_deployment(admin_client=admin_client)
    controller_deployment.wait_for_condition(
        condition="Available",
        status="True",
        timeout=AI_GATEWAY_CONTROLLER_DEPLOYMENT_AVAILABLE_TIMEOUT,
    )
    LOGGER.info(f"Deployment '{controller_deployment.namespace}/{controller_deployment.name}' is Available")


def verify_ai_gateway_controller_pods_running(admin_client: DynamicClient) -> None:
    """Assert ai-gateway-controller pods are Running and ready."""
    LOGGER.info(f"Checking pods in namespace '{py_config['applications_namespace']}' for ai-gateway-controller health")
    wait_for_pods_running(admin_client=admin_client, namespace_name=py_config["applications_namespace"])


def verify_ai_gateway_controller_rbac_exists(admin_client: DynamicClient) -> None:
    """Assert ai-gateway-controller ServiceAccount, cluster RBAC, binding wiring, and AITenant watch rules exist."""
    service_account = ServiceAccount(
        client=admin_client,
        name=AI_GATEWAY_CONTROLLER_SERVICE_ACCOUNT_NAME,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
    assert service_account.exists, (
        f"ServiceAccount '{service_account.namespace}/{AI_GATEWAY_CONTROLLER_SERVICE_ACCOUNT_NAME}' not found"
    )

    cluster_role = ClusterRole(
        client=admin_client,
        name=AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_NAME,
        ensure_exists=True,
    )
    assert cluster_role.exists, f"ClusterRole '{AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_NAME}' not found"

    cluster_role_rules = cluster_role.instance.rules or []
    assert any(
        ApiGroups.MAAS_IO in (rule.apiGroups or [])
        and AI_GATEWAY_CONTROLLER_AITENANT_RESOURCE in (rule.resources or [])
        and AI_GATEWAY_CONTROLLER_AITENANT_WATCH_VERBS.issubset(set(rule.verbs or []))
        for rule in cluster_role_rules
    ), (
        f"ClusterRole '{AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_NAME}' missing rule "
        f"{ApiGroups.MAAS_IO}/{AI_GATEWAY_CONTROLLER_AITENANT_RESOURCE} with verbs "
        f"{', '.join(sorted(AI_GATEWAY_CONTROLLER_AITENANT_WATCH_VERBS))}"
    )

    cluster_role_binding = ClusterRoleBinding(
        client=admin_client,
        name=AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_BINDING_NAME,
        ensure_exists=True,
    )
    assert cluster_role_binding.exists, (
        f"ClusterRoleBinding '{AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_BINDING_NAME}' not found"
    )

    role_ref = cluster_role_binding.instance.roleRef
    assert role_ref.name == AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_NAME, (
        f"ClusterRoleBinding '{AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_BINDING_NAME}' has roleRef.name "
        f"'{role_ref.name}', expected '{AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_NAME}'"
    )

    subjects = cluster_role_binding.instance.subjects or []
    assert any(
        subject.kind == "ServiceAccount"
        and subject.name == AI_GATEWAY_CONTROLLER_SERVICE_ACCOUNT_NAME
        and subject.namespace == service_account.namespace
        for subject in subjects
    ), (
        f"ClusterRoleBinding '{AI_GATEWAY_CONTROLLER_CLUSTER_ROLE_BINDING_NAME}' missing subject "
        f"ServiceAccount/{AI_GATEWAY_CONTROLLER_SERVICE_ACCOUNT_NAME} in '{service_account.namespace}'"
    )

    LOGGER.info("ai-gateway-controller ServiceAccount, cluster RBAC, and AITenant watch rules are present")


def verify_ai_gateway_controller_parameters_configmap(admin_client: DynamicClient) -> None:
    """Assert ai-gateway-controller-parameters ConfigMap exists with expected image keys."""
    parameters_configmap = ConfigMap(
        client=admin_client,
        name=AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )
    assert parameters_configmap.exists, (
        f"ConfigMap '{parameters_configmap.namespace}/{AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME}' not found"
    )

    configmap_data: dict[str, str] = dict(parameters_configmap.instance.data or {})

    missing_keys: list[str] = []
    for configmap_key in (AI_GATEWAY_CONTROLLER_IMAGE_CONFIGMAP_KEY, PRAXIS_EXTPROC_IMAGE_CONFIGMAP_KEY):
        if configmap_key not in configmap_data:
            missing_keys.append(configmap_key)
            continue
        if not configmap_data[configmap_key].strip():
            missing_keys.append(f"{configmap_key} (empty)")

    assert not missing_keys, (
        f"ConfigMap '{parameters_configmap.namespace}/{AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME}' "
        f"is missing or has empty keys: {', '.join(missing_keys)}"
    )
    LOGGER.info(
        f"ConfigMap '{parameters_configmap.namespace}/{AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME}' "
        "has controller and praxis-extproc image parameters"
    )


def verify_aigateway_models_as_a_service_ready(admin_client: DynamicClient) -> None:
    """Assert AIGateway/default-aigateway reports ModelsAsAServiceReady=True."""
    if not dsc_uses_aigateway_maas_schema(admin_client=admin_client):
        pytest.skip("AIGateway CR checks apply only when DSC uses aigateway MaaS schema (3.5+)")

    aigateway = AIGateway(
        client=admin_client,
        name=AIGATEWAY_CR_NAME,
        ensure_exists=True,
    )
    assert aigateway.exists, (
        f"AIGateway '{AIGATEWAY_CR_NAME}' not found - "
        "expected ODH operator to bootstrap the component CR when aigateway is Managed"
    )

    try:
        aigateway.wait_for_condition(
            condition=MODELS_AS_A_SERVICE_READY_CONDITION,
            status="True",
            timeout=AIGATEWAY_CR_CONDITION_TIMEOUT,
        )
    except TimeoutExpiredError:
        condition_status = "not found"
        status_conditions = aigateway.instance.status.conditions or []
        for status_condition in status_conditions:
            if status_condition.type == MODELS_AS_A_SERVICE_READY_CONDITION:
                condition_status = status_condition.status
                break
        pytest.fail(
            f"AIGateway '{AIGATEWAY_CR_NAME}' condition '{MODELS_AS_A_SERVICE_READY_CONDITION}' "
            f"is not True after {AIGATEWAY_CR_CONDITION_TIMEOUT}s (last status: {condition_status})"
        )

    LOGGER.info(f"AIGateway '{AIGATEWAY_CR_NAME}' condition '{MODELS_AS_A_SERVICE_READY_CONDITION}' is True")


def _find_manager_container(deployment: Deployment) -> Any:
    for container in deployment.instance.spec.template.spec.containers:
        if container.name == AI_GATEWAY_CONTROLLER_MANAGER_CONTAINER_NAME:
            return container
    raise AssertionError(
        f"Container '{AI_GATEWAY_CONTROLLER_MANAGER_CONTAINER_NAME}' not found in "
        f"Deployment '{deployment.namespace}/{deployment.name}'"
    )


def verify_ai_gateway_controller_health_probes_configured(admin_client: DynamicClient) -> None:
    """Assert the manager container exposes standard controller-runtime health probes."""
    controller_deployment = get_ai_gateway_controller_deployment(admin_client=admin_client)
    manager_container = _find_manager_container(deployment=controller_deployment)

    liveness_probe = manager_container.livenessProbe
    assert liveness_probe is not None, (
        f"Deployment '{controller_deployment.namespace}/{controller_deployment.name}' "
        f"container '{AI_GATEWAY_CONTROLLER_MANAGER_CONTAINER_NAME}' has no livenessProbe"
    )
    assert liveness_probe.httpGet.path == AI_GATEWAY_CONTROLLER_LIVENESS_PROBE_PATH, (
        f"Unexpected liveness probe path '{liveness_probe.httpGet.path}' "
        f"(expected '{AI_GATEWAY_CONTROLLER_LIVENESS_PROBE_PATH}')"
    )
    assert liveness_probe.httpGet.port == AI_GATEWAY_CONTROLLER_HEALTH_PROBE_PORT, (
        f"Unexpected liveness probe port '{liveness_probe.httpGet.port}' "
        f"(expected {AI_GATEWAY_CONTROLLER_HEALTH_PROBE_PORT})"
    )

    readiness_probe = manager_container.readinessProbe
    assert readiness_probe is not None, (
        f"Deployment '{controller_deployment.namespace}/{controller_deployment.name}' "
        f"container '{AI_GATEWAY_CONTROLLER_MANAGER_CONTAINER_NAME}' has no readinessProbe"
    )
    assert readiness_probe.httpGet.path == AI_GATEWAY_CONTROLLER_READINESS_PROBE_PATH, (
        f"Unexpected readiness probe path '{readiness_probe.httpGet.path}' "
        f"(expected '{AI_GATEWAY_CONTROLLER_READINESS_PROBE_PATH}')"
    )
    assert readiness_probe.httpGet.port == AI_GATEWAY_CONTROLLER_HEALTH_PROBE_PORT, (
        f"Unexpected readiness probe port '{readiness_probe.httpGet.port}' "
        f"(expected {AI_GATEWAY_CONTROLLER_HEALTH_PROBE_PORT})"
    )

    LOGGER.info(
        f"Deployment '{controller_deployment.namespace}/{controller_deployment.name}' "
        "has liveness and readiness probes configured on the manager container"
    )


def verify_ai_gateway_controller_praxis_image_env_from_configmap(admin_client: DynamicClient) -> None:
    """Assert RELATED_IMAGE_ODH_PRAXIS_EXTPROC_IMAGE is wired from ai-gateway-controller-parameters."""
    controller_deployment = get_ai_gateway_controller_deployment(admin_client=admin_client)
    parameters_configmap = ConfigMap(
        client=admin_client,
        name=AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME,
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )

    configmap_data: dict[str, str] = dict(parameters_configmap.instance.data or {})
    assert PRAXIS_EXTPROC_IMAGE_CONFIGMAP_KEY in configmap_data, (
        f"ConfigMap '{parameters_configmap.namespace}/{AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME}' "
        f"is missing key '{PRAXIS_EXTPROC_IMAGE_CONFIGMAP_KEY}'"
    )
    assert configmap_data[PRAXIS_EXTPROC_IMAGE_CONFIGMAP_KEY].strip(), (
        f"ConfigMap key '{PRAXIS_EXTPROC_IMAGE_CONFIGMAP_KEY}' is empty in "
        f"'{parameters_configmap.namespace}/{AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME}'"
    )

    manager_container = _find_manager_container(deployment=controller_deployment)
    praxis_env_var = None
    for env_var in manager_container.env or []:
        if env_var.name == RELATED_IMAGE_ODH_PRAXIS_EXTPROC_ENV_NAME:
            praxis_env_var = env_var
            break

    assert praxis_env_var is not None, (
        f"Deployment '{controller_deployment.namespace}/{controller_deployment.name}' "
        f"container '{AI_GATEWAY_CONTROLLER_MANAGER_CONTAINER_NAME}' is missing env "
        f"'{RELATED_IMAGE_ODH_PRAXIS_EXTPROC_ENV_NAME}'"
    )

    configmap_key_ref = praxis_env_var.valueFrom.configMapKeyRef
    assert configmap_key_ref is not None, (
        f"Env '{RELATED_IMAGE_ODH_PRAXIS_EXTPROC_ENV_NAME}' is not sourced from a ConfigMap key reference"
    )
    assert configmap_key_ref.name == AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME, (
        f"Env '{RELATED_IMAGE_ODH_PRAXIS_EXTPROC_ENV_NAME}' references ConfigMap "
        f"'{configmap_key_ref.name}', expected '{AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME}'"
    )
    assert configmap_key_ref.key == PRAXIS_EXTPROC_IMAGE_CONFIGMAP_KEY, (
        f"Env '{RELATED_IMAGE_ODH_PRAXIS_EXTPROC_ENV_NAME}' references key "
        f"'{configmap_key_ref.key}', expected '{PRAXIS_EXTPROC_IMAGE_CONFIGMAP_KEY}'"
    )

    LOGGER.info(
        f"Deployment '{controller_deployment.namespace}/{controller_deployment.name}' "
        f"env '{RELATED_IMAGE_ODH_PRAXIS_EXTPROC_ENV_NAME}' is wired from "
        f"ConfigMap '{AI_GATEWAY_CONTROLLER_PARAMETERS_CONFIGMAP_NAME}'"
    )
