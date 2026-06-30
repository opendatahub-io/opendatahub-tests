"""Utilities and verification helpers for body-based routing (BBR) tests."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
import requests
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.service import Service

from tests.model_serving.maas_billing.utils import create_api_key, revoke_api_key
from utilities.constants import MAAS_GATEWAY_NAMESPACE
from utilities.general import generate_random_name
from utilities.resources.destination_rule import DestinationRule
from utilities.resources.envoy_filter import EnvoyFilter

LOGGER = structlog.get_logger(name=__name__)

BBR_PRE_PROCESSING_DEPLOYMENT_NAME: str = "payload-pre-processing"
BBR_PRE_PROCESSING_SERVICE_NAME: str = "payload-pre-processing"
BBR_PRE_PROCESSING_GRPC_PORT: int = 9004
BBR_PRE_PROCESSING_DESTINATION_RULE_NAME: str = "payload-pre-processing"
BBR_POST_PROCESSING_DEPLOYMENT_NAME: str = "payload-processing"
BBR_POST_PROCESSING_SERVICE_NAME: str = "payload-processing"
BBR_POST_PROCESSING_GRPC_PORT: int = 9004
BBR_POST_PROCESSING_DESTINATION_RULE_NAME: str = "payload-processing"
BBR_ENVOY_FILTER_NAME: str = "payload-processing"
BBR_PRE_FILTER_NAME: str = "envoy.filters.http.ext_proc.bbr-pre"
BBR_POST_FILTER_NAME: str = "envoy.filters.http.ext_proc.bbr"
ENVOY_FILTER_INSERT_BEFORE: str = "INSERT_BEFORE"
ENVOY_FILTER_INSERT_AFTER: str = "INSERT_AFTER"
BBR_PLUGINS_CONFIGMAP_NAME: str = "payload-processing-plugins"
BBR_POST_AUTH_CONFIGMAP_KEY: str = "custom-ipp-config.yaml"
BBR_PRE_AUTH_CONFIGMAP_KEY: str = "custom-pre-processing-ipp-config.yaml"
BBR_POST_AUTH_EXPECTED_PLUGIN_TYPES: list[str] = [
    "model-provider-resolver",
    "api-translation",
    "apikey-injection",
]
BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE: str = "body-field-to-header"
BBR_PRE_AUTH_PLUGIN_FIELD_NAME: str = "model"
BBR_PRE_AUTH_PLUGIN_HEADER_NAME: str = "X-Gateway-Model-Name"
BBR_RATE_LIMIT_TOKENS_PER_MINUTE: int = 100
BBR_RATE_LIMIT_CHAT_MAX_TOKENS: int = 80


@contextmanager
def bbr_api_key_lifecycle(
    request_session_http: requests.Session,
    base_url: str,
    ocp_token_for_actor: str,
    subscription_name: str,
    key_name_prefix: str,
    fixture_label: str,
) -> Generator[str, Any, Any]:
    """Create a BBR API key, yield the plaintext key, then revoke it on teardown."""
    key_name = f"{key_name_prefix}-{generate_random_name()}"
    _, api_key_data = create_api_key(
        base_url=base_url,
        ocp_user_token=ocp_token_for_actor,
        request_session_http=request_session_http,
        api_key_name=key_name,
        subscription=subscription_name,
    )
    assert "id" in api_key_data, f"{fixture_label}: create_api_key response missing 'id'"
    assert "key" in api_key_data, f"{fixture_label}: create_api_key response missing 'key'"
    key_id: str = api_key_data["id"]
    plaintext_key: str = api_key_data["key"]
    LOGGER.info(f"{fixture_label}: created key id={key_id} name={key_name}")
    yield plaintext_key
    revoke_response, _ = revoke_api_key(
        request_session_http=request_session_http,
        base_url=base_url,
        key_id=key_id,
        ocp_user_token=ocp_token_for_actor,
    )
    if revoke_response.status_code not in (200, 404):
        raise AssertionError(
            f"Unexpected teardown status for {fixture_label} key id={key_id}: {revoke_response.status_code}"
        )


def verify_bbr_pre_processing_deployment_ready(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-pre-processing Deployment exists and all replicas are ready."""
    deployment = Deployment(
        client=admin_client,
        name=BBR_PRE_PROCESSING_DEPLOYMENT_NAME,
        namespace=gateway_namespace,
    )
    assert deployment.exists, (
        f"Deployment '{gateway_namespace}/{BBR_PRE_PROCESSING_DEPLOYMENT_NAME}' not found — "
        "expected to be created by the controller after reconciliation"
    )
    ready_replicas: int = deployment.instance.status.readyReplicas or 0
    desired_replicas: int = deployment.instance.spec.replicas if deployment.instance.spec.replicas is not None else 1
    assert ready_replicas >= desired_replicas, (
        f"Deployment '{BBR_PRE_PROCESSING_DEPLOYMENT_NAME}' has {ready_replicas}/{desired_replicas} ready replicas "
        f"in namespace '{gateway_namespace}'"
    )
    LOGGER.info(
        f"Deployment '{gateway_namespace}/{BBR_PRE_PROCESSING_DEPLOYMENT_NAME}' is ready "
        f"({ready_replicas}/{desired_replicas} replicas)"
    )


def verify_bbr_pre_processing_service_port(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-pre-processing Service exists and exposes the expected gRPC port."""
    service = Service(
        client=admin_client,
        name=BBR_PRE_PROCESSING_SERVICE_NAME,
        namespace=gateway_namespace,
    )
    assert service.exists, f"Service '{gateway_namespace}/{BBR_PRE_PROCESSING_SERVICE_NAME}' not found"
    service_ports = service.instance.spec.ports or []
    exposed_port_numbers = [port.port for port in service_ports]
    assert BBR_PRE_PROCESSING_GRPC_PORT in exposed_port_numbers, (
        f"Service '{gateway_namespace}/{BBR_PRE_PROCESSING_SERVICE_NAME}' does not expose "
        f"port {BBR_PRE_PROCESSING_GRPC_PORT} — found: {exposed_port_numbers!r}"
    )
    LOGGER.info(
        f"Service '{gateway_namespace}/{BBR_PRE_PROCESSING_SERVICE_NAME}' "
        f"exposes gRPC port {BBR_PRE_PROCESSING_GRPC_PORT}"
    )


def verify_bbr_pre_processing_destination_rule_exists(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-pre-processing DestinationRule exists in the gateway namespace."""
    destination_rule = DestinationRule(
        client=admin_client,
        name=BBR_PRE_PROCESSING_DESTINATION_RULE_NAME,
        namespace=gateway_namespace,
    )
    assert destination_rule.exists, (
        f"DestinationRule '{gateway_namespace}/{BBR_PRE_PROCESSING_DESTINATION_RULE_NAME}' not found — "
        "expected to be created by the controller after reconciliation"
    )
    LOGGER.info(f"DestinationRule '{gateway_namespace}/{BBR_PRE_PROCESSING_DESTINATION_RULE_NAME}' exists")


def get_bbr_envoy_filter_config_patches(
    admin_client: DynamicClient,
    gateway_namespace: str,
) -> list[Any]:
    """Assert the BBR EnvoyFilter exists and return its configPatches."""
    envoy_filter = EnvoyFilter(
        client=admin_client,
        name=BBR_ENVOY_FILTER_NAME,
        namespace=gateway_namespace,
    )
    assert envoy_filter.exists, (
        f"EnvoyFilter '{gateway_namespace}/{BBR_ENVOY_FILTER_NAME}' not found — "
        "expected after the controller reconciles"
    )
    return envoy_filter.instance.spec.configPatches or []


def _extract_cluster_name_from_patch_value(patch_value: Any) -> str | None:
    """Extract gRPC cluster name from a configPatch value object; handles both camelCase and snake_case."""
    typed_config = getattr(patch_value, "typedConfig", None) or getattr(patch_value, "typed_config", None)
    if not typed_config:
        return None
    grpc_service = getattr(typed_config, "grpcService", None) or getattr(typed_config, "grpc_service", None)
    if not grpc_service:
        return None
    envoy_grpc = getattr(grpc_service, "envoyGrpc", None) or getattr(grpc_service, "envoy_grpc", None)
    if not envoy_grpc:
        return None
    return getattr(envoy_grpc, "clusterName", None) or getattr(envoy_grpc, "cluster_name", None)


def verify_bbr_envoy_filter_has_pre_and_post_auth_stages(
    config_patches: list[Any],
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the BBR EnvoyFilter contains both the pre-auth (bbr-pre) and post-auth (bbr) filter patches."""
    filter_names: list[str] = []
    for config_patch in config_patches:
        patch = getattr(config_patch, "patch", None)
        patch_value = getattr(patch, "value", None) if patch is not None else None
        filter_name = getattr(patch_value, "name", None) if patch_value is not None else None
        if filter_name:
            filter_names.append(filter_name)
    assert BBR_PRE_FILTER_NAME in filter_names, (
        f"EnvoyFilter '{BBR_ENVOY_FILTER_NAME}' missing pre-auth stage '{BBR_PRE_FILTER_NAME}' — "
        f"found filters: {filter_names!r}"
    )
    assert BBR_POST_FILTER_NAME in filter_names, (
        f"EnvoyFilter '{BBR_ENVOY_FILTER_NAME}' missing post-auth stage '{BBR_POST_FILTER_NAME}' — "
        f"found filters: {filter_names!r}"
    )
    LOGGER.info(
        f"EnvoyFilter '{gateway_namespace}/{BBR_ENVOY_FILTER_NAME}' has pre-auth and post-auth stages: {filter_names!r}"
    )


def verify_bbr_stage_filter_operation(
    config_patches: list[Any],
    filter_name: str,
    expected_operation: str,
    stage_label: str,
    wasm_plugin_position: str,
) -> None:
    """Assert a BBR configPatch uses the expected insert operation relative to the WasmPlugin.

    Args:
        config_patches: EnvoyFilter configPatches fetched once per test class.
        filter_name: Envoy filter name in the configPatch value (e.g. bbr-pre or bbr).
        expected_operation: Expected patch operation (INSERT_BEFORE or INSERT_AFTER).
        stage_label: Human-readable stage label for assertion messages (e.g. Pre-auth).
        wasm_plugin_position: Relative WasmPlugin position for assertion messages (before/after).
    """
    for config_patch in config_patches:
        patch = getattr(config_patch, "patch", None)
        if patch is None:
            continue
        patch_value = getattr(patch, "value", None)
        patch_filter_name = getattr(patch_value, "name", None) if patch_value is not None else None
        if patch_filter_name != filter_name:
            continue
        operation = getattr(patch, "operation", None)
        assert operation == expected_operation, (
            f"{stage_label} stage '{filter_name}' has operation '{operation}', "
            f"expected '{expected_operation}' — it must run {wasm_plugin_position} the WasmPlugin"
        )
        LOGGER.info(f"{stage_label} stage '{filter_name}' correctly uses {expected_operation}")
        return
    pytest.fail(
        f"{stage_label} filter '{filter_name}' not found in EnvoyFilter '{BBR_ENVOY_FILTER_NAME}' configPatches"
    )


def verify_bbr_pre_stage_is_insert_before_wasm_plugin(
    config_patches: list[Any],
) -> None:
    """Assert the bbr-pre configPatch uses INSERT_BEFORE so it runs before the WasmPlugin auth stage."""
    verify_bbr_stage_filter_operation(
        config_patches=config_patches,
        filter_name=BBR_PRE_FILTER_NAME,
        expected_operation=ENVOY_FILTER_INSERT_BEFORE,
        stage_label="Pre-auth",
        wasm_plugin_position="before",
    )


def verify_bbr_post_stage_is_insert_after_wasm_plugin(
    config_patches: list[Any],
) -> None:
    """Assert the bbr post-auth configPatch uses INSERT_AFTER so it runs after the WasmPlugin auth stage."""
    verify_bbr_stage_filter_operation(
        config_patches=config_patches,
        filter_name=BBR_POST_FILTER_NAME,
        expected_operation=ENVOY_FILTER_INSERT_AFTER,
        stage_label="Post-auth",
        wasm_plugin_position="after",
    )


def verify_bbr_envoy_filter_cluster_names_contain_gateway_namespace(
    config_patches: list[Any],
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert all gRPC cluster names in the BBR EnvoyFilter point to services in the gateway namespace."""
    cluster_names: list[str] = []
    for config_patch in config_patches:
        patch = getattr(config_patch, "patch", None)
        patch_value = getattr(patch, "value", None) if patch is not None else None
        if patch_value is None:
            continue
        cluster_name = _extract_cluster_name_from_patch_value(patch_value=patch_value)
        if cluster_name:
            cluster_names.append(cluster_name)
    assert cluster_names, f"No gRPC cluster names found in EnvoyFilter '{BBR_ENVOY_FILTER_NAME}' configPatches"
    fqdn_segment = f".{gateway_namespace}.svc"
    for cluster_name in cluster_names:
        assert fqdn_segment in cluster_name, (
            f"Cluster name '{cluster_name}' does not reference gateway namespace '{gateway_namespace}' "
            f"in the service FQDN (expected '{fqdn_segment}' in cluster name)"
        )
    LOGGER.info(f"All gRPC cluster names reference gateway namespace '{gateway_namespace}': {cluster_names!r}")


def verify_bbr_post_auth_processing_deployment_ready(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the post-auth payload-processing Deployment exists and all replicas are ready."""
    deployment = Deployment(
        client=admin_client,
        name=BBR_POST_PROCESSING_DEPLOYMENT_NAME,
        namespace=gateway_namespace,
    )
    assert deployment.exists, f"Deployment '{gateway_namespace}/{BBR_POST_PROCESSING_DEPLOYMENT_NAME}' not found"
    ready_replicas: int = deployment.instance.status.readyReplicas or 0
    desired_replicas: int = deployment.instance.spec.replicas if deployment.instance.spec.replicas is not None else 1
    assert ready_replicas >= desired_replicas, (
        f"Deployment '{BBR_POST_PROCESSING_DEPLOYMENT_NAME}' has {ready_replicas}/{desired_replicas} ready replicas "
        f"in namespace '{gateway_namespace}'"
    )
    LOGGER.info(
        f"Deployment '{gateway_namespace}/{BBR_POST_PROCESSING_DEPLOYMENT_NAME}' is ready "
        f"({ready_replicas}/{desired_replicas} replicas)"
    )


def verify_bbr_post_processing_service_port(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-processing Service exists and exposes the expected gRPC port."""
    service = Service(
        client=admin_client,
        name=BBR_POST_PROCESSING_SERVICE_NAME,
        namespace=gateway_namespace,
    )
    assert service.exists, f"Service '{gateway_namespace}/{BBR_POST_PROCESSING_SERVICE_NAME}' not found"
    service_ports = service.instance.spec.ports or []
    exposed_port_numbers = [port.port for port in service_ports]
    assert BBR_POST_PROCESSING_GRPC_PORT in exposed_port_numbers, (
        f"Service '{gateway_namespace}/{BBR_POST_PROCESSING_SERVICE_NAME}' does not expose "
        f"port {BBR_POST_PROCESSING_GRPC_PORT} — found: {exposed_port_numbers!r}"
    )
    LOGGER.info(
        f"Service '{gateway_namespace}/{BBR_POST_PROCESSING_SERVICE_NAME}' "
        f"exposes gRPC port {BBR_POST_PROCESSING_GRPC_PORT}"
    )


def verify_bbr_post_processing_destination_rule_exists(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-processing DestinationRule exists in the gateway namespace."""
    destination_rule = DestinationRule(
        client=admin_client,
        name=BBR_POST_PROCESSING_DESTINATION_RULE_NAME,
        namespace=gateway_namespace,
    )
    assert destination_rule.exists, (
        f"DestinationRule '{gateway_namespace}/{BBR_POST_PROCESSING_DESTINATION_RULE_NAME}' not found — "
        "expected to be created by the controller after reconciliation"
    )
    LOGGER.info(f"DestinationRule '{gateway_namespace}/{BBR_POST_PROCESSING_DESTINATION_RULE_NAME}' exists")


def verify_bbr_plugins_configmap_has_expected_plugins(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
) -> None:
    """Assert the payload-processing-plugins ConfigMap exists with expected post-auth and pre-auth plugin types."""
    config_map = ConfigMap(
        client=admin_client,
        name=BBR_PLUGINS_CONFIGMAP_NAME,
        namespace=gateway_namespace,
    )
    assert config_map.exists, (
        f"ConfigMap '{gateway_namespace}/{BBR_PLUGINS_CONFIGMAP_NAME}' not found — "
        "expected to be created by the controller after reconciliation"
    )
    config_map_data: dict[str, str] = config_map.instance.data or {}

    assert BBR_POST_AUTH_CONFIGMAP_KEY in config_map_data, (
        f"Key '{BBR_POST_AUTH_CONFIGMAP_KEY}' missing from ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    post_auth_config = yaml.safe_load(config_map_data[BBR_POST_AUTH_CONFIGMAP_KEY])
    assert "plugins" in post_auth_config, (
        f"No 'plugins' key in '{BBR_POST_AUTH_CONFIGMAP_KEY}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    post_auth_plugin_types = [plugin["type"] for plugin in post_auth_config["plugins"]]
    for expected_type in BBR_POST_AUTH_EXPECTED_PLUGIN_TYPES:
        assert expected_type in post_auth_plugin_types, (
            f"Post-auth plugin '{expected_type}' not found in '{BBR_POST_AUTH_CONFIGMAP_KEY}' — "
            f"found: {post_auth_plugin_types!r}"
        )
    LOGGER.info(f"Post-auth plugins verified: {post_auth_plugin_types!r}")

    assert BBR_PRE_AUTH_CONFIGMAP_KEY in config_map_data, (
        f"Key '{BBR_PRE_AUTH_CONFIGMAP_KEY}' missing from ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    pre_auth_config = yaml.safe_load(config_map_data[BBR_PRE_AUTH_CONFIGMAP_KEY])
    assert "plugins" in pre_auth_config, (
        f"No 'plugins' key in '{BBR_PRE_AUTH_CONFIGMAP_KEY}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    pre_auth_plugin_types = [plugin["type"] for plugin in pre_auth_config["plugins"]]
    assert BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE in pre_auth_plugin_types, (
        f"Pre-auth plugin '{BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE}' not found in '{BBR_PRE_AUTH_CONFIGMAP_KEY}' — "
        f"found: {pre_auth_plugin_types!r}"
    )
    pre_auth_plugin = next(
        plugin for plugin in pre_auth_config["plugins"] if plugin["type"] == BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE
    )
    assert "parameters" in pre_auth_plugin, (
        f"Pre-auth plugin '{BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE}' has no 'parameters' key"
    )
    params: dict[str, str] = pre_auth_plugin["parameters"]
    assert params["fieldName"] == BBR_PRE_AUTH_PLUGIN_FIELD_NAME, (
        f"Pre-auth plugin fieldName is '{params['fieldName']}', expected '{BBR_PRE_AUTH_PLUGIN_FIELD_NAME}'"
    )
    assert params["headerName"] == BBR_PRE_AUTH_PLUGIN_HEADER_NAME, (
        f"Pre-auth plugin headerName is '{params['headerName']}', expected '{BBR_PRE_AUTH_PLUGIN_HEADER_NAME}'"
    )
    LOGGER.info(
        f"Pre-auth plugin '{BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE}' correctly maps "
        f"'{BBR_PRE_AUTH_PLUGIN_FIELD_NAME}' → '{BBR_PRE_AUTH_PLUGIN_HEADER_NAME}'"
    )


def assert_bbr_inference_status(
    session: requests.Session,
    inference_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    expected_status: int,
) -> None:
    """Verify a POST to the BBR inference endpoint returns the expected HTTP status."""
    response = session.post(url=inference_url, headers=headers, json=payload, timeout=60)
    assert response.status_code == expected_status, (
        f"Expected {expected_status} on BBR inference, got {response.status_code}"
    )
    LOGGER.info(f"BBR inference POST {inference_url} returned {response.status_code}")
