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
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, retry

from tests.ai_gateway.models_as_a_service.praxis.constants import (
    LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY,
    LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY,
    PRAXIS_EXTPROC_CONFIG_DATA_KEY,
    PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY,
)
from tests.ai_gateway.models_as_a_service.utils import create_api_key, revoke_api_key
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
BBR_PRE_FILTER_NAME: str = "envoy.filters.http.ext_proc.ipp-pre"
BBR_POST_FILTER_NAME: str = "envoy.filters.http.ext_proc.ipp"
ENVOY_FILTER_INSERT_BEFORE: str = "INSERT_BEFORE"
ENVOY_FILTER_INSERT_AFTER: str = "INSERT_AFTER"
ENVOY_FILTER_APPLY_TO_CLUSTER: str = "CLUSTER"
BBR_PLUGINS_CONFIGMAP_NAME: str = "payload-processing-plugins"
BBR_POST_AUTH_CONFIGMAP_KEY: str = LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY
BBR_PRE_AUTH_CONFIGMAP_KEY: str = LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY
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
BBR_STREAMING_SSE_MAX_LINES: int = 30


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
    try:
        yield plaintext_key
    finally:
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


def _attr_or_key(obj: Any, name: str, alternate: str | None = None) -> Any:
    """Return the first present attribute or dict key from obj for name or alternate."""
    candidate_names = (name, alternate) if alternate is not None else (name,)
    for candidate_name in candidate_names:
        if obj is None:
            return None
        if isinstance(obj, dict):
            if candidate_name in obj:
                return obj[candidate_name]
            continue
        value = getattr(obj, candidate_name, None)
        if value is not None:
            return value
    return None


def _config_patch_apply_to(config_patch: Any) -> str | None:
    """Return the configPatch applyTo value."""
    return _attr_or_key(obj=config_patch, name="applyTo", alternate="apply_to")


def _extract_cluster_name_from_patch_value(patch_value: Any) -> str | None:
    """Extract gRPC cluster name from a configPatch value object; handles both camelCase and snake_case."""
    typed_config = _attr_or_key(obj=patch_value, name="typedConfig", alternate="typed_config")
    if not typed_config:
        return None
    grpc_service = _attr_or_key(obj=typed_config, name="grpcService", alternate="grpc_service")
    if not grpc_service:
        return None
    envoy_grpc = _attr_or_key(obj=grpc_service, name="envoyGrpc", alternate="envoy_grpc")
    if not envoy_grpc:
        return None
    return _attr_or_key(obj=envoy_grpc, name="clusterName", alternate="cluster_name")


def _bbr_stage_extproc_cluster_names_by_filter(config_patches: list[Any]) -> dict[str, str]:
    """Map BBR pre/post HTTP filter names to their ext_proc gRPC cluster names."""
    bbr_stage_filter_names = {BBR_PRE_FILTER_NAME, BBR_POST_FILTER_NAME}
    cluster_names_by_filter: dict[str, str] = {}
    for config_patch in config_patches:
        patch = _attr_or_key(obj=config_patch, name="patch")
        patch_value = _attr_or_key(obj=patch, name="value")
        if patch_value is None:
            continue
        filter_name = _attr_or_key(obj=patch_value, name="name")
        if filter_name not in bbr_stage_filter_names:
            continue
        cluster_name = _extract_cluster_name_from_patch_value(patch_value=patch_value)
        if cluster_name:
            cluster_names_by_filter[filter_name] = cluster_name
    return cluster_names_by_filter


def _cluster_endpoint_address_from_cluster_patch_value(patch_value: Any) -> tuple[str | None, str | None]:
    """Return (cluster name, upstream socket address) from a CLUSTER configPatch value."""
    cluster_name = _attr_or_key(obj=patch_value, name="name")
    load_assignment = _attr_or_key(obj=patch_value, name="loadAssignment", alternate="load_assignment")
    endpoints = _attr_or_key(obj=load_assignment, name="endpoints") if load_assignment is not None else None
    if not endpoints:
        return cluster_name, None
    first_endpoint = endpoints[0]
    lb_endpoints = _attr_or_key(obj=first_endpoint, name="lbEndpoints", alternate="lb_endpoints")
    if not lb_endpoints:
        return cluster_name, None
    endpoint = _attr_or_key(obj=lb_endpoints[0], name="endpoint")
    address = _attr_or_key(obj=endpoint, name="address")
    socket_address = _attr_or_key(obj=address, name="socketAddress", alternate="socket_address")
    endpoint_address = _attr_or_key(obj=socket_address, name="address")
    return cluster_name, endpoint_address


def _cluster_endpoint_addresses_by_name(config_patches: list[Any]) -> dict[str, str]:
    """Collect upstream socket addresses from CLUSTER configPatches keyed by cluster name."""
    addresses_by_name: dict[str, str] = {}
    for config_patch in config_patches:
        if _config_patch_apply_to(config_patch) != ENVOY_FILTER_APPLY_TO_CLUSTER:
            continue
        patch = _attr_or_key(obj=config_patch, name="patch")
        patch_value = _attr_or_key(obj=patch, name="value")
        if patch_value is None:
            continue
        cluster_name, endpoint_address = _cluster_endpoint_address_from_cluster_patch_value(patch_value=patch_value)
        if cluster_name and endpoint_address:
            addresses_by_name[cluster_name] = endpoint_address
    return addresses_by_name


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
    """Assert BBR pre/post ext_proc CLUSTER upstream addresses use Services in the gateway namespace."""
    bbr_stage_filter_names = {BBR_PRE_FILTER_NAME, BBR_POST_FILTER_NAME}
    cluster_names_by_filter = _bbr_stage_extproc_cluster_names_by_filter(config_patches=config_patches)
    missing_filters = bbr_stage_filter_names - cluster_names_by_filter.keys()
    assert not missing_filters, (
        f"No gRPC cluster names found for BBR stage filters {sorted(missing_filters)!r} in EnvoyFilter "
        f"'{BBR_ENVOY_FILTER_NAME}' configPatches (found: {sorted(cluster_names_by_filter)!r})"
    )
    cluster_endpoint_addresses = _cluster_endpoint_addresses_by_name(config_patches=config_patches)
    fqdn_segment = f".{gateway_namespace}.svc"
    for filter_name, cluster_name in cluster_names_by_filter.items():
        endpoint_address = cluster_endpoint_addresses.get(cluster_name)
        assert endpoint_address, (
            f"No CLUSTER configPatch with upstream address found for BBR cluster '{cluster_name}' "
            f"(filter '{filter_name}') in EnvoyFilter '{BBR_ENVOY_FILTER_NAME}'"
        )
        assert fqdn_segment in endpoint_address, (
            f"BBR cluster '{cluster_name}' for filter '{filter_name}' resolves to '{endpoint_address}', "
            f"expected gateway namespace '{gateway_namespace}' in the service FQDN "
            f"(expected '{fqdn_segment}' in the address)"
        )
    LOGGER.info(
        f"BBR stage ext_proc clusters reference gateway namespace '{gateway_namespace}': "
        f"{cluster_names_by_filter!r} -> {cluster_endpoint_addresses!r}"
    )


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


def resolve_bbr_plugins_configmap_keys(config_map_data: dict[str, str]) -> tuple[str, str]:
    """Return post-auth and pre-auth data keys for Praxis default or legacy IPP plugin configs."""
    praxis_post_key = PRAXIS_EXTPROC_CONFIG_DATA_KEY
    praxis_pre_key = PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY
    legacy_post_key = LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY
    legacy_pre_key = LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY

    if config_map_data.get(praxis_post_key) and config_map_data.get(praxis_pre_key):
        return praxis_post_key, praxis_pre_key
    if config_map_data.get(legacy_post_key) and config_map_data.get(legacy_pre_key):
        return legacy_post_key, legacy_pre_key

    raise AssertionError(
        f"Expected Praxis keys ({praxis_post_key!r}, {praxis_pre_key!r}) or legacy IPP keys "
        f"({legacy_post_key!r}, {legacy_pre_key!r}) in ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'; "
        f"found keys: {sorted(config_map_data)!r}"
    )


def _fetch_bbr_plugins_configmap_data_if_ready(
    admin_client: DynamicClient,
    gateway_namespace: str,
) -> dict[str, str]:
    """Return ConfigMap data when both plugin config keys are populated."""
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
    resolve_bbr_plugins_configmap_keys(config_map_data=config_map_data)
    return config_map_data


def wait_for_bbr_plugins_configmap_data(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    timeout: int = 300,
    sleep: int = 5,
) -> dict[str, str]:
    """Poll until the BBR plugins ConfigMap exists with post-auth and pre-auth keys populated."""
    try:
        for config_map_data in TimeoutSampler(
            wait_timeout=timeout,
            sleep=sleep,
            func=_fetch_bbr_plugins_configmap_data_if_ready,
            exceptions_dict={AssertionError: []},
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
        ):
            LOGGER.info(
                f"ConfigMap '{gateway_namespace}/{BBR_PLUGINS_CONFIGMAP_NAME}' has post-auth and pre-auth plugin keys"
            )
            return config_map_data
    except TimeoutExpiredError:
        pending_config_map = ConfigMap(
            client=admin_client,
            name=BBR_PLUGINS_CONFIGMAP_NAME,
            namespace=gateway_namespace,
        )
        if not pending_config_map.exists:
            pytest.fail(
                f"ConfigMap '{gateway_namespace}/{BBR_PLUGINS_CONFIGMAP_NAME}' not found after {timeout}s — "
                "expected to be created by the controller after reconciliation"
            )
        found_keys = list((pending_config_map.instance.data or {}).keys())
        pytest.fail(
            f"Timed out after {timeout}s waiting for plugin keys in "
            f"ConfigMap '{gateway_namespace}/{BBR_PLUGINS_CONFIGMAP_NAME}'. "
            f"Expected Praxis keys ({PRAXIS_EXTPROC_CONFIG_DATA_KEY!r}, "
            f"{PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY!r}) or legacy IPP keys "
            f"({LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY!r}, {LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY!r}); "
            f"found: {found_keys!r}"
        )
    raise AssertionError(f"ConfigMap '{gateway_namespace}/{BBR_PLUGINS_CONFIGMAP_NAME}' plugin keys were not populated")


def _verify_bbr_legacy_ipp_plugins_configmap_plugin_specs(
    config_map_data: dict[str, str],
    post_auth_configmap_key: str,
    pre_auth_configmap_key: str,
) -> None:
    """Assert legacy PayloadProcessorConfig entries define expected BBR pre-auth and post-auth plugins."""
    post_auth_config = yaml.safe_load(config_map_data[post_auth_configmap_key])
    assert isinstance(post_auth_config, dict), (
        f"Expected mapping in '{post_auth_configmap_key}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    assert "plugins" in post_auth_config, (
        f"No 'plugins' key in '{post_auth_configmap_key}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    post_auth_plugin_types = [plugin["type"] for plugin in post_auth_config["plugins"]]
    for expected_type in BBR_POST_AUTH_EXPECTED_PLUGIN_TYPES:
        assert expected_type in post_auth_plugin_types, (
            f"Post-auth plugin '{expected_type}' not found in '{post_auth_configmap_key}' — "
            f"found: {post_auth_plugin_types!r}"
        )
    LOGGER.info(f"Post-auth plugins verified: {post_auth_plugin_types!r}")

    pre_auth_config = yaml.safe_load(config_map_data[pre_auth_configmap_key])
    assert isinstance(pre_auth_config, dict), (
        f"Expected mapping in '{pre_auth_configmap_key}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    assert "plugins" in pre_auth_config, (
        f"No 'plugins' key in '{pre_auth_configmap_key}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    pre_auth_plugin_types = [plugin["type"] for plugin in pre_auth_config["plugins"]]
    assert BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE in pre_auth_plugin_types, (
        f"Pre-auth plugin '{BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE}' not found in '{pre_auth_configmap_key}' — "
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


def _praxis_pre_extproc_references_bbr_model_header(pre_auth_yaml: str) -> bool:
    """Return True when pre-extproc YAML maps the JSON model field to the gateway model header."""
    normalized = pre_auth_yaml.lower()
    header_markers = (
        BBR_PRE_AUTH_PLUGIN_HEADER_NAME.lower(),
        "x-gateway-model-name",
    )
    if not any(marker in normalized for marker in header_markers):
        return False
    field_markers = (
        f"fieldname: {BBR_PRE_AUTH_PLUGIN_FIELD_NAME}",
        f"field_name: {BBR_PRE_AUTH_PLUGIN_FIELD_NAME}",
        f"field: {BBR_PRE_AUTH_PLUGIN_FIELD_NAME}",
        f'"{BBR_PRE_AUTH_PLUGIN_FIELD_NAME}"',
        f"'{BBR_PRE_AUTH_PLUGIN_FIELD_NAME}'",
    )
    if any(marker in normalized for marker in field_markers):
        return True
    # Praxis filter chains often reference the JSON path field name without legacy IPP parameter keys.
    if BBR_PRE_AUTH_PLUGIN_FIELD_NAME in normalized:
        return True
    # Legacy IPP plugin type name may still appear when AGC embeds a translated snippet.
    return BBR_PRE_AUTH_EXPECTED_PLUGIN_TYPE in pre_auth_yaml


def _verify_bbr_praxis_extproc_plugins_configmap_content(
    config_map_data: dict[str, str],
    post_auth_configmap_key: str,
    pre_auth_configmap_key: str,
) -> None:
    """Assert Praxis extproc YAML is present and encodes BBR pre-auth model extraction.

    Post-auth Praxis config uses ``server`` / ``filter_chains`` (ai-gateway-controller), not legacy
    ``PayloadProcessorConfig`` plugin ``type`` strings such as ``model-provider-resolver``.
    """
    post_auth_yaml = config_map_data[post_auth_configmap_key]
    pre_auth_yaml = config_map_data[pre_auth_configmap_key]
    assert post_auth_yaml.strip(), f"Empty '{post_auth_configmap_key}' in ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    assert pre_auth_yaml.strip(), f"Empty '{pre_auth_configmap_key}' in ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    post_auth_config = yaml.safe_load(post_auth_yaml)
    pre_auth_config = yaml.safe_load(pre_auth_yaml)
    assert post_auth_config is not None, (
        f"Could not parse YAML in '{post_auth_configmap_key}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    assert pre_auth_config is not None, (
        f"Could not parse YAML in '{pre_auth_configmap_key}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    assert isinstance(post_auth_config, dict), (
        f"Expected mapping in '{post_auth_configmap_key}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )
    assert isinstance(pre_auth_config, dict), (
        f"Expected mapping in '{pre_auth_configmap_key}' of ConfigMap '{BBR_PLUGINS_CONFIGMAP_NAME}'"
    )

    server = post_auth_config.get("server")
    assert isinstance(server, dict), (
        f"Praxis post-auth '{post_auth_configmap_key}' missing 'server' mapping — keys: {sorted(post_auth_config)!r}"
    )
    assert server.get("grpc_address"), f"Praxis post-auth '{post_auth_configmap_key}' server.grpc_address is empty"
    filter_chains = post_auth_config.get("filter_chains")
    if filter_chains is not None:
        assert isinstance(filter_chains, list) and filter_chains, (
            f"Praxis post-auth '{post_auth_configmap_key}' filter_chains must be a non-empty list"
        )
    LOGGER.info(
        f"Praxis post-auth extproc server verified in '{post_auth_configmap_key}' "
        f"(filter_chains={'present' if filter_chains else 'absent'})"
    )

    assert _praxis_pre_extproc_references_bbr_model_header(pre_auth_yaml=pre_auth_yaml), (
        f"Pre-auth '{pre_auth_configmap_key}' does not map JSON field "
        f"'{BBR_PRE_AUTH_PLUGIN_FIELD_NAME}' to header '{BBR_PRE_AUTH_PLUGIN_HEADER_NAME}'"
    )
    LOGGER.info(
        f"Praxis pre-auth extproc config verified: '{BBR_PRE_AUTH_PLUGIN_FIELD_NAME}' → "
        f"'{BBR_PRE_AUTH_PLUGIN_HEADER_NAME}'"
    )


def verify_bbr_plugins_configmap_has_expected_plugins(
    admin_client: DynamicClient,
    gateway_namespace: str = MAAS_GATEWAY_NAMESPACE,
    timeout: int = 300,
    sleep: int = 5,
) -> None:
    """Assert the payload-processing-plugins ConfigMap exists with expected post-auth and pre-auth plugin types."""
    config_map_data = wait_for_bbr_plugins_configmap_data(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        timeout=timeout,
        sleep=sleep,
    )
    post_auth_configmap_key, pre_auth_configmap_key = resolve_bbr_plugins_configmap_keys(
        config_map_data=config_map_data,
    )

    if post_auth_configmap_key == PRAXIS_EXTPROC_CONFIG_DATA_KEY:
        _verify_bbr_praxis_extproc_plugins_configmap_content(
            config_map_data=config_map_data,
            post_auth_configmap_key=post_auth_configmap_key,
            pre_auth_configmap_key=pre_auth_configmap_key,
        )
        return

    _verify_bbr_legacy_ipp_plugins_configmap_plugin_specs(
        config_map_data=config_map_data,
        post_auth_configmap_key=post_auth_configmap_key,
        pre_auth_configmap_key=pre_auth_configmap_key,
    )


def warm_up_bbr_inference_upstream(
    session: requests.Session,
    inference_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> None:
    """Warm up the BBR inference endpoint until Envoy upstream is routable.

    Retries POSTs every 3s for up to 30s while the gateway returns a transient
    503 with 'no healthy upstream' in the response body.
    """
    try:
        _send_bbr_warm_up_request(
            session=session,
            inference_url=inference_url,
            headers=headers,
            payload=payload,
        )
    except TimeoutExpiredError:
        pytest.fail(
            f"BBR inference warm-up retries exhausted for {inference_url} "
            f"(transient 503 no healthy upstream did not clear within 30s)"
        )


@retry(wait_timeout=30, sleep=3)
def _send_bbr_warm_up_request(
    session: requests.Session,
    inference_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> bool:
    """Send one warm-up request; return True to stop retrying, False to retry."""
    LOGGER.info(f"BBR inference warm-up POST {inference_url}")
    response = session.post(url=inference_url, headers=headers, json=payload, timeout=60)
    body = response.text or ""
    LOGGER.info(f"BBR inference warm-up returned {response.status_code}")
    return not (response.status_code == 503 and "no healthy upstream" in body)


def assert_bbr_inference_status(
    session: requests.Session,
    inference_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    expected_status: int,
) -> None:
    """Verify a POST to the BBR inference endpoint returns the expected HTTP status."""
    warm_up_bbr_inference_upstream(
        session=session,
        inference_url=inference_url,
        headers=headers,
        payload=payload,
    )
    response = session.post(url=inference_url, headers=headers, json=payload, timeout=60)
    assert response.status_code == expected_status, (
        f"Expected {expected_status} on BBR inference, got {response.status_code}"
    )
    LOGGER.info(f"BBR inference POST {inference_url} returned {response.status_code}")


def assert_bbr_inference_streaming_returns_sse(
    session: requests.Session,
    inference_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> None:
    """Verify streaming BBR inference returns 200, text/event-stream, and SSE data: lines.

    Retries the streaming POST every 3s for up to 30s when the gateway returns a transient
    503 or a 200 event-stream body with no data: lines yet.
    """
    warm_up_bbr_inference_upstream(
        session=session,
        inference_url=inference_url,
        headers=headers,
        payload=payload,
    )
    try:
        _send_bbr_streaming_sse_request(
            session=session,
            inference_url=inference_url,
            headers=headers,
            payload=payload,
        )
    except TimeoutExpiredError:
        pytest.fail(
            f"BBR streaming SSE retries exhausted for {inference_url} (no data: lines in text/event-stream within 30s)"
        )


@retry(wait_timeout=30, sleep=3)
def _send_bbr_streaming_sse_request(
    session: requests.Session,
    inference_url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> bool:
    """POST with stream=True; return True when SSE data: lines are present, False to retry."""
    LOGGER.info(f"BBR streaming inference POST {inference_url}")
    with session.post(
        url=inference_url,
        headers=headers,
        json=payload,
        timeout=60,
        stream=True,
    ) as response:
        if response.status_code == 503 and "no healthy upstream" in (response.text or ""):
            LOGGER.info("BBR streaming inference returned 503 no healthy upstream")
            return False
        if response.status_code != 200:
            pytest.fail(f"Expected 200 for streaming BBR inference, got {response.status_code}")
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" not in content_type:
            pytest.fail(f"Expected text/event-stream content type for streaming response, got '{content_type}'")
        data_lines = [
            line
            for line, _ in zip(response.iter_lines(), range(BBR_STREAMING_SSE_MAX_LINES), strict=False)
            if line and line.startswith(b"data:")
        ]
        if data_lines:
            LOGGER.info(f"BBR streaming inference returned {len(data_lines)} SSE data: lines")
            return True
        LOGGER.info("BBR streaming inference returned 200 event-stream but no data: lines yet")
        return False
