"""Helpers for Praxis MaasTenantConfig opt-in and maas-controller platform tests."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import (
    AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
    tenant_namespace_name_from_aitenant,
)
from tests.ai_gateway.models_as_a_service.multitenancy.utils import gateway_ref_from_aitenant
from tests.ai_gateway.models_as_a_service.praxis.constants import (
    DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
    DEFAULT_PRAXIS_FINALIZER_WAIT_TIMEOUT_SECONDS,
    LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY,
    LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY,
    LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE,
    LEGACY_IPP_POLL_INTERVAL_SECONDS,
    LEGACY_IPP_POST_PROCESSING_NAME_BASE,
    LEGACY_IPP_PRE_PROCESSING_NAME_BASE,
    LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT_SECONDS,
    LEGACY_POST_PROCESSING_CONTAINER_CONFIG_ARG,
    LEGACY_PRE_PROCESSING_CONTAINER_CONFIG_ARG,
    MAAS_PAYLOAD_PROCESSING_STATUS_ANNOTATION,
    MAAS_PAYLOAD_PROCESSING_STATUS_CLEANUP_COMPLETE_VALUE,
    PRAXIS_CLEANUP_FINALIZER,
    PRAXIS_EXTPROC_CONFIG_DATA_KEY,
    PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION,
    PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
    PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG,
    PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY,
    PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG,
)
from tests.ai_gateway.models_as_a_service.utils import (
    aitenant_from_spec,
    bootstrap_gateway_context,
    bootstrap_gateway_ref,
    build_aitenant_spec,
    deploy_and_verify_aitenant_ready,
    fresh_aitenant,
    verify_aitenant_ready,
    verify_maas_tenant_config_ready,
)
from utilities.general import generate_random_name
from utilities.resources.aitenant import AITenant
from utilities.resources.envoy_filter import EnvoyFilter
from utilities.resources.maastenantconfig import MaasTenantConfig


def fresh_maastenantconfig_for_aitenant(admin_client: DynamicClient, aitenant: AITenant) -> MaasTenantConfig:
    """Return a new handle to re-read MaasTenantConfig from the API."""
    tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=aitenant)
    return MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        wait_for_resource=False,
    )


def maas_tenant_config_for_aitenant(admin_client: DynamicClient, aitenant: AITenant) -> MaasTenantConfig:
    """Return the bootstrapped MaasTenantConfig for a Ready AITenant."""
    tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=aitenant)
    bootstrapped_tenant_config = fresh_maastenantconfig_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    assert bootstrapped_tenant_config.exists, (
        f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} not found in '{tenant_namespace_name}'"
    )
    return bootstrapped_tenant_config


def maastenantconfig_metadata_annotations(bootstrapped_tenant_config: MaasTenantConfig) -> dict[str, str]:
    """Return MaasTenantConfig metadata annotations as a string dict."""
    return dict(bootstrapped_tenant_config.instance.metadata.annotations or {})


def read_maastenantconfig_payload_processing_type(
    admin_client: DynamicClient,
    aitenant: AITenant,
) -> str | None:
    """Return payload-processing-type on MaasTenantConfig, or None when absent."""
    bootstrapped_tenant_config = maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    metadata_annotations = maastenantconfig_metadata_annotations(bootstrapped_tenant_config=bootstrapped_tenant_config)
    if PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION not in metadata_annotations:
        return None
    return str(metadata_annotations[PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION])


def verify_maastenantconfig_payload_processing_type(
    admin_client: DynamicClient,
    aitenant: AITenant,
    expected_value: str | None,
) -> None:
    """Assert MaasTenantConfig carries the expected payload-processing-type annotation."""
    actual_value = read_maastenantconfig_payload_processing_type(admin_client=admin_client, aitenant=aitenant)
    bootstrapped_tenant_config = maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    assert actual_value == expected_value, (
        f"MaasTenantConfig '{bootstrapped_tenant_config.namespace}/{bootstrapped_tenant_config.name}' annotation "
        f"'{PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION}' expected {expected_value!r}, got {actual_value!r}"
    )


def read_aitenant_payload_processing_type(aitenant: AITenant) -> str | None:
    """Return payload-processing-type on AITenant when present (should not mirror MaasTenantConfig opt-in)."""
    metadata_annotations = dict(fresh_aitenant(aitenant=aitenant).instance.metadata.annotations or {})
    if PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION not in metadata_annotations:
        return None
    return str(metadata_annotations[PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION])


def verify_aitenant_lacks_payload_processing_type_annotation(aitenant: AITenant) -> None:
    """Assert the AITenant does not carry payload-processing-type (IPP config lives on MaasTenantConfig)."""
    actual_value = read_aitenant_payload_processing_type(aitenant=aitenant)
    assert actual_value is None, (
        f"AITenant '{aitenant.namespace}/{aitenant.name}' should not have annotation "
        f"'{PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION}'; got {actual_value!r}"
    )


def read_maastenantconfig_finalizers(admin_client: DynamicClient, aitenant: AITenant) -> list[str]:
    """Return finalizer names on MaasTenantConfig, or an empty list when none are set."""
    bootstrapped_tenant_config = maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    metadata_finalizers = bootstrapped_tenant_config.instance.metadata.finalizers
    if not metadata_finalizers:
        return []
    return [str(finalizer) for finalizer in metadata_finalizers]


def verify_maastenantconfig_has_praxis_cleanup_finalizer(
    admin_client: DynamicClient,
    aitenant: AITenant,
    timeout: int = DEFAULT_PRAXIS_FINALIZER_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Assert ai-gateway-controller attached the praxis cleanup finalizer on MaasTenantConfig."""
    bootstrapped_tenant_config = maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    try:
        for has_finalizer in TimeoutSampler(
            wait_timeout=timeout,
            sleep=2,
            func=lambda: (
                PRAXIS_CLEANUP_FINALIZER
                in read_maastenantconfig_finalizers(admin_client=admin_client, aitenant=aitenant)
            ),
        ):
            if has_finalizer:
                return
    except TimeoutExpiredError:
        finalizers = read_maastenantconfig_finalizers(admin_client=admin_client, aitenant=aitenant)
        pytest.fail(
            f"MaasTenantConfig '{bootstrapped_tenant_config.namespace}/{bootstrapped_tenant_config.name}' "
            f"should have finalizer '{PRAXIS_CLEANUP_FINALIZER}' when payload-processing-type is praxis "
            f"(timeout {timeout}s); got {finalizers!r}"
        )


def wait_until_maastenantconfig_lacks_praxis_cleanup_finalizer(
    admin_client: DynamicClient,
    aitenant: AITenant,
    timeout: int = DEFAULT_PRAXIS_FINALIZER_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until the praxis cleanup finalizer is removed from MaasTenantConfig."""
    bootstrapped_tenant_config = maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    try:
        for lacks_finalizer in TimeoutSampler(
            wait_timeout=timeout,
            sleep=2,
            func=lambda: (
                PRAXIS_CLEANUP_FINALIZER
                not in read_maastenantconfig_finalizers(admin_client=admin_client, aitenant=aitenant)
            ),
        ):
            if lacks_finalizer:
                return
    except TimeoutExpiredError:
        finalizers = read_maastenantconfig_finalizers(admin_client=admin_client, aitenant=aitenant)
        pytest.fail(
            f"MaasTenantConfig '{bootstrapped_tenant_config.namespace}/{bootstrapped_tenant_config.name}' "
            f"should not have finalizer '{PRAXIS_CLEANUP_FINALIZER}' after leaving praxis opt-in "
            f"(timeout {timeout}s); got {finalizers!r}"
        )


def verify_maastenantconfig_lacks_praxis_cleanup_finalizer(admin_client: DynamicClient, aitenant: AITenant) -> None:
    """Assert MaasTenantConfig is not on the Praxis controller cleanup path."""
    bootstrapped_tenant_config = maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    finalizers = read_maastenantconfig_finalizers(admin_client=admin_client, aitenant=aitenant)
    assert PRAXIS_CLEANUP_FINALIZER not in finalizers, (
        f"MaasTenantConfig '{bootstrapped_tenant_config.namespace}/{bootstrapped_tenant_config.name}' "
        f"should not have finalizer '{PRAXIS_CLEANUP_FINALIZER}' without effective praxis opt-in; "
        f"got {finalizers!r}"
    )


def set_maastenantconfig_payload_processing_type_annotation(
    admin_client: DynamicClient,
    aitenant: AITenant,
    annotation_value: str | None,
) -> None:
    """Set or remove payload-processing-type on MaasTenantConfig for a Ready AITenant."""
    if annotation_value is None:
        bootstrapped_tenant_config = maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
        annotations = maastenantconfig_metadata_annotations(bootstrapped_tenant_config=bootstrapped_tenant_config)
        if PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION not in annotations:
            return
        annotation_patch: str | None = None
    else:
        annotation_patch = annotation_value

    patch_target = fresh_maastenantconfig_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    ResourceEditor(
        patches={
            patch_target: {
                "metadata": {
                    "annotations": {PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION: annotation_patch},
                },
            },
        },
    ).update()


def verify_maastenantconfig_non_praxis_payload_processing_uses_legacy_ipp(
    admin_client: DynamicClient,
    aitenant: AITenant,
    annotation_value: str,
) -> None:
    """Set a non-praxis payload-processing-type on MaasTenantConfig and assert Praxis cleanup stays off."""
    set_maastenantconfig_payload_processing_type_annotation(
        admin_client=admin_client,
        aitenant=aitenant,
        annotation_value=annotation_value,
    )
    verify_maastenantconfig_payload_processing_type(
        admin_client=admin_client,
        aitenant=aitenant,
        expected_value=annotation_value,
    )
    verify_maastenantconfig_lacks_praxis_cleanup_finalizer(admin_client=admin_client, aitenant=aitenant)
    verify_aitenant_lacks_payload_processing_type_annotation(aitenant=aitenant)


def praxis_aitenant_from_spec(
    admin_client: DynamicClient,
    aitenant_name: str,
    cr_namespace: str,
    aitenant_spec: dict[str, Any],
    teardown: bool = False,
) -> AITenant:
    """Return an AITenant configured from spec without payload-processing annotations (set on MaasTenantConfig)."""
    return aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=aitenant_name,
        cr_namespace=cr_namespace,
        aitenant_spec=aitenant_spec,
        teardown=teardown,
    )


def verify_aitenant_bootstrap_reaches_ready_with_refs(aitenant: AITenant) -> None:
    """Assert the AITenant is Ready with tenantNamespace and gatewayRef populated."""
    verify_aitenant_ready(aitenant=aitenant)
    aitenant_status = fresh_aitenant(aitenant=aitenant).instance.status
    tenant_namespace = getattr(aitenant_status, "tenantNamespace", None)
    assert tenant_namespace, (
        f"AITenant '{aitenant.namespace}/{aitenant.name}' status.tenantNamespace should be set after bootstrap"
    )
    gateway_ref = getattr(aitenant_status, "gatewayRef", None)
    assert gateway_ref is not None, (
        f"AITenant '{aitenant.namespace}/{aitenant.name}' status.gatewayRef should be set after bootstrap"
    )
    assert gateway_ref.name, f"AITenant '{aitenant.namespace}/{aitenant.name}' status.gatewayRef.name should be set"
    assert gateway_ref.namespace, (
        f"AITenant '{aitenant.namespace}/{aitenant.name}' status.gatewayRef.namespace should be set"
    )


@contextmanager
def praxis_aitenant_with_bootstrap_gateway(
    admin_client: DynamicClient,
    cr_namespace: str,
    teardown: bool,
    aitenant_name: str | None = None,
) -> Generator[AITenant]:
    """Yield an AITenant after its bootstrap Gateway exists."""
    resolved_aitenant_name = aitenant_name or f"e2e-praxis-{generate_random_name()}"
    aitenant_spec = build_aitenant_spec(aitenant_name=resolved_aitenant_name)
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=resolved_aitenant_name,
        aitenant_spec=aitenant_spec,
    )
    with (
        bootstrap_gateway_context(
            admin_client=admin_client,
            gateway_name=gateway_name,
            gateway_namespace=gateway_namespace,
            teardown=teardown,
        ),
        praxis_aitenant_from_spec(
            admin_client=admin_client,
            aitenant_name=resolved_aitenant_name,
            cr_namespace=cr_namespace,
            aitenant_spec=aitenant_spec,
            teardown=teardown,
        ) as aitenant,
    ):
        yield aitenant


def wait_until_bootstrapped_maastenantconfig_exists(
    admin_client: DynamicClient,
    aitenant: AITenant,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> MaasTenantConfig:
    """Poll until MaasTenantConfig/default-tenant exists for an AITenant (before or after Ready)."""
    tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=aitenant)
    missing_bootstrapped_tenant_config_message = (
        f"Timed out after {timeout}s waiting for MaasTenantConfig/"
        f"{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} in namespace '{tenant_namespace_name}' "
        f"for AITenant '{aitenant.namespace}/{aitenant.name}'"
    )
    try:
        for bootstrapped_tenant_config_exists in TimeoutSampler(
            wait_timeout=timeout,
            sleep=LEGACY_IPP_POLL_INTERVAL_SECONDS,
            func=lambda: (
                MaasTenantConfig(
                    client=admin_client,
                    name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
                    namespace=tenant_namespace_name,
                    wait_for_resource=False,
                ).exists
            ),
        ):
            if bootstrapped_tenant_config_exists:
                return maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    except TimeoutExpiredError:
        pytest.fail(missing_bootstrapped_tenant_config_message)
    pytest.fail(missing_bootstrapped_tenant_config_message)


def deploy_aitenant_with_maastenantconfig_praxis_opt_in_before_legacy_ipp(
    admin_client: DynamicClient,
    aitenant: AITenant,
    annotation_value: str = PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
) -> None:
    """Create the AITenant, opt into praxis on MaasTenantConfig as soon as it exists, then wait for Ready."""
    if not aitenant.exists:
        aitenant.deploy()
    assert aitenant.exists, f"AITenant '{aitenant.namespace}/{aitenant.name}' was not created"
    wait_until_bootstrapped_maastenantconfig_exists(admin_client=admin_client, aitenant=aitenant)
    set_maastenantconfig_payload_processing_type_annotation(
        admin_client=admin_client,
        aitenant=aitenant,
        annotation_value=annotation_value,
    )
    verify_maastenantconfig_payload_processing_type(
        admin_client=admin_client,
        aitenant=aitenant,
        expected_value=annotation_value,
    )
    deploy_and_verify_aitenant_ready(aitenant=aitenant)


def deploy_praxis_aitenant_and_verify_maastenantconfig_annotation(
    admin_client: DynamicClient,
    aitenant: AITenant,
    expected_annotation_value: str,
) -> None:
    """Create the AITenant if missing, opt in on MaasTenantConfig early, and wait until Ready."""
    deploy_aitenant_with_maastenantconfig_praxis_opt_in_before_legacy_ipp(
        admin_client=admin_client,
        aitenant=aitenant,
        annotation_value=expected_annotation_value,
    )


def per_tenant_legacy_ipp_resource_name(base_name: str, aitenant_name: str) -> str:
    """Return the maas-controller legacy IPP resource name for an AITenant-managed tenant."""
    return f"{base_name}-{aitenant_name}"


def legacy_ipp_post_processing_deployment_name(aitenant_name: str) -> str:
    """Return the per-tenant legacy post-auth IPP Deployment name in the gateway namespace."""
    return per_tenant_legacy_ipp_resource_name(
        base_name=LEGACY_IPP_POST_PROCESSING_NAME_BASE,
        aitenant_name=aitenant_name,
    )


def legacy_ipp_pre_processing_deployment_name(aitenant_name: str) -> str:
    """Return the per-tenant legacy pre-auth IPP Deployment name in the gateway namespace."""
    return per_tenant_legacy_ipp_resource_name(
        base_name=LEGACY_IPP_PRE_PROCESSING_NAME_BASE,
        aitenant_name=aitenant_name,
    )


def legacy_ipp_plugins_configmap_name(aitenant_name: str) -> str:
    """Return the per-tenant legacy IPP plugins ConfigMap name in the gateway namespace."""
    return per_tenant_legacy_ipp_resource_name(
        base_name=LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE,
        aitenant_name=aitenant_name,
    )


def _legacy_ipp_plugins_configmap_data_keys(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> set[str]:
    """Return data keys on the per-tenant legacy IPP plugins ConfigMap, or empty if absent."""
    configmap_name = legacy_ipp_plugins_configmap_name(aitenant_name=aitenant_name)
    plugins_configmap = ConfigMap(
        client=admin_client,
        name=configmap_name,
        namespace=gateway_namespace,
    )
    if not plugins_configmap.exists:
        return set()
    configmap_data: dict[str, str] = dict(plugins_configmap.instance.to_dict().get("data") or {})
    return set(configmap_data.keys())


def _legacy_ipp_plugins_configmap_has_full_maas_config(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> bool:
    """Return True when the tenant plugins ConfigMap includes both maas legacy IPP config keys."""
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    required_legacy_keys = {
        LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY,
        LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY,
    }
    return required_legacy_keys.issubset(configmap_data_keys)


def _legacy_ipp_envoy_filter_exists(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> bool:
    """Return True when maas-controller legacy IPP EnvoyFilter exists (same name as post-processing Deployment)."""
    envoy_filter_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    legacy_ipp_envoy_filter = EnvoyFilter(
        client=admin_client,
        name=envoy_filter_name,
        namespace=gateway_namespace,
        wait_for_resource=False,
    )
    return legacy_ipp_envoy_filter.exists


def _legacy_ipp_deployment_exists(
    admin_client: DynamicClient,
    deployment_name: str,
    gateway_namespace: str,
) -> bool:
    """Return True when a legacy IPP Deployment exists in the gateway namespace."""
    deployment = Deployment(
        client=admin_client,
        name=deployment_name,
        namespace=gateway_namespace,
        wait_for_resource=False,
    )
    return deployment.exists


def _maas_legacy_ipp_configmap_keys_present(configmap_data_keys: set[str]) -> bool:
    """Return True when plugins ConfigMap data includes any maas legacy IPP config key."""
    legacy_marker_keys = {
        LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY,
        LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY,
    }
    return bool(configmap_data_keys & legacy_marker_keys)


def _praxis_ipp_configmap_keys_present(configmap_data_keys: set[str]) -> bool:
    """Return True when plugins ConfigMap data includes any Praxis extproc config key."""
    praxis_marker_keys = {
        PRAXIS_EXTPROC_CONFIG_DATA_KEY,
        PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY,
    }
    return bool(configmap_data_keys & praxis_marker_keys)


def maas_legacy_ipp_markers_present_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> bool:
    """Return True when maas-controller legacy IPP markers exist in the gateway namespace.

    Uses per-tenant plugins ConfigMap data keys that maas-controller sets for legacy IPP and
    ai-gateway-controller does not set for Praxis. Praxis may still have ``payload-processing-{tenant}``
    Deployments from ai-gateway-controller; those are not maas IPP.
    """
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    return _maas_legacy_ipp_configmap_keys_present(configmap_data_keys=configmap_data_keys)


def _describe_maas_legacy_ipp_markers_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> str:
    """Return a short summary of maas legacy IPP markers still present in the gateway namespace."""
    plugins_configmap_name = legacy_ipp_plugins_configmap_name(aitenant_name=aitenant_name)
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    if _maas_legacy_ipp_configmap_keys_present(configmap_data_keys=configmap_data_keys):
        legacy_keys = sorted(
            configmap_data_keys
            & {
                LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY,
                LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY,
            }
        )
        return f"ConfigMap/{plugins_configmap_name} still contains maas legacy IPP keys {legacy_keys}"
    return "no maas legacy IPP markers detected"


def _describe_legacy_ipp_stack_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> str:
    """Return a short summary of per-tenant legacy IPP resources still present in the gateway namespace."""
    present_resources: list[str] = []
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    pre_processing_name = legacy_ipp_pre_processing_deployment_name(aitenant_name=aitenant_name)
    plugins_configmap_name = legacy_ipp_plugins_configmap_name(aitenant_name=aitenant_name)
    if _legacy_ipp_deployment_exists(
        admin_client=admin_client,
        deployment_name=post_processing_name,
        gateway_namespace=gateway_namespace,
    ):
        present_resources.append(f"Deployment/{post_processing_name}")
    if _legacy_ipp_deployment_exists(
        admin_client=admin_client,
        deployment_name=pre_processing_name,
        gateway_namespace=gateway_namespace,
    ):
        present_resources.append(f"Deployment/{pre_processing_name}")
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    if _maas_legacy_ipp_configmap_keys_present(configmap_data_keys=configmap_data_keys):
        legacy_keys = sorted(
            configmap_data_keys
            & {
                LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY,
                LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY,
            }
        )
        present_resources.append(f"ConfigMap/{plugins_configmap_name} keys={legacy_keys}")
    if _praxis_ipp_configmap_keys_present(configmap_data_keys=configmap_data_keys):
        praxis_keys = sorted(
            configmap_data_keys
            & {
                PRAXIS_EXTPROC_CONFIG_DATA_KEY,
                PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY,
            }
        )
        present_resources.append(f"ConfigMap/{plugins_configmap_name} praxis keys={praxis_keys}")
    if _legacy_ipp_envoy_filter_exists(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    ):
        present_resources.append(f"EnvoyFilter/{post_processing_name}")
    if not present_resources:
        return "no legacy IPP marker resources detected"
    return ", ".join(present_resources)


def wait_until_legacy_ipp_absent_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until maas-controller legacy IPP markers are absent from the gateway namespace."""
    try:
        for absent in TimeoutSampler(
            wait_timeout=timeout,
            sleep=LEGACY_IPP_POLL_INTERVAL_SECONDS,
            func=lambda: (
                not maas_legacy_ipp_markers_present_in_gateway_namespace(
                    admin_client=admin_client,
                    gateway_namespace=gateway_namespace,
                    aitenant_name=aitenant_name,
                )
            ),
        ):
            if absent:
                return
    except TimeoutExpiredError:
        remaining_markers = _describe_maas_legacy_ipp_markers_in_gateway_namespace(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        pytest.fail(
            f"Maas legacy IPP markers for AITenant '{aitenant_name}' in gateway namespace "
            f"'{gateway_namespace}' are still present after {timeout}s; found: {remaining_markers}"
        )


def _legacy_ipp_stack_ready_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> bool:
    """Return True when maas-controller legacy IPP stack is present (CM keys, Deployments, legacy container args)."""
    if not _legacy_ipp_plugins_configmap_has_full_maas_config(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    ):
        return False
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    if _praxis_ipp_configmap_keys_present(configmap_data_keys=configmap_data_keys):
        return False
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    pre_processing_name = legacy_ipp_pre_processing_deployment_name(aitenant_name=aitenant_name)
    if not _legacy_ipp_deployment_exists(
        admin_client=admin_client,
        deployment_name=post_processing_name,
        gateway_namespace=gateway_namespace,
    ):
        return False
    if not _legacy_ipp_deployment_exists(
        admin_client=admin_client,
        deployment_name=pre_processing_name,
        gateway_namespace=gateway_namespace,
    ):
        return False
    return (
        _legacy_ipp_envoy_filter_exists(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        and _deployment_container_args_include(
            admin_client=admin_client,
            deployment_name=post_processing_name,
            gateway_namespace=gateway_namespace,
            expected_argument=LEGACY_POST_PROCESSING_CONTAINER_CONFIG_ARG,
        )
        and _deployment_container_args_include(
            admin_client=admin_client,
            deployment_name=pre_processing_name,
            gateway_namespace=gateway_namespace,
            expected_argument=LEGACY_PRE_PROCESSING_CONTAINER_CONFIG_ARG,
        )
        and not _deployment_container_args_include(
            admin_client=admin_client,
            deployment_name=post_processing_name,
            gateway_namespace=gateway_namespace,
            expected_argument=PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG,
        )
        and not _deployment_container_args_include(
            admin_client=admin_client,
            deployment_name=pre_processing_name,
            gateway_namespace=gateway_namespace,
            expected_argument=PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG,
        )
    )


def wait_until_legacy_ipp_present_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until the full maas-controller legacy IPP stack is present in the gateway namespace."""
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    pre_processing_name = legacy_ipp_pre_processing_deployment_name(aitenant_name=aitenant_name)
    plugins_configmap_name = legacy_ipp_plugins_configmap_name(aitenant_name=aitenant_name)

    legacy_stack_signals_ready = False
    try:
        for stack_ready in TimeoutSampler(
            wait_timeout=timeout,
            sleep=LEGACY_IPP_POLL_INTERVAL_SECONDS,
            func=lambda: _legacy_ipp_stack_ready_in_gateway_namespace(
                admin_client=admin_client,
                gateway_namespace=gateway_namespace,
                aitenant_name=aitenant_name,
            ),
        ):
            if stack_ready:
                legacy_stack_signals_ready = True
                break
    except TimeoutExpiredError:
        pass

    if not legacy_stack_signals_ready:
        stack_summary = _describe_legacy_ipp_stack_in_gateway_namespace(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        pytest.fail(
            f"Timed out after {timeout}s waiting for full legacy IPP stack (Deployments "
            f"'{post_processing_name}' and '{pre_processing_name}', EnvoyFilter '{post_processing_name}', "
            f"ConfigMap '{plugins_configmap_name}' with both maas legacy keys) in gateway namespace "
            f"'{gateway_namespace}'; observed: {stack_summary}"
        )

    post_processing_deployment = Deployment(
        client=admin_client,
        name=post_processing_name,
        namespace=gateway_namespace,
        ensure_exists=True,
    )
    pre_processing_deployment = Deployment(
        client=admin_client,
        name=pre_processing_name,
        namespace=gateway_namespace,
        ensure_exists=True,
    )
    try:
        post_processing_deployment.wait_for_condition(condition="Available", status="True", timeout=timeout)
        pre_processing_deployment.wait_for_condition(condition="Available", status="True", timeout=timeout)
    except TimeoutExpiredError:
        pytest.fail(
            f"Timed out after {timeout}s waiting for legacy IPP Deployments "
            f"'{post_processing_name}' and '{pre_processing_name}' to reach Available=True "
            f"in gateway namespace '{gateway_namespace}'"
        )


def verify_legacy_ipp_installed_for_aitenant(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Assert maas-controller installed legacy IPP for an unannotated AITenant in the gateway namespace."""
    wait_until_legacy_ipp_present_in_gateway_namespace(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
        timeout=timeout,
    )


def verify_legacy_ipp_not_installed_for_aitenant(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Assert maas-controller did not leave legacy IPP markers on the tenant plugins ConfigMap."""
    wait_until_legacy_ipp_absent_in_gateway_namespace(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
        timeout=timeout,
    )


def _deployment_container_args_include(
    admin_client: DynamicClient,
    deployment_name: str,
    gateway_namespace: str,
    expected_argument: str,
) -> bool:
    """Return True when a Deployment container args include the Praxis extproc config path."""
    deployment = Deployment(
        client=admin_client,
        name=deployment_name,
        namespace=gateway_namespace,
        wait_for_resource=False,
    )
    if not deployment.exists:
        return False
    deployment_dict = deployment.instance.to_dict()
    pod_spec = ((deployment_dict.get("spec") or {}).get("template") or {}).get("spec") or {}
    containers = pod_spec.get("containers") or []
    for container in containers:
        if not isinstance(container, dict):
            continue
        container_args = container.get("args") or []
        if expected_argument in container_args:
            return True
    return False


def _maas_ipp_handoff_complete(admin_client: DynamicClient, aitenant: AITenant) -> bool:
    """Return True when MaaS released legacy IPP for a praxis-opted-in MaasTenantConfig.

    ``cleanup-complete`` may be consumed by ai-gateway-controller after claim; handoff also
    completes when praxis is set and maas legacy IPP markers are gone from the gateway namespace.
    """
    bootstrapped_tenant_config = MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name_from_aitenant(aitenant=aitenant),
        wait_for_resource=False,
    )
    if not bootstrapped_tenant_config.exists:
        return False
    config_annotations = maastenantconfig_metadata_annotations(bootstrapped_tenant_config=bootstrapped_tenant_config)
    if MAAS_PAYLOAD_PROCESSING_STATUS_ANNOTATION in config_annotations:
        payload_processing_status = config_annotations[MAAS_PAYLOAD_PROCESSING_STATUS_ANNOTATION]
        if payload_processing_status == MAAS_PAYLOAD_PROCESSING_STATUS_CLEANUP_COMPLETE_VALUE:
            return True
    payload_processing_type = read_maastenantconfig_payload_processing_type(
        admin_client=admin_client,
        aitenant=aitenant,
    )
    if payload_processing_type != PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE:
        return False
    gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(aitenant=aitenant)
    legacy_ipp_markers_remain = maas_legacy_ipp_markers_present_in_gateway_namespace(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant.name,
    )
    return not legacy_ipp_markers_remain


def wait_for_maas_ipp_handoff_for_aitenant(
    admin_client: DynamicClient,
    aitenant: AITenant,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until MaaS signals legacy IPP release before ai-gateway applies the Praxis bundle."""
    try:
        for handoff_complete in TimeoutSampler(
            wait_timeout=timeout,
            sleep=LEGACY_IPP_POLL_INTERVAL_SECONDS,
            func=lambda: _maas_ipp_handoff_complete(admin_client=admin_client, aitenant=aitenant),
        ):
            if handoff_complete:
                return
    except TimeoutExpiredError:
        tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=aitenant)
        pytest.fail(
            f"Timed out after {timeout}s waiting for MaaS IPP handoff for AITenant "
            f"'{aitenant.namespace}/{aitenant.name}' (expected MaasTenantConfig/"
            f"{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} annotation "
            f"'{MAAS_PAYLOAD_PROCESSING_STATUS_ANNOTATION}="
            f"{MAAS_PAYLOAD_PROCESSING_STATUS_CLEANUP_COMPLETE_VALUE}' or praxis opt-in with "
            f"legacy IPP markers removed from the gateway namespace in tenant namespace "
            f"'{tenant_namespace_name}')"
        )


def _praxis_ipp_plugins_configmap_ready(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> bool:
    """Return True when the tenant plugins ConfigMap carries the Praxis extproc config keys."""
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    if _maas_legacy_ipp_configmap_keys_present(configmap_data_keys=configmap_data_keys):
        return False
    required_praxis_keys = {PRAXIS_EXTPROC_CONFIG_DATA_KEY, PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY}
    return required_praxis_keys.issubset(configmap_data_keys)


def _praxis_ipp_bundle_ready_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> bool:
    """Return True when ai-gateway-controller Praxis IPP resources are present in the gateway namespace."""
    if not _praxis_ipp_plugins_configmap_ready(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    ):
        return False
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    pre_processing_name = legacy_ipp_pre_processing_deployment_name(aitenant_name=aitenant_name)
    return (
        _legacy_ipp_envoy_filter_exists(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        and _deployment_container_args_include(
            admin_client=admin_client,
            deployment_name=post_processing_name,
            gateway_namespace=gateway_namespace,
            expected_argument=PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG,
        )
        and _deployment_container_args_include(
            admin_client=admin_client,
            deployment_name=pre_processing_name,
            gateway_namespace=gateway_namespace,
            expected_argument=PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG,
        )
    )


def _describe_praxis_ipp_bundle_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> str:
    """Return a short summary of Praxis IPP bundle state in the gateway namespace."""
    plugins_configmap_name = legacy_ipp_plugins_configmap_name(aitenant_name=aitenant_name)
    configmap_data_keys = sorted(
        _legacy_ipp_plugins_configmap_data_keys(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
    )
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    pre_processing_name = legacy_ipp_pre_processing_deployment_name(aitenant_name=aitenant_name)
    post_deployment_has_praxis_args = _deployment_container_args_include(
        admin_client=admin_client,
        deployment_name=post_processing_name,
        gateway_namespace=gateway_namespace,
        expected_argument=PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG,
    )
    pre_deployment_has_praxis_args = _deployment_container_args_include(
        admin_client=admin_client,
        deployment_name=pre_processing_name,
        gateway_namespace=gateway_namespace,
        expected_argument=PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG,
    )
    envoy_filter_present = _legacy_ipp_envoy_filter_exists(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    return (
        f"ConfigMap/{plugins_configmap_name} keys={configmap_data_keys!r}; "
        f"Deployment/{post_processing_name} praxis_args={post_deployment_has_praxis_args}; "
        f"Deployment/{pre_processing_name} praxis_args={pre_deployment_has_praxis_args}; "
        f"EnvoyFilter/{post_processing_name} present={envoy_filter_present}"
    )


def wait_until_praxis_ipp_bundle_ready_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until ai-gateway-controller installed the Praxis IPP bundle in the gateway namespace."""
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    pre_processing_name = legacy_ipp_pre_processing_deployment_name(aitenant_name=aitenant_name)
    praxis_bundle_signals_ready = False
    try:
        for bundle_ready in TimeoutSampler(
            wait_timeout=timeout,
            sleep=LEGACY_IPP_POLL_INTERVAL_SECONDS,
            func=lambda: _praxis_ipp_bundle_ready_in_gateway_namespace(
                admin_client=admin_client,
                gateway_namespace=gateway_namespace,
                aitenant_name=aitenant_name,
            ),
        ):
            if bundle_ready:
                praxis_bundle_signals_ready = True
                break
    except TimeoutExpiredError:
        pass

    if not praxis_bundle_signals_ready:
        bundle_summary = _describe_praxis_ipp_bundle_in_gateway_namespace(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        pytest.fail(
            f"Timed out after {timeout}s waiting for Praxis IPP bundle for AITenant '{aitenant_name}' "
            f"in gateway namespace '{gateway_namespace}'; observed: {bundle_summary}"
        )

    post_processing_deployment = Deployment(
        client=admin_client,
        name=post_processing_name,
        namespace=gateway_namespace,
        ensure_exists=True,
    )
    pre_processing_deployment = Deployment(
        client=admin_client,
        name=pre_processing_name,
        namespace=gateway_namespace,
        ensure_exists=True,
    )
    try:
        post_processing_deployment.wait_for_condition(
            condition="Available",
            status="True",
            timeout=timeout,
        )
        pre_processing_deployment.wait_for_condition(
            condition="Available",
            status="True",
            timeout=timeout,
        )
    except TimeoutExpiredError:
        pytest.fail(
            f"Timed out after {timeout}s waiting for Praxis IPP Deployments "
            f"'{post_processing_name}' and '{pre_processing_name}' to reach Available=True "
            f"in gateway namespace '{gateway_namespace}'"
        )


def verify_praxis_ipp_bundle_installed_for_aitenant(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Assert ai-gateway-controller installed the Praxis extproc IPP bundle in the gateway namespace."""
    wait_until_praxis_ipp_bundle_ready_in_gateway_namespace(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
        timeout=timeout,
    )


def _praxis_ipp_bundle_absent_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> bool:
    """Return True when no Praxis extproc CM keys or Praxis container args remain on tenant IPP workloads."""
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    pre_processing_name = legacy_ipp_pre_processing_deployment_name(aitenant_name=aitenant_name)
    praxis_signals_remain = (
        _praxis_ipp_configmap_keys_present(configmap_data_keys=configmap_data_keys)
        or _deployment_container_args_include(
            admin_client=admin_client,
            deployment_name=post_processing_name,
            gateway_namespace=gateway_namespace,
            expected_argument=PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG,
        )
        or _deployment_container_args_include(
            admin_client=admin_client,
            deployment_name=pre_processing_name,
            gateway_namespace=gateway_namespace,
            expected_argument=PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG,
        )
    )
    return not praxis_signals_remain


def _describe_praxis_ipp_bundle_absence_blockers_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> str:
    """Return a short summary of Praxis IPP signals still present while waiting for teardown."""
    blockers: list[str] = []
    plugins_configmap_name = legacy_ipp_plugins_configmap_name(aitenant_name=aitenant_name)
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    praxis_keys = sorted(
        configmap_data_keys
        & {
            PRAXIS_EXTPROC_CONFIG_DATA_KEY,
            PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY,
        }
    )
    if praxis_keys:
        blockers.append(f"ConfigMap/{plugins_configmap_name} still has Praxis keys {praxis_keys}")
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    pre_processing_name = legacy_ipp_pre_processing_deployment_name(aitenant_name=aitenant_name)
    if _deployment_container_args_include(
        admin_client=admin_client,
        deployment_name=post_processing_name,
        gateway_namespace=gateway_namespace,
        expected_argument=PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG,
    ):
        blockers.append(f"Deployment/{post_processing_name} still uses Praxis extproc container args")
    if _deployment_container_args_include(
        admin_client=admin_client,
        deployment_name=pre_processing_name,
        gateway_namespace=gateway_namespace,
        expected_argument=PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG,
    ):
        blockers.append(f"Deployment/{pre_processing_name} still uses Praxis pre-extproc container args")
    if not blockers:
        return "no Praxis IPP teardown blockers detected"
    return "; ".join(blockers)


def wait_until_praxis_ipp_bundle_absent_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until Praxis extproc ConfigMap keys and container args are gone from the gateway namespace."""
    try:
        for bundle_absent in TimeoutSampler(
            wait_timeout=timeout,
            sleep=LEGACY_IPP_POLL_INTERVAL_SECONDS,
            func=lambda: _praxis_ipp_bundle_absent_in_gateway_namespace(
                admin_client=admin_client,
                gateway_namespace=gateway_namespace,
                aitenant_name=aitenant_name,
            ),
        ):
            if bundle_absent:
                return
    except TimeoutExpiredError:
        absence_blockers = _describe_praxis_ipp_bundle_absence_blockers_in_gateway_namespace(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        pytest.fail(
            f"Praxis IPP bundle for AITenant '{aitenant_name}' in gateway namespace "
            f"'{gateway_namespace}' is still present after {timeout}s; observed: {absence_blockers}"
        )


def verify_praxis_payload_processing_active_for_aitenant(
    admin_client: DynamicClient,
    aitenant: AITenant,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Assert MaaS released legacy IPP and ai-gateway installed the Praxis bundle in the gateway namespace."""
    gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(aitenant=aitenant)
    aitenant_name = aitenant.name
    wait_for_maas_ipp_handoff_for_aitenant(admin_client=admin_client, aitenant=aitenant, timeout=timeout)
    verify_legacy_ipp_not_installed_for_aitenant(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
        timeout=timeout,
    )
    verify_praxis_ipp_bundle_installed_for_aitenant(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
        timeout=timeout,
    )


def migrate_legacy_aitenant_to_praxis_payload_processing(
    admin_client: DynamicClient,
    aitenant: AITenant,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Migrate a legacy tenant to Praxis via MaasTenantConfig and wait for Praxis payload processing.

    Verifies legacy IPP is installed before migration, then applies the praxis payload-processing-type
    annotation on MaasTenantConfig and waits for the praxis-cleanup finalizer and active Praxis bundle.
    """
    gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(aitenant=aitenant)
    aitenant_name = aitenant.name
    verify_legacy_ipp_installed_for_aitenant(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
        timeout=timeout,
    )
    set_maastenantconfig_payload_processing_type_annotation(
        admin_client=admin_client,
        aitenant=aitenant,
        annotation_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
    )
    verify_maastenantconfig_has_praxis_cleanup_finalizer(
        admin_client=admin_client,
        aitenant=aitenant,
        timeout=timeout,
    )
    verify_praxis_payload_processing_active_for_aitenant(
        admin_client=admin_client,
        aitenant=aitenant,
        timeout=timeout,
    )


def restore_legacy_aitenant_payload_processing(
    admin_client: DynamicClient,
    aitenant: AITenant,
    timeout: int = LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Remove praxis opt-in from MaasTenantConfig and wait until maas-controller legacy IPP is active again."""
    gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(aitenant=aitenant)
    aitenant_name = aitenant.name
    set_maastenantconfig_payload_processing_type_annotation(
        admin_client=admin_client,
        aitenant=aitenant,
        annotation_value=None,
    )
    verify_maastenantconfig_payload_processing_type(
        admin_client=admin_client,
        aitenant=aitenant,
        expected_value=None,
    )
    wait_until_maastenantconfig_lacks_praxis_cleanup_finalizer(
        admin_client=admin_client,
        aitenant=aitenant,
        timeout=timeout,
    )
    wait_until_praxis_ipp_bundle_absent_in_gateway_namespace(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
        timeout=timeout,
    )
    verify_legacy_ipp_installed_for_aitenant(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
        timeout=timeout,
    )


def verify_praxis_maas_tenant_config_ready(aitenant: AITenant, admin_client: DynamicClient) -> None:
    """Assert MaasTenantConfig is Ready without degraded EnvoyFilter / legacy IPP dependency errors."""
    bootstrapped_tenant_config = maas_tenant_config_for_aitenant(admin_client=admin_client, aitenant=aitenant)
    verify_maas_tenant_config_ready(maas_tenant_config=bootstrapped_tenant_config)
    status_conditions = getattr(bootstrapped_tenant_config.instance.status, "conditions", None) or []
    for condition in status_conditions:
        condition_type = getattr(condition, "type", None)
        condition_status = getattr(condition, "status", None)
        if condition_type == "Degraded" and condition_status == "True":
            condition_message = getattr(condition, "message", "") or ""
            assert "EnvoyFilter" not in condition_message, (
                f"MaasTenantConfig '{bootstrapped_tenant_config.namespace}/{bootstrapped_tenant_config.name}' "
                f"is Degraded with unexpected legacy IPP EnvoyFilter error: {condition_message}"
            )


def gateway_namespace_and_name_for_aitenant(aitenant: AITenant) -> tuple[str, str]:
    """Return gateway namespace and name from AITenant status.gatewayRef."""
    gateway_name, gateway_namespace = gateway_ref_from_aitenant(aitenant=aitenant)
    return gateway_namespace, gateway_name
