"""Helpers for Praxis AITenant annotation and maas-controller platform tests."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
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
    MAAS_IPP_MIGRATION_CLEANUP_COMPLETE_ANNOTATION,
    MAAS_IPP_RESOURCES_RELEASED_CONDITION,
    PRAXIS_AITENANT_CLEANUP_FINALIZER,
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
    fresh_aitenant,
    verify_aitenant_ready,
    verify_maas_tenant_config_ready,
)
from utilities.general import generate_random_name
from utilities.resources.aitenant import AITenant
from utilities.resources.envoy_filter import EnvoyFilter
from utilities.resources.maastenantconfig import MaasTenantConfig


def payload_processing_type_annotations(annotation_value: str | None) -> dict[str, str] | None:
    """Return AITenant metadata annotations for payload-processing-type, or None to omit the key."""
    if annotation_value is None:
        return None
    return {PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION: annotation_value}


def praxis_aitenant_from_spec(
    admin_client: DynamicClient,
    aitenant_name: str,
    cr_namespace: str,
    aitenant_spec: dict[str, Any],
    payload_processing_type: str | None = None,
    teardown: bool = False,
) -> AITenant:
    """Return an AITenant configured from spec with optional payload-processing-type annotation."""
    return aitenant_from_spec(
        admin_client=admin_client,
        aitenant_name=aitenant_name,
        cr_namespace=cr_namespace,
        aitenant_spec=aitenant_spec,
        teardown=teardown,
        annotations=payload_processing_type_annotations(annotation_value=payload_processing_type),
    )


def read_payload_processing_type_annotation(aitenant: AITenant) -> str | None:
    """Return the payload-processing-type annotation value, or None when absent."""
    metadata_annotations = dict(fresh_aitenant(aitenant=aitenant).instance.metadata.annotations or {})
    if PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION not in metadata_annotations:
        return None
    return str(metadata_annotations[PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION])


def verify_aitenant_payload_processing_annotation(
    aitenant: AITenant,
    expected_value: str | None,
) -> None:
    """Assert the AITenant carries the expected payload-processing-type annotation value."""
    actual_value = read_payload_processing_type_annotation(aitenant=aitenant)
    assert actual_value == expected_value, (
        f"AITenant '{aitenant.namespace}/{aitenant.name}' annotation "
        f"'{PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION}' expected {expected_value!r}, got {actual_value!r}"
    )


def read_aitenant_finalizers(aitenant: AITenant) -> list[str]:
    """Return finalizer names on the AITenant, or an empty list when none are set."""
    metadata_finalizers = getattr(fresh_aitenant(aitenant=aitenant).instance.metadata, "finalizers", None)
    if not metadata_finalizers:
        return []
    return [str(finalizer) for finalizer in metadata_finalizers]


def verify_aitenant_has_praxis_cleanup_finalizer(
    aitenant: AITenant,
    timeout: int = DEFAULT_PRAXIS_FINALIZER_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Assert ai-gateway-controller attached the praxis cleanup finalizer (effective praxis opt-in)."""
    try:
        for has_finalizer in TimeoutSampler(
            wait_timeout=timeout,
            sleep=2,
            func=lambda: PRAXIS_AITENANT_CLEANUP_FINALIZER in read_aitenant_finalizers(aitenant=aitenant),
        ):
            if has_finalizer:
                return
    except TimeoutExpiredError:
        finalizers = read_aitenant_finalizers(aitenant=aitenant)
        pytest.fail(
            f"AITenant '{aitenant.namespace}/{aitenant.name}' should have finalizer "
            f"'{PRAXIS_AITENANT_CLEANUP_FINALIZER}' when payload-processing-type is praxis "
            f"(timeout {timeout}s); got {finalizers!r}"
        )


def wait_until_aitenant_lacks_praxis_cleanup_finalizer(
    aitenant: AITenant,
    timeout: int = DEFAULT_PRAXIS_FINALIZER_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until the praxis cleanup finalizer is removed (legacy / de-annotated tenant)."""
    try:
        for lacks_finalizer in TimeoutSampler(
            wait_timeout=timeout,
            sleep=2,
            func=lambda: PRAXIS_AITENANT_CLEANUP_FINALIZER not in read_aitenant_finalizers(aitenant=aitenant),
        ):
            if lacks_finalizer:
                return
    except TimeoutExpiredError:
        finalizers = read_aitenant_finalizers(aitenant=aitenant)
        pytest.fail(
            f"AITenant '{aitenant.namespace}/{aitenant.name}' should not have finalizer "
            f"'{PRAXIS_AITENANT_CLEANUP_FINALIZER}' after leaving praxis opt-in "
            f"(timeout {timeout}s); got {finalizers!r}"
        )


def verify_aitenant_lacks_praxis_cleanup_finalizer(aitenant: AITenant) -> None:
    """Assert the AITenant is not on the Praxis controller cleanup path (legacy / non-praxis annotation)."""
    finalizers = read_aitenant_finalizers(aitenant=aitenant)
    assert PRAXIS_AITENANT_CLEANUP_FINALIZER not in finalizers, (
        f"AITenant '{aitenant.namespace}/{aitenant.name}' should not have finalizer "
        f"'{PRAXIS_AITENANT_CLEANUP_FINALIZER}' without effective praxis opt-in; got {finalizers!r}"
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
    payload_processing_type: str | None,
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
            payload_processing_type=payload_processing_type,
            teardown=teardown,
        ) as aitenant,
    ):
        yield aitenant


def deploy_praxis_aitenant_and_verify_annotation(
    aitenant: AITenant,
    expected_annotation_value: str,
) -> None:
    """Create the AITenant if missing and assert the praxis annotation is persisted."""
    if not aitenant.exists:
        aitenant.deploy()
    assert aitenant.exists, f"AITenant '{aitenant.namespace}/{aitenant.name}' was not created"
    verify_aitenant_payload_processing_annotation(aitenant=aitenant, expected_value=expected_annotation_value)


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


def _legacy_ipp_plugins_configmap_has_custom_config(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
) -> bool:
    """Return True when the per-tenant legacy IPP plugins ConfigMap exists with custom-ipp-config.yaml."""
    configmap_data_keys = _legacy_ipp_plugins_configmap_data_keys(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    return LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY in configmap_data_keys


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
    """Return True when plugins ConfigMap data includes maas legacy IPP config keys."""
    legacy_marker_keys = {
        LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY,
        LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY,
    }
    return bool(configmap_data_keys & legacy_marker_keys)


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
    if _legacy_ipp_plugins_configmap_has_custom_config(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    ):
        present_resources.append(f"ConfigMap/{plugins_configmap_name}")
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


def wait_until_legacy_ipp_present_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until per-tenant legacy IPP post-processing and plugins ConfigMap are present."""
    post_processing_name = legacy_ipp_post_processing_deployment_name(aitenant_name=aitenant_name)
    plugins_configmap_name = legacy_ipp_plugins_configmap_name(aitenant_name=aitenant_name)

    def legacy_ipp_ready() -> bool:
        post_processing_deployment = Deployment(
            client=admin_client,
            name=post_processing_name,
            namespace=gateway_namespace,
            wait_for_resource=False,
        )
        if not post_processing_deployment.exists:
            return False
        if not _legacy_ipp_plugins_configmap_has_custom_config(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        ):
            return False
        return _legacy_ipp_envoy_filter_exists(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )

    try:
        for ready in TimeoutSampler(
            wait_timeout=timeout,
            sleep=LEGACY_IPP_POLL_INTERVAL_SECONDS,
            func=legacy_ipp_ready,
        ):
            if ready:
                post_processing_deployment = Deployment(
                    client=admin_client,
                    name=post_processing_name,
                    namespace=gateway_namespace,
                    ensure_exists=True,
                )
                post_processing_deployment.wait_for_condition(condition="Available", status="True", timeout=timeout)
                return
    except TimeoutExpiredError:
        stack_summary = _describe_legacy_ipp_stack_in_gateway_namespace(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        pytest.fail(
            f"Timed out after {timeout}s waiting for legacy IPP Deployment '{post_processing_name}', "
            f"EnvoyFilter '{post_processing_name}', and ConfigMap '{plugins_configmap_name}' in gateway "
            f"namespace '{gateway_namespace}'; observed: {stack_summary}"
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
    """Return True when maas-controller released legacy IPP (ai-gateway #9 handoff gate)."""
    aitenant_annotations = dict(fresh_aitenant(aitenant=aitenant).instance.metadata.annotations or {})
    if aitenant_annotations.get(MAAS_IPP_MIGRATION_CLEANUP_COMPLETE_ANNOTATION) == "true":
        return True
    tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=aitenant)
    maas_tenant_config = MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        wait_for_resource=False,
    )
    if not maas_tenant_config.exists:
        return False
    maas_tenant_config_dict = maas_tenant_config.instance.to_dict()
    config_annotations = dict((maas_tenant_config_dict.get("metadata") or {}).get("annotations") or {})
    if config_annotations.get(MAAS_IPP_MIGRATION_CLEANUP_COMPLETE_ANNOTATION) == "true":
        return True
    status_conditions = (maas_tenant_config_dict.get("status") or {}).get("conditions") or []
    for condition in status_conditions:
        if not isinstance(condition, dict):
            continue
        if condition.get("type") == MAAS_IPP_RESOURCES_RELEASED_CONDITION and condition.get("status") == "True":
            return True
    return False


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
            f"'{aitenant.namespace}/{aitenant.name}' (expected "
            f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} condition "
            f"'{MAAS_IPP_RESOURCES_RELEASED_CONDITION}=True' or annotation "
            f"'{MAAS_IPP_MIGRATION_CLEANUP_COMPLETE_ANNOTATION}=true' in tenant namespace "
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
                return
    except TimeoutExpiredError:
        bundle_summary = _describe_praxis_ipp_bundle_in_gateway_namespace(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        pytest.fail(
            f"Timed out after {timeout}s waiting for Praxis IPP bundle for AITenant '{aitenant_name}' "
            f"in gateway namespace '{gateway_namespace}'; observed: {bundle_summary}"
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


def wait_until_praxis_ipp_bundle_absent_in_gateway_namespace(
    admin_client: DynamicClient,
    gateway_namespace: str,
    aitenant_name: str,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Poll until the Praxis IPP bundle is no longer present in the gateway namespace."""
    try:
        for bundle_absent in TimeoutSampler(
            wait_timeout=timeout,
            sleep=LEGACY_IPP_POLL_INTERVAL_SECONDS,
            func=lambda: (
                not _praxis_ipp_bundle_ready_in_gateway_namespace(
                    admin_client=admin_client,
                    gateway_namespace=gateway_namespace,
                    aitenant_name=aitenant_name,
                )
            ),
        ):
            if bundle_absent:
                return
    except TimeoutExpiredError:
        bundle_summary = _describe_praxis_ipp_bundle_in_gateway_namespace(
            admin_client=admin_client,
            gateway_namespace=gateway_namespace,
            aitenant_name=aitenant_name,
        )
        pytest.fail(
            f"Praxis IPP bundle for AITenant '{aitenant_name}' in gateway namespace "
            f"'{gateway_namespace}' is still present after {timeout}s; observed: {bundle_summary}"
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


def set_aitenant_payload_processing_type_annotation(
    aitenant: AITenant,
    annotation_value: str | None,
) -> None:
    """Set or remove the payload-processing-type annotation on an existing AITenant."""
    refreshed_aitenant = fresh_aitenant(aitenant=aitenant)
    resource_dict = refreshed_aitenant.instance.to_dict()
    metadata = resource_dict.setdefault("metadata", {})
    annotations = dict(metadata["annotations"]) if metadata.get("annotations") else {}
    if annotation_value is None:
        if PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION in annotations:
            del annotations[PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION]
    else:
        annotations[PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION] = annotation_value
    metadata["annotations"] = annotations
    refreshed_aitenant.update(resource_dict=resource_dict)


def migrate_legacy_aitenant_to_praxis_payload_processing(
    admin_client: DynamicClient,
    aitenant: AITenant,
    timeout: int = DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS,
) -> None:
    """Migrate a legacy tenant to Praxis by annotating the AITenant and waiting for Praxis payload processing.

    Verifies legacy IPP is installed before migration, then applies the praxis payload-processing-type
    annotation and waits for the praxis-cleanup finalizer and active Praxis bundle in the gateway namespace.
    """
    gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(aitenant=aitenant)
    aitenant_name = aitenant.name
    verify_legacy_ipp_installed_for_aitenant(
        admin_client=admin_client,
        gateway_namespace=gateway_namespace,
        aitenant_name=aitenant_name,
    )
    set_aitenant_payload_processing_type_annotation(
        aitenant=aitenant,
        annotation_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
    )
    verify_aitenant_has_praxis_cleanup_finalizer(aitenant=aitenant)
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
    """Remove praxis opt-in from an AITenant and wait until maas-controller legacy IPP is active again."""
    gateway_namespace, _gateway_name = gateway_namespace_and_name_for_aitenant(aitenant=aitenant)
    aitenant_name = aitenant.name
    set_aitenant_payload_processing_type_annotation(aitenant=aitenant, annotation_value=None)
    wait_until_aitenant_lacks_praxis_cleanup_finalizer(aitenant=aitenant, timeout=timeout)
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


def maas_tenant_config_for_aitenant(admin_client: DynamicClient, aitenant: AITenant) -> MaasTenantConfig:
    """Return the bootstrapped MaasTenantConfig for a Ready AITenant."""
    tenant_namespace_name = tenant_namespace_name_from_aitenant(aitenant=aitenant)
    bootstrapped_tenant_config = MaasTenantConfig(
        client=admin_client,
        name=AIGATEWAY_BOOTSTRAPPED_TENANT_NAME,
        namespace=tenant_namespace_name,
        ensure_exists=True,
    )
    assert bootstrapped_tenant_config.exists, (
        f"MaasTenantConfig/{AIGATEWAY_BOOTSTRAPPED_TENANT_NAME} not found in '{tenant_namespace_name}'"
    )
    return bootstrapped_tenant_config


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
