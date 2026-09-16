"""Helpers for Praxis AITenant annotation contract tests."""

from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from timeout_sampler import TimeoutSampler

from tests.ai_gateway.models_as_a_service.praxis.constants import (
    PRAXIS_AITENANT_CLEANUP_FINALIZER,
    PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION,
)
from tests.ai_gateway.models_as_a_service.utils import (
    aitenant_from_spec,
    bootstrap_gateway_context,
    bootstrap_gateway_ref,
    build_aitenant_spec,
    fresh_aitenant,
    verify_aitenant_ready,
)
from utilities.general import generate_random_name
from utilities.resources.aitenant import AITenant


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
    raw_value = metadata_annotations.get(PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION)
    if raw_value is None:
        return None
    return str(raw_value)


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


def verify_aitenant_has_praxis_cleanup_finalizer(aitenant: AITenant, timeout: int = 120) -> None:
    """Assert ai-gateway-controller attached the praxis cleanup finalizer (effective praxis opt-in)."""
    for has_finalizer in TimeoutSampler(
        wait_timeout=timeout,
        sleep=2,
        func=lambda: PRAXIS_AITENANT_CLEANUP_FINALIZER in read_aitenant_finalizers(aitenant=aitenant),
    ):
        if has_finalizer:
            return
    finalizers = read_aitenant_finalizers(aitenant=aitenant)
    pytest.fail(
        f"AITenant '{aitenant.namespace}/{aitenant.name}' should have finalizer "
        f"'{PRAXIS_AITENANT_CLEANUP_FINALIZER}' when payload-processing-type is praxis "
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
) -> Generator[AITenant, None, None]:
    """Yield an AITenant after its bootstrap Gateway exists."""
    resolved_aitenant_name = aitenant_name or f"e2e-praxis-{generate_random_name()}"
    aitenant_spec = build_aitenant_spec(aitenant_name=resolved_aitenant_name)
    gateway_name, gateway_namespace = bootstrap_gateway_ref(
        aitenant_name=resolved_aitenant_name,
        aitenant_spec=aitenant_spec,
    )
    with bootstrap_gateway_context(
        admin_client=admin_client,
        gateway_name=gateway_name,
        gateway_namespace=gateway_namespace,
        teardown=teardown,
    ):
        with praxis_aitenant_from_spec(
            admin_client=admin_client,
            aitenant_name=resolved_aitenant_name,
            cr_namespace=cr_namespace,
            aitenant_spec=aitenant_spec,
            payload_processing_type=payload_processing_type,
            teardown=teardown,
        ) as aitenant:
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
