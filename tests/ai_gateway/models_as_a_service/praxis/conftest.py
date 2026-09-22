from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.praxis.constants import PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE
from tests.ai_gateway.models_as_a_service.praxis.utils import (
    deploy_aitenant_with_maastenantconfig_praxis_opt_in_before_legacy_ipp,
    praxis_aitenant_with_bootstrap_gateway,
    verify_maastenantconfig_has_praxis_cleanup_finalizer,
    verify_maastenantconfig_lacks_praxis_cleanup_finalizer,
    verify_maastenantconfig_payload_processing_type,
)
from tests.ai_gateway.models_as_a_service.utils import deploy_and_verify_aitenant_ready
from utilities.resources.aitenant import AITenant


@pytest.fixture
def ready_praxis_annotated_aitenant(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy a Ready AITenant with praxis opt-in on MaasTenantConfig."""
    with praxis_aitenant_with_bootstrap_gateway(
        admin_client=admin_client,
        cr_namespace=aitenant_infra_namespace,
        teardown=teardown_resources,
    ) as aitenant:
        deploy_aitenant_with_maastenantconfig_praxis_opt_in_before_legacy_ipp(
            admin_client=admin_client,
            aitenant=aitenant,
            annotation_value=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
        )
        verify_maastenantconfig_has_praxis_cleanup_finalizer(admin_client=admin_client, aitenant=aitenant)
        yield aitenant


@pytest.fixture
def ready_aitenant_without_praxis_annotation(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy a Ready AITenant without praxis opt-in on MaasTenantConfig."""
    with praxis_aitenant_with_bootstrap_gateway(
        admin_client=admin_client,
        cr_namespace=aitenant_infra_namespace,
        teardown=teardown_resources,
    ) as aitenant:
        deploy_and_verify_aitenant_ready(aitenant=aitenant)
        verify_maastenantconfig_payload_processing_type(
            admin_client=admin_client,
            aitenant=aitenant,
            expected_value=None,
        )
        verify_maastenantconfig_lacks_praxis_cleanup_finalizer(admin_client=admin_client, aitenant=aitenant)
        yield aitenant
