from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.praxis.utils import (
    opt_in_legacy_ipp_payload_processing_for_aitenant,
    praxis_aitenant_with_bootstrap_gateway,
    verify_default_dataplane_praxis_for_aitenant,
)
from tests.ai_gateway.models_as_a_service.utils import deploy_and_verify_aitenant_ready
from utilities.resources.aitenant import AITenant


@pytest.fixture
def ready_aitenant_default_dataplane(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy a Ready AITenant whose MaasTenantConfig uses default Praxis payload processing."""
    with praxis_aitenant_with_bootstrap_gateway(
        admin_client=admin_client,
        cr_namespace=aitenant_infra_namespace,
        teardown=teardown_resources,
    ) as aitenant:
        deploy_and_verify_aitenant_ready(aitenant=aitenant)
        verify_default_dataplane_praxis_for_aitenant(admin_client=admin_client, aitenant=aitenant)
        yield aitenant


@pytest.fixture
def ready_praxis_annotated_aitenant(ready_aitenant_default_dataplane: AITenant) -> AITenant:
    """Alias for a Ready tenant on default Praxis (post MaaS #1579)."""
    return ready_aitenant_default_dataplane


@pytest.fixture
def ready_aitenant_legacy_ipp(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy a Ready AITenant with legacy IPP opted in via MaasTenantConfig."""
    with praxis_aitenant_with_bootstrap_gateway(
        admin_client=admin_client,
        cr_namespace=aitenant_infra_namespace,
        teardown=teardown_resources,
    ) as aitenant:
        deploy_and_verify_aitenant_ready(aitenant=aitenant)
        opt_in_legacy_ipp_payload_processing_for_aitenant(admin_client=admin_client, aitenant=aitenant)
        yield aitenant
