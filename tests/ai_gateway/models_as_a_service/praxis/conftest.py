from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.praxis.constants import PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE
from tests.ai_gateway.models_as_a_service.praxis.utils import praxis_aitenant_with_bootstrap_gateway
from tests.ai_gateway.models_as_a_service.utils import deploy_and_verify_aitenant_ready
from utilities.resources.aitenant import AITenant


@pytest.fixture
def ready_praxis_annotated_aitenant(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy a Ready AITenant with the praxis payload-processing-type annotation."""
    with praxis_aitenant_with_bootstrap_gateway(
        admin_client=admin_client,
        cr_namespace=aitenant_infra_namespace,
        payload_processing_type=PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE,
        teardown=teardown_resources,
    ) as aitenant:
        deploy_and_verify_aitenant_ready(aitenant=aitenant)
        yield aitenant


@pytest.fixture
def ready_aitenant_without_praxis_annotation(
    admin_client: DynamicClient,
    aitenant_infra_namespace: str,
    teardown_resources: bool,
) -> Generator[AITenant, Any, Any]:
    """Deploy a Ready AITenant without a payload-processing-type annotation."""
    with praxis_aitenant_with_bootstrap_gateway(
        admin_client=admin_client,
        cr_namespace=aitenant_infra_namespace,
        payload_processing_type=None,
        teardown=teardown_resources,
    ) as aitenant:
        deploy_and_verify_aitenant_ready(aitenant=aitenant)
        yield aitenant
