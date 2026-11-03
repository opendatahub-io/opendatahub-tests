"""Fixtures for multi-tenant usage-log E2E (RHOAIENG-99404)."""

from collections.abc import Generator
from contextlib import ExitStack

import pytest
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import AITenantTestContext
from tests.ai_gateway.models_as_a_service.observability.constants import USAGE_LOGS_INGESTION_WAIT_TIMEOUT
from tests.ai_gateway.models_as_a_service.observability.usage_logs_query_utils import (
    maastenantconfig_for_namespace,
    patch_maastenantconfig_usage_telemetry,
    usage_logs_e2e_stack_ready,
)


@pytest.fixture(scope="session")
def usage_logs_multi_tenant_e2e_monitoring_namespace(admin_client: DynamicClient) -> str:
    """Skip when Loki gateway or usage-logs tenancy proxy is not ready on the cluster."""
    ready, detail = usage_logs_e2e_stack_ready(admin_client=admin_client)
    if not ready:
        pytest.skip(detail)
    return detail


@pytest.fixture(scope="class")
def multitenancy_usage_logs_telemetry_enabled(
    admin_client: DynamicClient,
    two_aitenant_test_contexts: tuple[AITenantTestContext, AITenantTestContext],
) -> Generator[None]:
    """Enable telemetry and logs.captureUser on both isolation AITenant MaasTenantConfigs."""
    with ExitStack() as stack:
        for test_context in two_aitenant_test_contexts:
            tenant_config = maastenantconfig_for_namespace(
                admin_client=admin_client,
                tenant_namespace=test_context["tenant_namespace_name"],
            )
            stack.enter_context(patch_maastenantconfig_usage_telemetry(tenant_config))
        yield


@pytest.fixture(scope="class")
def usage_logs_ingestion_wait_timeout() -> int:
    """Return the Loki ingestion poll timeout for usage-log assertions."""
    return USAGE_LOGS_INGESTION_WAIT_TIMEOUT
