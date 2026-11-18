"""Fixtures for multi-tenant usage-log E2E (RHOAIENG-99404)."""

from collections.abc import Generator
from contextlib import ExitStack

import pytest
from _pytest.config import Config
from kubernetes.dynamic import DynamicClient

from tests.ai_gateway.models_as_a_service.multitenancy.aitenant.utils import AITenantTestContext
from tests.ai_gateway.models_as_a_service.multitenancy.usage_logs.utils import (
    maastenantconfig_for_namespace,
    patch_maastenantconfig_usage_telemetry,
    verify_usage_logs_multi_tenant_e2e_prerequisites,
)
from tests.ai_gateway.models_as_a_service.observability.constants import USAGE_LOGS_PROPAGATION_TIMEOUT
from utilities.constants import ModelStorage


@pytest.fixture(scope="session", autouse=True)
def _usage_logs_oci_multitenancy_pytest_defaults(pytestconfig: Config) -> None:
    """Defaults so this package can run OCI TinyLlama without shadowing root session AWS fixtures."""
    if not pytestconfig.option.aws_access_key_id:
        pytestconfig.option.aws_access_key_id = "unused-oci-tenant-model"
    if not pytestconfig.option.aws_secret_access_key:
        pytestconfig.option.aws_secret_access_key = "unused-oci-tenant-model"  # pragma: allowlist secret
    if not pytestconfig.option.models_s3_bucket_name:
        pytestconfig.option.models_s3_bucket_name = "unused"
    if not pytestconfig.option.models_s3_bucket_region:
        pytestconfig.option.models_s3_bucket_region = ModelStorage.S3.ODS_CI_WISDOM_REGION
    if not pytestconfig.option.models_s3_bucket_endpoint:
        pytestconfig.option.models_s3_bucket_endpoint = ModelStorage.S3.ODS_CI_WISDOM_ENDPOINT


@pytest.fixture(scope="class")
def tenant_isolation_model_storage_uri() -> str:  # noqa: UFN001
    """Override parent fixture: OCI TinyLlama, no ods-ci-wisdom credentials."""
    return ModelStorage.OCI.TINYLLAMA


@pytest.fixture(scope="session")
def usage_logs_multi_tenant_e2e_monitoring_namespace(admin_client: DynamicClient) -> str:
    """Fail fast when pipeline prerequisites for usage-log multitenancy E2E are missing."""
    monitoring_namespace, failure = verify_usage_logs_multi_tenant_e2e_prerequisites(admin_client=admin_client)
    if failure:
        pytest.fail(failure)
    assert monitoring_namespace is not None
    return monitoring_namespace


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
            stack.enter_context(
                cm=patch_maastenantconfig_usage_telemetry(maas_tenant_config=tenant_config),
            )
        yield


@pytest.fixture(scope="class")
def usage_logs_ingestion_wait_timeout() -> int:
    """Return the Loki ingestion poll timeout for usage-log assertions."""
    return USAGE_LOGS_PROPAGATION_TIMEOUT
