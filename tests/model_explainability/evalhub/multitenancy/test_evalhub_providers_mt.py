import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.model_explainability.evalhub.constants import EVALHUB_PROVIDERS_PATH
from tests.model_explainability.evalhub.utils import (
    validate_evalhub_providers,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-providers-mt"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
@pytest.mark.usefixtures("evalhub_mt_ready")
class TestEvalHubProvidersMT:
    """Multi-tenancy tests for the EvalHub providers endpoint.

    Three scenarios:
    - Authorized tenant: user with RBAC in tenant-a lists providers → 200
    - Cross-tenant:      same user lists providers for tenant-b → denied (400/403)
    - Missing tenant:    request without X-Tenant header → 400
    """

    def test_providers_authorized_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """User with providers-access in tenant-a can list providers."""
        validate_evalhub_providers(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant_namespace=tenant_a_namespace.name,
        )

    def test_providers_cross_tenant_forbidden(
        self,
        tenant_a_token: str,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """User with providers-access in tenant-a is denied for tenant-b."""
        validate_evalhub_request_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_PROVIDERS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
        )

    def test_providers_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Request without X-Tenant header is rejected with 400."""
        validate_evalhub_request_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_PROVIDERS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
        )
