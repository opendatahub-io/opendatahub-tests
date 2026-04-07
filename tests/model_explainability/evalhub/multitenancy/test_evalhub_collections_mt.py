import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.model_explainability.evalhub.constants import EVALHUB_COLLECTIONS_PATH
from tests.model_explainability.evalhub.utils import (
    list_evalhub_collections,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-collections-mt"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubCollectionsMT:
    """Multi-tenancy tests for the EvalHub collections endpoint.

    Three scenarios:
    - Authorized tenant: user with RBAC in tenant-a lists collections → 200
    - Cross-tenant:      same user lists collections for tenant-b → denied
    - Missing tenant:    request without X-Tenant header → 400
    """

    def test_collections_authorized_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """User with collections-access RBAC in tenant-a can list collections."""
        data = list_evalhub_collections(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
        )
        # Collections may be empty but the response should be valid
        assert isinstance(data.get("items"), list), f"Expected 'items' list in collections response, got: {data}"

    def test_collections_cross_tenant_denied(
        self,
        tenant_a_token: str,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """User with RBAC in tenant-a is denied for tenant-b collections."""
        validate_evalhub_request_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_COLLECTIONS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
        )

    def test_collections_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Request without X-Tenant header is rejected with 400."""
        validate_evalhub_request_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_COLLECTIONS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
        )
