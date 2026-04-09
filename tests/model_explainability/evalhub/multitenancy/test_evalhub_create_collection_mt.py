import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.model_explainability.evalhub.constants import EVALHUB_COLLECTIONS_PATH
from tests.model_explainability.evalhub.utils import (
    submit_evalhub_collection,
    validate_evalhub_post_denied,
    validate_evalhub_post_no_tenant,
)

COLLECTION_PAYLOAD: dict = {
    "name": "MT Test Collection",
    "category": "test",
    "description": "Collection created by multi-tenancy integration test",
    "benchmarks": [
        {
            "id": "arc_easy",
            "provider_id": "lm_evaluation_harness",
        }
    ],
}


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-create-coll-mt"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubCreateCollectionMT:
    """Multi-tenancy tests for creating custom collections.

    Three scenarios:
    - Authorized tenant: user with RBAC in tenant-a creates a collection → 201
    - Cross-tenant:      same user creates for tenant-b → denied
    - Missing tenant:    POST without X-Tenant header → 400
    """

    def test_create_collection_authorized_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """User with collections-create RBAC in tenant-a can create a collection."""
        response = submit_evalhub_collection(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=COLLECTION_PAYLOAD,
        )
        assert response.status_code == 201, f"Expected 201 Created, got {response.status_code}: {response.text}"
        data = response.json()
        resource = data.get("resource", {})
        assert resource.get("id"), f"Collection response missing 'resource.id': {data}"
        assert data.get("name") == COLLECTION_PAYLOAD["name"]

    def test_create_collection_cross_tenant_denied(
        self,
        tenant_a_token: str,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """User with RBAC in tenant-a is denied collection creation for tenant-b."""
        validate_evalhub_post_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_COLLECTIONS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
            payload=COLLECTION_PAYLOAD,
        )

    def test_create_collection_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Collection creation without X-Tenant header is rejected with 400."""
        validate_evalhub_post_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_COLLECTIONS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            payload=COLLECTION_PAYLOAD,
        )
