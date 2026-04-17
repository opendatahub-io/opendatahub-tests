import pytest
import requests
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.model_explainability.evalhub.constants import EVALHUB_PROVIDERS_PATH
from tests.model_explainability.evalhub.utils import (
    build_headers,
    validate_evalhub_providers,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
)

# ---------------------------------------------------------------------------
# Shared payloads for provider CRUD tests
# ---------------------------------------------------------------------------

USER_PROVIDER_PAYLOAD: dict = {
    "name": "Test Provider",
    "description": "A test provider",
    "benchmarks": [{"id": "arc_easy", "name": "lm_evaluation_harness"}],
}

USER_PROVIDER_UPDATE_PAYLOAD: dict = {
    "name": "Updated Provider Name",
    "description": "Updated description for FVT",
    "benchmarks": [{"id": "arc_easy", "name": "lm_evaluation_harness"}],
}

USER_PROVIDER_PATCH_PAYLOAD: list = [
    {"op": "replace", "path": "/description", "value": "Patched description for FVT"},
    {"op": "replace", "path": "/name", "value": "Patched Provider Name"},
]


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


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-providers-feat"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubProvidersFeature:
    """Feature tests for the EvalHub providers endpoint.

    Covers pagination, query parameter validation, scope filtering,
    search filters, and GET-by-id behaviour.
    """

    def test_list_providers_returns_200_with_pagination(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """GET /providers returns 200 with items, total_count, and pagination structure."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?limit=2"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200
        body = response.json()
        assert "items" in body, "Response missing 'items' key"
        assert "total_count" in body, "Response missing 'total_count' key"
        assert isinstance(body["items"], list)
        assert body["total_count"] >= 1, "Expected at least one system provider"
        assert len(body["items"]) <= 2, f"Limit=2 but got {len(body['items'])} items"

    @pytest.mark.parametrize(
        "query,expected_code",
        [
            pytest.param("offset=-1", "query_parameter_invalid", id="negative-offset"),
            pytest.param("offset=not-a-number", "query_parameter_invalid", id="non-numeric-offset"),
            pytest.param("limit=-1", "query_parameter_invalid", id="negative-limit"),
            pytest.param("limit=invalid", "query_parameter_invalid", id="non-numeric-limit"),
        ],
    )
    def test_list_providers_invalid_query_params(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        query: str,
        expected_code: str,
    ) -> None:
        """List providers with invalid limit or offset returns 400."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?{query}"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 400, f"Expected 400 for '{query}', got {response.status_code}: {response.text}"
        assert response.json().get("message_code") == expected_code

    def test_list_providers_invalid_scope(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """List providers with scope=invalid returns 400."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?scope=invalid"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 400, (
            f"Expected 400 for scope=invalid, got {response.status_code}: {response.text}"
        )

    def test_list_providers_scope_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """scope=tenant returns only user-defined providers (none expected in a fresh tenant)."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?scope=tenant&limit=100"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total_count"] == 0, f"Expected 0 user providers in a fresh tenant, got {body['total_count']}"

    def test_list_providers_scope_system(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """scope=system returns only system-defined providers."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?scope=system&limit=100"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total_count"] >= 1, "Expected at least one system provider"
        assert len(body["items"]) >= 1

    def test_list_providers_no_scope_returns_all(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Omitting scope returns both system and user providers."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)

        resp_all = requests.get(
            url=f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?limit=100",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp_all.status_code == 200

        resp_sys = requests.get(
            url=f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?scope=system&limit=100",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp_sys.status_code == 200

        all_count = resp_all.json()["total_count"]
        sys_count = resp_sys.json()["total_count"]
        assert all_count >= sys_count, f"No-scope count ({all_count}) should be >= system count ({sys_count})"

    def test_list_system_providers_pagination(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """System providers can be paginated with offset."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}"

        resp1 = requests.get(
            url=f"{base}?scope=system&limit=2&offset=0",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp1.status_code == 200
        body1 = resp1.json()
        assert len(body1["items"]) <= 2

        if body1["total_count"] > 2:
            resp2 = requests.get(
                url=f"{base}?scope=system&limit=2&offset=2",
                headers=headers,
                verify=evalhub_mt_ca_bundle_file,
                timeout=10,
            )
            assert resp2.status_code == 200
            body2 = resp2.json()
            assert len(body2["items"]) >= 1

            ids_page1 = {p["resource"]["id"] for p in body1["items"]}
            ids_page2 = {p["resource"]["id"] for p in body2["items"]}
            assert ids_page1.isdisjoint(ids_page2), f"Pages overlap: {ids_page1 & ids_page2}"

    def test_list_providers_with_search_filters(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """List providers combining name, tags, benchmarks, and pagination params."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}"

        # Filter by name (system provider "LM Evaluation Harness" should exist)
        resp = requests.get(
            url=f"{base}?name=LM Evaluation Harness&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] >= 1

        # Filter by non-matching name
        resp = requests.get(
            url=f"{base}?name=nonexistent-provider-xyz&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 0

        # benchmarks=false should return providers without benchmark details
        resp = requests.get(
            url=f"{base}?benchmarks=false&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        for item in body["items"]:
            assert item.get("benchmarks") is None or item["benchmarks"] == [], (
                f"Expected empty benchmarks with benchmarks=false, got: {item.get('benchmarks')}"
            )

    def test_list_providers_by_tags_and_name(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """List providers with tag and name filters (AND/OR semantics)."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}"

        # Non-matching tag returns 0
        resp = requests.get(
            url=f"{base}?tags=nonexistent-tag-xyz&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 0

        # AND semantics (comma): two non-matching tags
        resp = requests.get(
            url=f"{base}?tags=tag-a,tag-b&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 0

        # OR semantics (pipe): non-matching tags
        resp = requests.get(
            url=f"{base}?tags=tag-a|tag-b&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 0

    def test_get_provider_nonexistent_id_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """GET for a non-existent provider id returns 404."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}/00000000-0000-0000-0000-000000000000"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 404, (
            f"Expected 404 for non-existent provider, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "resource_not_found"

    # ------------------------------------------------------------------
    # GET existing system provider
    # ------------------------------------------------------------------

    def test_get_provider_existing_system_id(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """GET for a known system provider id returns 200."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}/lm_evaluation_harness"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, (
            f"Expected 200 for system provider, got {response.status_code}: {response.text}"
        )
        body = response.json()
        assert body["resource"]["id"] == "lm_evaluation_harness"
        assert body["name"] == "LM Evaluation Harness"

    # ------------------------------------------------------------------
    # benchmarks=false
    # ------------------------------------------------------------------

    def test_list_providers_without_benchmarks(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """benchmarks=false returns providers with empty benchmark arrays."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}?benchmarks=false&limit=10"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["total_count"] >= 1
        for item in body["items"]:
            assert item.get("benchmarks") is None or item["benchmarks"] == [], (
                f"Expected empty benchmarks, got: {item.get('benchmarks')}"
            )

    # ------------------------------------------------------------------
    # GET / PUT with empty path → 404
    # ------------------------------------------------------------------

    def test_get_provider_empty_path_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """GET /providers/ (trailing slash, no id) returns 404."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}/"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 404, (
            f"Expected 404 for empty provider path, got {response.status_code}: {response.text}"
        )

    def test_update_provider_empty_path_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """PUT /providers/ (trailing slash, no id) returns 404."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}/"
        response = requests.put(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=USER_PROVIDER_UPDATE_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 404, (
            f"Expected 404 for empty provider PUT path, got {response.status_code}: {response.text}"
        )

    # ------------------------------------------------------------------
    # Update / Patch system provider → 400
    # ------------------------------------------------------------------

    def test_update_system_provider_returns_400(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """PUT on a system provider returns 400 read_only_provider."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}/lm_evaluation_harness"
        response = requests.put(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=USER_PROVIDER_UPDATE_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 400, (
            f"Expected 400 for system provider update, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "read_only_provider"

    def test_patch_system_provider_returns_400(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """PATCH on a system provider returns 400 read_only_provider."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}/lm_evaluation_harness"
        response = requests.patch(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=USER_PROVIDER_PATCH_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 400, (
            f"Expected 400 for system provider patch, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "read_only_provider"

    # ------------------------------------------------------------------
    # Update / Patch non-existent provider → 404
    # ------------------------------------------------------------------

    def test_update_nonexistent_provider_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """PUT on a non-existent provider returns 404."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}/00000000-0000-0000-0000-000000000000"
        response = requests.put(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=USER_PROVIDER_UPDATE_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 404, (
            f"Expected 404 for non-existent provider update, got {response.status_code}: {response.text}"
        )

    def test_patch_nonexistent_provider_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """PATCH on a non-existent provider returns 404."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}/00000000-0000-0000-0000-000000000000"
        response = requests.patch(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=USER_PROVIDER_PATCH_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 404, (
            f"Expected 404 for non-existent provider patch, got {response.status_code}: {response.text}"
        )

    # ------------------------------------------------------------------
    # CRUD lifecycle: Create → GET → Update → Patch → DELETE
    # ------------------------------------------------------------------

    def test_create_get_update_patch_delete_user_provider(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Full CRUD lifecycle for a user-defined provider."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}"

        # CREATE
        resp = requests.post(
            url=base,
            headers=headers,
            json=USER_PROVIDER_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 201, f"Expected 201 for provider create, got {resp.status_code}: {resp.text}"
        body = resp.json()
        assert body["name"] == "Test Provider"
        provider_id = body["resource"]["id"]

        # GET
        resp = requests.get(
            url=f"{base}/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Test Provider"

        # UPDATE (PUT)
        resp = requests.put(
            url=f"{base}/{provider_id}",
            headers=headers,
            json=USER_PROVIDER_UPDATE_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Provider Name"
        assert resp.json()["description"] == "Updated description for FVT"

        # Verify update persisted
        resp = requests.get(
            url=f"{base}/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Updated Provider Name"

        # PATCH (JSON Patch: name + description)
        resp = requests.patch(
            url=f"{base}/{provider_id}",
            headers=headers,
            json=USER_PROVIDER_PATCH_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "Patched Provider Name"
        assert resp.json()["description"] == "Patched description for FVT"

        # PATCH runtime
        resp = requests.patch(
            url=f"{base}/{provider_id}",
            headers=headers,
            json=[{"op": "replace", "path": "/runtime", "value": {"local": {"command": "echo hello"}}}],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["runtime"]["local"]["command"] == "echo hello"

        # PATCH tags
        resp = requests.patch(
            url=f"{base}/{provider_id}",
            headers=headers,
            json=[{"op": "add", "path": "/tags", "value": ["foo", "bar"]}],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert "foo" in resp.json()["tags"]
        assert "bar" in resp.json()["tags"]

        # DELETE
        resp = requests.delete(
            url=f"{base}/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 204

        # Verify deleted
        resp = requests.get(
            url=f"{base}/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 404

    # ------------------------------------------------------------------
    # PATCH validation errors
    # ------------------------------------------------------------------

    def test_patch_provider_invalid_operation(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """PATCH with an invalid op returns 400 invalid_patch_operation."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}"

        # Create a provider to patch
        resp = requests.post(
            url=base,
            headers=headers,
            json=USER_PROVIDER_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 201
        provider_id = resp.json()["resource"]["id"]

        # Invalid operation
        resp = requests.patch(
            url=f"{base}/{provider_id}",
            headers=headers,
            json=[{"op": "invalid_op", "path": "/name", "value": "x"}],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400
        assert resp.json().get("message_code") == "invalid_patch_operation"

        # Cleanup
        requests.delete(
            url=f"{base}/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )

    def test_patch_provider_unallowed_path_or_remove_required(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """PATCH on unallowed path or removing a required field returns 400."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}"

        # Create a provider to patch
        resp = requests.post(
            url=base,
            headers=headers,
            json=USER_PROVIDER_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 201
        provider_id = resp.json()["resource"]["id"]

        # Unallowed path: try to change resource.id
        resp = requests.patch(
            url=f"{base}/{provider_id}",
            headers=headers,
            json=[{"op": "replace", "path": "/resource/id", "value": "hacked-id"}],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400
        assert resp.json().get("message_code") == "unallowed_patch"

        # Remove required field: /name
        resp = requests.patch(
            url=f"{base}/{provider_id}",
            headers=headers,
            json=[{"op": "remove", "path": "/name"}],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400
        assert resp.json().get("message_code") == "unallowed_patch"

        # Cleanup
        requests.delete(
            url=f"{base}/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
