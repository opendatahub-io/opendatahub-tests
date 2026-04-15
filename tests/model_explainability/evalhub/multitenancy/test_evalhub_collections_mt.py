from collections.abc import Generator

import pytest
import requests
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.model_explainability.evalhub.constants import (
    EVALHUB_COLLECTIONS_PATH,
    EVALHUB_PROVIDERS_PATH,
)
from tests.model_explainability.evalhub.utils import (
    build_headers,
    list_evalhub_collections,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
)

# ---------------------------------------------------------------------------
# Shared payloads for collection feature tests
# ---------------------------------------------------------------------------

COLLECTION_PAYLOAD: dict = {
    "name": "test-benchmarks-collection",
    "description": "Collection of benchmarks for FVT",
    "category": "test",
    "benchmarks": [
        {
            "id": "arc_easy",
            "provider_id": "lm_evaluation_harness",
            "parameters": {"weight": 3},
        }
    ],
}

COLLECTION_NO_BENCHMARKS_PAYLOAD: dict = {
    "name": "no-benchmarks-collection",
    "description": "Collection without benchmarks field",
    "category": "test",
}

COLLECTION_EMPTY_BENCHMARKS_PAYLOAD: dict = {
    "name": "empty-benchmarks-collection",
    "description": "Collection with empty benchmarks array",
    "category": "test",
    "benchmarks": [],
}

COLLECTION_NO_NAME_PAYLOAD: dict = {
    "description": "Collection without name field",
    "category": "test",
    "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
}

COLLECTION_NO_CATEGORY_PAYLOAD: dict = {
    "name": "test-benchmarks-collection",
    "description": "Collection of benchmarks for FVT",
    "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
}

COLLECTION_NO_DESCRIPTION_PAYLOAD: dict = {
    "name": "no-description-collection",
    "category": "test",
    "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
}

COLLECTION_BENCHMARK_NO_ID_PAYLOAD: dict = {
    "name": "benchmark-no-id",
    "description": "Collection with benchmark missing id",
    "category": "test",
    "benchmarks": [{"provider_id": "lm_evaluation_harness"}],
}

COLLECTION_BENCHMARK_NO_PROVIDER_ID_PAYLOAD: dict = {
    "name": "benchmark-no-provider-id",
    "description": "Collection with benchmark missing provider_id",
    "category": "test",
    "benchmarks": [{"id": "arc_easy"}],
}

COLLECTION_UPDATE_PAYLOAD: dict = {
    "name": "updated-collection-name",
    "description": "Updated description for FVT",
    "category": "test",
    "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
}

COLLECTION_UPDATE_NO_NAME_PAYLOAD: dict = {
    "description": "Updated description for FVT",
    "category": "test",
    "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
}

COLLECTION_UPDATE_NO_DESCRIPTION_PAYLOAD: dict = {
    "name": "updated-no-description",
    "category": "test",
    "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
}

CUSTOM_PROVIDER_BENCHMARK_URL_PAYLOAD: dict = {
    "name": "FVT Provider Benchmark URL",
    "description": "Custom provider with a benchmark that defines url for collection enrichment FVT",
    "title": "FVT Provider Benchmark URL",
    "benchmarks": [
        {
            "id": "bench_with_url",
            "name": "Benchmark With URL",
            "category": "test",
            "url": "https://example.com/fvt-custom-provider-benchmark",
        }
    ],
    "runtime": {"local": {"command": "true"}},
}


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


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-collections-feat"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubCollectionsFeature:
    """Feature tests for the EvalHub collections endpoint.

    Covers CRUD lifecycle (create, get, update, patch, delete),
    validation (missing/empty fields), pagination, filtering,
    and benchmark URL enrichment from custom providers.
    """

    # ------------------------------------------------------------------
    # Fixtures: setup (create) and teardown (delete) via yield
    # ------------------------------------------------------------------

    @pytest.fixture()
    def collection(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> Generator[dict]:
        """Create a collection with COLLECTION_PAYLOAD; delete on teardown."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"

        resp = requests.post(
            url=base,
            headers=headers,
            json=COLLECTION_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 201, f"Expected 201 for collection create, got {resp.status_code}: {resp.text}"
        body = resp.json()
        collection_id = body["resource"]["id"]

        yield {"id": collection_id, "body": body, "base": base, "headers": headers}

        requests.delete(
            url=f"{base}/{collection_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )

    @pytest.fixture()
    def collection_no_description(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> Generator[dict]:
        """Create a collection without description; delete on teardown."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"

        resp = requests.post(
            url=base,
            headers=headers,
            json=COLLECTION_NO_DESCRIPTION_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 201, (
            f"Expected 201 for collection without description, got {resp.status_code}: {resp.text}"
        )
        collection_id = resp.json()["resource"]["id"]

        yield {"id": collection_id, "base": base, "headers": headers}

        requests.delete(
            url=f"{base}/{collection_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )

    @pytest.fixture()
    def provider_and_collection_with_url(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> Generator[dict]:
        """Create a custom provider with benchmark URL, then a collection referencing it.

        Yields both IDs; deletes collection then provider on teardown.
        """
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        providers_base = f"https://{evalhub_mt_route.host}{EVALHUB_PROVIDERS_PATH}"
        collections_base = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"

        # Create custom provider
        resp = requests.post(
            url=providers_base,
            headers=headers,
            json=CUSTOM_PROVIDER_BENCHMARK_URL_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 201, f"Expected 201 for provider create, got {resp.status_code}: {resp.text}"
        provider_body = resp.json()
        provider_id = provider_body["resource"]["id"]

        # Create collection referencing the custom provider's benchmark
        collection_payload = {
            "name": "fvt-benchmark-url-collection",
            "description": "Collection referencing custom provider benchmark with url",
            "category": "test",
            "benchmarks": [
                {
                    "id": "bench_with_url",
                    "provider_id": provider_id,
                }
            ],
        }
        resp = requests.post(
            url=collections_base,
            headers=headers,
            json=collection_payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 201, f"Expected 201 for collection create, got {resp.status_code}: {resp.text}"
        collection_body = resp.json()
        collection_id = collection_body["resource"]["id"]

        yield {
            "provider_id": provider_id,
            "provider_body": provider_body,
            "collection_id": collection_id,
            "collection_body": collection_body,
            "collections_base": collections_base,
            "headers": headers,
        }

        requests.delete(
            url=f"{collections_base}/{collection_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        requests.delete(
            url=f"{providers_base}/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )

    @pytest.fixture()
    def three_collections(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> Generator[dict]:
        """Create 3 collections with different tags/categories; delete all on teardown."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"
        created_ids = []

        payloads = [
            {
                "name": "test-collection-1",
                "description": "FVT",
                "category": "test",
                "tags": ["test-tag-1", "test-tag-2"],
                "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
            },
            {
                "name": "test-collection-2",
                "description": "FVT",
                "category": "test",
                "tags": ["test-tag-1"],
                "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
            },
            {
                "name": "test-collection-3",
                "description": "FVT",
                "category": "test3",
                "tags": ["test-tag-3", "test-tag-2", "test-tag-1"],
                "benchmarks": [{"id": "arc_easy", "provider_id": "lm_evaluation_harness"}],
            },
        ]
        for p in payloads:
            resp = requests.post(
                url=base,
                headers=headers,
                json=p,
                verify=evalhub_mt_ca_bundle_file,
                timeout=30,
            )
            assert resp.status_code == 201
            created_ids.append(resp.json()["resource"]["id"])

        yield {"ids": created_ids, "base": base, "headers": headers}

        for cid in created_ids:
            requests.delete(
                url=f"{base}/{cid}?hard_delete=true",
                headers=headers,
                verify=evalhub_mt_ca_bundle_file,
                timeout=10,
            )

    @pytest.fixture()
    def three_collections_for_pagination(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> Generator[dict]:
        """Create 3 collections for pagination tests; delete all on teardown."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"
        created_ids = []

        for _ in range(3):
            resp = requests.post(
                url=base,
                headers=headers,
                json=COLLECTION_PAYLOAD,
                verify=evalhub_mt_ca_bundle_file,
                timeout=30,
            )
            assert resp.status_code == 201
            created_ids.append(resp.json()["resource"]["id"])

        yield {"ids": created_ids, "base": base, "headers": headers}

        for cid in created_ids:
            requests.delete(
                url=f"{base}/{cid}?hard_delete=true",
                headers=headers,
                verify=evalhub_mt_ca_bundle_file,
                timeout=10,
            )

    # ------------------------------------------------------------------
    # Create and get by id → 201 / 200
    # ------------------------------------------------------------------

    def test_create_and_get_collection(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """POST collection → 201, GET by id → 200 with correct fields."""
        body = collection["body"]
        assert body["name"] == COLLECTION_PAYLOAD["name"]

        resp = requests.get(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == COLLECTION_PAYLOAD["name"]
        assert body["resource"]["id"] == collection["id"]
        assert len(body["benchmarks"]) == 1
        assert body["benchmarks"][0]["id"] == "arc_easy"
        assert body["benchmarks"][0]["parameters"]["weight"] == 3

    # ------------------------------------------------------------------
    # Validation: missing / empty required fields → 400
    # ------------------------------------------------------------------

    @pytest.mark.parametrize(
        "payload,field",
        [
            pytest.param(COLLECTION_NO_BENCHMARKS_PAYLOAD, "benchmarks", id="missing-benchmarks"),
            pytest.param(COLLECTION_EMPTY_BENCHMARKS_PAYLOAD, "benchmarks", id="empty-benchmarks"),
            pytest.param(COLLECTION_NO_NAME_PAYLOAD, "name", id="missing-name"),
            pytest.param(COLLECTION_NO_CATEGORY_PAYLOAD, "category", id="missing-category"),
            pytest.param(COLLECTION_BENCHMARK_NO_ID_PAYLOAD, "benchmark id", id="benchmark-missing-id"),
            pytest.param(
                COLLECTION_BENCHMARK_NO_PROVIDER_ID_PAYLOAD, "benchmark provider_id", id="benchmark-missing-provider-id"
            ),
        ],
    )
    def test_create_collection_missing_required_field_returns_400(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        payload: dict,
        field: str,
    ) -> None:
        """POST collection with missing or empty required field → 400."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"
        resp = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 400, f"Expected 400 for invalid '{field}', got {resp.status_code}: {resp.text}"

    def test_create_collection_without_description_returns_201(
        self,
        collection_no_description: dict,
    ) -> None:
        """POST collection without description (optional) → 201."""
        assert collection_no_description["id"]

    # ------------------------------------------------------------------
    # Benchmark URL enrichment from custom provider
    # ------------------------------------------------------------------

    def test_create_collection_persists_benchmark_url(
        self,
        provider_and_collection_with_url: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """Create a custom provider with a benchmark URL, then create a collection
        referencing it. The collection should enrich and persist the URL."""
        ctx = provider_and_collection_with_url
        expected_url = "https://example.com/fvt-custom-provider-benchmark"

        # Verify provider stored the URL
        assert ctx["provider_body"]["benchmarks"][0].get("url") == expected_url, (
            f"Provider benchmark URL not stored: {ctx['provider_body']['benchmarks']}"
        )

        # Verify URL enriched in the create response
        assert ctx["collection_body"]["benchmarks"][0].get("url") == expected_url, (
            f"Expected enriched benchmark URL in create response, got: {ctx['collection_body']['benchmarks'][0]}"
        )

        # GET collection and verify benchmark URL persisted
        resp = requests.get(
            url=f"{ctx['collections_base']}/{ctx['collection_id']}",
            headers=ctx["headers"],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["benchmarks"]) == 1
        assert body["benchmarks"][0].get("url") == expected_url, (
            f"Expected enriched benchmark URL in GET response, got: {body['benchmarks'][0]}"
        )

    # ------------------------------------------------------------------
    # Non-existent / empty id → 404 (GET, PUT, PATCH, DELETE)
    # ------------------------------------------------------------------

    NON_EXISTENT_PATH = "/00000000-0000-0000-0000-000000000000"

    @pytest.mark.parametrize(
        "method,path_suffix,json_body",
        [
            pytest.param("GET", NON_EXISTENT_PATH, None, id="get-nonexistent"),
            pytest.param("GET", "/", None, id="get-empty"),
            pytest.param("PUT", NON_EXISTENT_PATH, COLLECTION_UPDATE_PAYLOAD, id="put-nonexistent"),
            pytest.param("PUT", "/", COLLECTION_UPDATE_PAYLOAD, id="put-empty"),
            pytest.param(
                "PATCH",
                NON_EXISTENT_PATH,
                [{"op": "replace", "path": "/name", "value": "x"}],
                id="patch-nonexistent",
            ),
            pytest.param("PATCH", "/", [{"op": "replace", "path": "/name", "value": "x"}], id="patch-empty"),
            pytest.param("DELETE", NON_EXISTENT_PATH + "?hard_delete=true", None, id="delete-nonexistent"),
        ],
    )
    def test_collection_nonexistent_or_empty_id_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        method: str,
        path_suffix: str,
        json_body: dict | list | None,
    ) -> None:
        """Request against a non-existent or empty collection id returns 404."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}{path_suffix}"
        resp = requests.request(
            method=method,
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=json_body,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 404, f"Expected 404 for {method} {path_suffix}, got {resp.status_code}: {resp.text}"

    # ------------------------------------------------------------------
    # Update (PUT) lifecycle
    # ------------------------------------------------------------------

    def test_update_collection_returns_200_and_persists(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PUT collection → 200, changes persisted on GET."""
        resp = requests.put(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=COLLECTION_UPDATE_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "updated-collection-name"
        assert resp.json()["description"] == "Updated description for FVT"

        resp = requests.get(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "updated-collection-name"
        assert resp.json()["description"] == "Updated description for FVT"

    # ------------------------------------------------------------------
    # Update validation: missing / empty required fields → 400
    # ------------------------------------------------------------------

    def test_update_collection_without_name_returns_400(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PUT collection without name → 400."""
        resp = requests.put(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=COLLECTION_NO_NAME_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400, f"Expected 400 for update without name, got {resp.status_code}: {resp.text}"

    def test_update_collection_without_description_returns_200(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PUT collection without description (optional) → 200."""
        resp = requests.put(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=COLLECTION_UPDATE_NO_DESCRIPTION_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200, (
            f"Expected 200 for update without description, got {resp.status_code}: {resp.text}"
        )

    def test_update_collection_without_benchmarks_returns_400(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PUT collection without benchmarks field → 400."""
        resp = requests.put(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=COLLECTION_NO_BENCHMARKS_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for update without benchmarks, got {resp.status_code}: {resp.text}"
        )

    def test_update_collection_with_empty_benchmarks_returns_400(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PUT collection with empty benchmarks array → 400."""
        resp = requests.put(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=COLLECTION_EMPTY_BENCHMARKS_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for update with empty benchmarks, got {resp.status_code}: {resp.text}"
        )

    def test_update_collection_benchmark_missing_id_returns_400(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PUT collection with benchmark missing id → 400."""
        resp = requests.put(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=COLLECTION_BENCHMARK_NO_ID_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for update with benchmark missing id, got {resp.status_code}: {resp.text}"
        )

    def test_update_collection_benchmark_missing_provider_id_returns_400(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PUT collection with benchmark missing provider_id → 400."""
        resp = requests.put(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=COLLECTION_BENCHMARK_NO_PROVIDER_ID_PAYLOAD,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400, (
            f"Expected 400 for update with benchmark missing provider_id, got {resp.status_code}: {resp.text}"
        )

    # ------------------------------------------------------------------
    # PATCH collection
    # ------------------------------------------------------------------

    def test_patch_collection_name_returns_200(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PATCH collection name → 200, change persisted on GET."""
        resp = requests.patch(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=[{"op": "replace", "path": "/name", "value": "patched-collection-name"}],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200

        resp = requests.get(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["name"] == "patched-collection-name"

    def test_patch_benchmark_element_in_collection_returns_200(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PATCH a single benchmark field → 200, change persisted."""
        resp = requests.patch(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=[{"op": "replace", "path": "/benchmarks/0/id", "value": "patched-benchmark-id"}],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200

        resp = requests.get(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["benchmarks"][0]["id"] == "patched-benchmark-id"
        assert len(resp.json()["benchmarks"]) == 1

    def test_patch_entire_benchmark_element_in_collection_returns_200(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PATCH replace entire benchmark element → 200, change persisted."""
        resp = requests.patch(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=[
                {
                    "op": "replace",
                    "path": "/benchmarks/0",
                    "value": {"id": "replaced-benchmark-id", "provider_id": "other_provider"},
                }
            ],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200

        resp = requests.get(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["benchmarks"][0]["id"] == "replaced-benchmark-id"
        assert resp.json()["benchmarks"][0]["provider_id"] == "other_provider"
        assert len(resp.json()["benchmarks"]) == 1

    def test_patch_collection_invalid_body_returns_400(
        self,
        collection: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """PATCH with invalid op returns 400."""
        resp = requests.patch(
            url=f"{collection['base']}/{collection['id']}",
            headers=collection["headers"],
            json=[{"op": "invalid_op", "path": "/name", "value": "x"}],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400, f"Expected 400 for invalid patch op, got {resp.status_code}: {resp.text}"

    # ------------------------------------------------------------------
    # List collections
    # ------------------------------------------------------------------

    def test_list_collections_returns_200_with_pagination(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """GET /collections returns 200 with items, limit, and total_count."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"
        resp = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "limit" in body
        assert "total_count" in body

    def test_list_collections_with_pagination_params(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """GET /collections?limit=5&offset=0 returns 200 with pagination fields."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}?limit=5&offset=0"
        resp = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "items" in body
        assert "limit" in body
        assert "total_count" in body

    def test_list_collections_pagination_next_href(
        self,
        three_collections_for_pagination: dict,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Create 3 collections, list with limit=2, follow next href, verify second page."""
        ctx = three_collections_for_pagination

        # First page
        resp = requests.get(
            url=f"{ctx['base']}?limit=2&offset=0",
            headers=ctx["headers"],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 2
        assert "next" in body, "Expected 'next' in paginated response"
        next_href = body["next"]["href"]

        # Second page — next_href may be relative or absolute
        if next_href.startswith("http"):
            from urllib.parse import urlparse

            parsed = urlparse(url=next_href)
            assert parsed.hostname == evalhub_mt_route.host, (
                f"next href points to unexpected host: {parsed.hostname} != {evalhub_mt_route.host}"
            )
            next_url = next_href
        else:
            next_url = f"https://{evalhub_mt_route.host}{next_href}"
        resp = requests.get(
            url=next_url,
            headers=ctx["headers"],
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert len(resp.json()["items"]) >= 1

    @pytest.mark.parametrize(
        "query,expected_code",
        [
            pytest.param("limit=invalid", "query_parameter_invalid", id="invalid-limit"),
            pytest.param("offset=not-a-number", "query_parameter_invalid", id="invalid-offset"),
        ],
    )
    def test_list_collections_invalid_limit_or_offset(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        query: str,
        expected_code: str,
    ) -> None:
        """List collections with invalid limit or offset returns 400."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}?{query}"
        resp = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400, f"Expected 400 for '{query}', got {resp.status_code}: {resp.text}"
        assert resp.json().get("message_code") == expected_code

    def test_list_collections_invalid_scope(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """List collections with invalid scope returns 400."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}?scope=invalid"
        resp = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400, f"Expected 400 for invalid scope, got {resp.status_code}: {resp.text}"
        assert resp.json().get("message_code") == "query_parameter_value_invalid"

    # ------------------------------------------------------------------
    # List by tags, name, and category (AND/OR semantics)
    # ------------------------------------------------------------------

    def test_list_collections_by_tags_name_and_category(
        self,
        three_collections: dict,
        evalhub_mt_ca_bundle_file: str,
    ) -> None:
        """Create 3 collections with different tags/categories, verify filter semantics."""
        ctx = three_collections

        def count(query: str) -> int:
            r = requests.get(
                url=f"{ctx['base']}?{query}",
                headers=ctx["headers"],
                verify=evalhub_mt_ca_bundle_file,
                timeout=10,
            )
            assert r.status_code == 200
            return len(r.json()["items"])

        # Tags: single tag
        assert count("tags=test-tag-1") == 3
        assert count("tags=test-tag-2") == 2
        assert count("tags=test-tag-3") == 1
        assert count("tags=test-tag-4") == 0

        # Tags: AND (comma) and OR (pipe)
        assert count("tags=test-tag-2,test-tag-3") == 1  # AND
        assert count("tags=test-tag-2|test-tag-3") == 2  # OR

        # Name filter
        assert count("name=test-collection-1") == 1
        assert count("name=test-collection-4") == 0

        # Category filter
        assert count("category=test") == 2
        assert count("category=test3") == 1
        assert count("category=test4") == 0

        # Combined: category + name
        assert count("category=test&name=test-collection-1") == 1
        assert count("category=test&name=test-collection-3") == 0
        assert count("category=test3&name=test-collection-3") == 1

        # Combined: name + tags
        assert count("name=test-collection-1&tags=test-tag-1") == 1
        assert count("name=test-collection-1&tags=test-tag-3") == 0

    # ------------------------------------------------------------------
    # List system-defined collections with pagination
    # ------------------------------------------------------------------

    def test_list_system_collections_with_pagination(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """List system collections (scope=system) with pagination, verify at least 1."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base = f"https://{evalhub_mt_route.host}{EVALHUB_COLLECTIONS_PATH}"

        resp = requests.get(
            url=f"{base}?limit=50&offset=0&scope=system",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["limit"] == 50
        num_collections = body["total_count"]
        assert num_collections >= 1, f"Expected at least 1 system collection, got {num_collections}"
        assert len(body["items"]) == num_collections
