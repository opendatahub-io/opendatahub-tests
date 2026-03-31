"""
EvalHub Multi-Tenancy E2E Tests

Tests multi-tenant isolation, RBAC, and evaluation workflows across
separate tenant namespaces with dedicated service accounts.
"""
import pytest
import requests
import structlog
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.model_explainability.evalhub.constants import TENANT_A_NAME, TENANT_B_NAME
from tests.model_explainability.evalhub.utils import (
    create_evalhub_collection,
    create_evalhub_evaluation_job,
    get_evalhub_evaluation_job,
    get_evalhub_provider,
    list_evalhub_benchmarks,
    list_evalhub_collections,
    list_evalhub_evaluation_jobs,
    list_evalhub_providers,
)

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.usefixtures(
    "evalhub_cr",
    "evalhub_deployment",
    "multi_tenant_setup",
)
@pytest.mark.smoke
class TestEvalHubMultiTenancy:
    """
    Tests for EvalHub multi-tenancy functionality.

    This test class validates:
    1. Tenant isolation - each tenant can only access their own resources
    2. RBAC enforcement - unauthorized users cannot access EvalHub APIs
    3. Benchmark and evaluation job workflows across multiple tenants
    4. Collections management per tenant
    """

    def test_providers_list_with_tenant_scoping(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify both tenants can list providers with proper scoping."""
        for tenant_name in ["tenant_a", "tenant_b"]:
            tenant_data = multi_tenant_setup[tenant_name]
            providers = list_evalhub_providers(
                host=evalhub_route.host,
                token=tenant_data["token"],
                ca_bundle_file=evalhub_ca_bundle_file,
                tenant=tenant_data["namespace"].name,
            )

            assert "items" in providers, f"Providers response missing 'items' for {tenant_name}"
            assert "total_count" in providers, f"Providers response missing 'total_count' for {tenant_name}"
            assert providers["total_count"] > 0, f"No providers found for {tenant_name}"

            # Verify lm_evaluation_harness provider exists
            provider_ids = [p["id"] for p in providers["items"]]
            assert "lm_evaluation_harness" in provider_ids, (
                f"lm_evaluation_harness provider not found for {tenant_name}"
            )

    def test_unauthorised_user_cannot_access_providers(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify unauthorised service account cannot access providers."""
        with pytest.raises(requests.HTTPError) as exc_info:
            list_evalhub_providers(
                host=evalhub_route.host,
                token=multi_tenant_setup["unauthorised"]["token"],
                ca_bundle_file=evalhub_ca_bundle_file,
                tenant=TENANT_A_NAME,
            )

        assert exc_info.value.response.status_code == 403, (
            f"Expected 403 Forbidden for unauthorised user, got {exc_info.value.response.status_code}"
        )

    def test_provider_details_with_benchmarks(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify provider details include benchmark information."""
        tenant_data = multi_tenant_setup["tenant_a"]

        provider = get_evalhub_provider(
            host=evalhub_route.host,
            token=tenant_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_data["namespace"].name,
            provider_id="lm_evaluation_harness",
        )

        assert "id" in provider, "Provider missing 'id' field"
        assert provider["id"] == "lm_evaluation_harness", "Incorrect provider ID"
        assert "name" in provider, "Provider missing 'name' field"
        assert "benchmarks" in provider, "Provider missing 'benchmarks' field"
        assert isinstance(provider["benchmarks"], list), "Provider benchmarks should be a list"
        assert len(provider["benchmarks"]) > 0, "Provider has no benchmarks"

        # Verify benchmark structure
        for benchmark in provider["benchmarks"]:
            assert "id" in benchmark, "Benchmark missing 'id' field"
            assert "name" in benchmark, "Benchmark missing 'name' field"
            assert "category" in benchmark, "Benchmark missing 'category' field"

    def test_list_benchmarks_for_tenant(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify tenants can list available benchmarks."""
        tenant_data = multi_tenant_setup["tenant_a"]

        benchmarks = list_evalhub_benchmarks(
            host=evalhub_route.host,
            token=tenant_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_data["namespace"].name,
        )

        assert "items" in benchmarks, "Benchmarks response missing 'items'"
        assert isinstance(benchmarks["items"], list), "Benchmarks items should be a list"
        assert len(benchmarks["items"]) > 0, "No benchmarks available"

        # Verify at least one common benchmark exists
        benchmark_ids = [b["id"] for b in benchmarks["items"]]
        common_benchmarks = ["mmlu", "hellaswag", "arc_easy", "arc_challenge", "truthfulqa"]
        found_benchmarks = [b for b in common_benchmarks if b in benchmark_ids]
        assert len(found_benchmarks) > 0, f"Expected at least one of {common_benchmarks}, found: {benchmark_ids}"

    def test_create_collection_per_tenant(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify each tenant can create their own collections."""
        for tenant_name in ["tenant_a", "tenant_b"]:
            tenant_data = multi_tenant_setup[tenant_name]

            collection = create_evalhub_collection(
                host=evalhub_route.host,
                token=tenant_data["token"],
                ca_bundle_file=evalhub_ca_bundle_file,
                tenant=tenant_data["namespace"].name,
                name=f"{tenant_name}-test-collection",
                description=f"Test collection for {tenant_name}",
            )

            assert "id" in collection, f"Collection response missing 'id' for {tenant_name}"
            assert "name" in collection, f"Collection response missing 'name' for {tenant_name}"
            assert collection["name"] == f"{tenant_name}-test-collection", (
                f"Collection name mismatch for {tenant_name}"
            )

    def test_tenant_collections_are_isolated(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify tenants can only see their own collections."""
        # Create collections for both tenants
        for tenant_name in ["tenant_a", "tenant_b"]:
            tenant_data = multi_tenant_setup[tenant_name]
            create_evalhub_collection(
                host=evalhub_route.host,
                token=tenant_data["token"],
                ca_bundle_file=evalhub_ca_bundle_file,
                tenant=tenant_data["namespace"].name,
                name=f"{tenant_name}-isolated-collection",
                description=f"Isolated collection for {tenant_name}",
            )

        # Verify tenant-a only sees their own collections
        tenant_a_data = multi_tenant_setup["tenant_a"]
        tenant_a_collections = list_evalhub_collections(
            host=evalhub_route.host,
            token=tenant_a_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_a_data["namespace"].name,
        )

        tenant_a_collection_names = [c["name"] for c in tenant_a_collections.get("items", [])]
        assert "tenant_a-isolated-collection" in tenant_a_collection_names, (
            "Tenant A should see their own collection"
        )
        assert "tenant_b-isolated-collection" not in tenant_a_collection_names, (
            "Tenant A should NOT see Tenant B's collection"
        )

        # Verify tenant-b only sees their own collections
        tenant_b_data = multi_tenant_setup["tenant_b"]
        tenant_b_collections = list_evalhub_collections(
            host=evalhub_route.host,
            token=tenant_b_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_b_data["namespace"].name,
        )

        tenant_b_collection_names = [c["name"] for c in tenant_b_collections.get("items", [])]
        assert "tenant_b-isolated-collection" in tenant_b_collection_names, (
            "Tenant B should see their own collection"
        )
        assert "tenant_a-isolated-collection" not in tenant_b_collection_names, (
            "Tenant B should NOT see Tenant A's collection"
        )

    def test_create_evaluation_job_for_tenant(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify tenant can create evaluation jobs with benchmarks."""
        tenant_data = multi_tenant_setup["tenant_a"]

        # Create evaluation job with arc_easy benchmark
        job = create_evalhub_evaluation_job(
            host=evalhub_route.host,
            token=tenant_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_data["namespace"].name,
            model_url="http://test-model.tenant-a.svc.cluster.local:8000/v1",
            model_name="test-model/TestModel-1B",
            benchmarks=[
                {
                    "id": "arc_easy",
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        )

        assert "id" in job, "Evaluation job response missing 'id'"
        assert "status" in job, "Evaluation job response missing 'status'"
        assert "model" in job, "Evaluation job response missing 'model'"
        assert "benchmarks" in job, "Evaluation job response missing 'benchmarks'"

        # Verify job details
        assert job["model"]["name"] == "test-model/TestModel-1B", "Model name mismatch"
        assert len(job["benchmarks"]) == 1, "Expected 1 benchmark"
        assert job["benchmarks"][0]["id"] == "arc_easy", "Benchmark ID mismatch"

    def test_evaluation_jobs_are_tenant_isolated(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify evaluation jobs are isolated per tenant."""
        # Create jobs for both tenants
        tenant_a_data = multi_tenant_setup["tenant_a"]
        tenant_a_job = create_evalhub_evaluation_job(
            host=evalhub_route.host,
            token=tenant_a_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_a_data["namespace"].name,
            model_url="http://model-a.tenant-a.svc.cluster.local:8000/v1",
            model_name="tenant-a/model-a",
            benchmarks=[
                {
                    "id": "mmlu",
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        )

        tenant_b_data = multi_tenant_setup["tenant_b"]
        tenant_b_job = create_evalhub_evaluation_job(
            host=evalhub_route.host,
            token=tenant_b_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_b_data["namespace"].name,
            model_url="http://model-b.tenant-b.svc.cluster.local:8000/v1",
            model_name="tenant-b/model-b",
            benchmarks=[
                {
                    "id": "hellaswag",
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        )

        # Verify tenant-a can only see their own job
        tenant_a_jobs = list_evalhub_evaluation_jobs(
            host=evalhub_route.host,
            token=tenant_a_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_a_data["namespace"].name,
        )

        tenant_a_job_ids = [j["id"] for j in tenant_a_jobs.get("items", [])]
        assert tenant_a_job["id"] in tenant_a_job_ids, "Tenant A should see their own job"
        assert tenant_b_job["id"] not in tenant_a_job_ids, "Tenant A should NOT see Tenant B's job"

        # Verify tenant-b can only see their own job
        tenant_b_jobs = list_evalhub_evaluation_jobs(
            host=evalhub_route.host,
            token=tenant_b_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_b_data["namespace"].name,
        )

        tenant_b_job_ids = [j["id"] for j in tenant_b_jobs.get("items", [])]
        assert tenant_b_job["id"] in tenant_b_job_ids, "Tenant B should see their own job"
        assert tenant_a_job["id"] not in tenant_b_job_ids, "Tenant B should NOT see Tenant A's job"

    def test_get_evaluation_job_status(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify tenant can retrieve evaluation job status."""
        tenant_data = multi_tenant_setup["tenant_a"]

        # Create job
        job = create_evalhub_evaluation_job(
            host=evalhub_route.host,
            token=tenant_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_data["namespace"].name,
            model_url="http://status-test-model.tenant-a.svc.cluster.local:8000/v1",
            model_name="test/status-model",
            benchmarks=[
                {
                    "id": "truthfulqa",
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        )

        # Retrieve job status
        job_status = get_evalhub_evaluation_job(
            host=evalhub_route.host,
            token=tenant_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_data["namespace"].name,
            job_id=job["id"],
        )

        assert job_status["id"] == job["id"], "Job ID mismatch"
        assert "status" in job_status, "Job status missing"
        assert job_status["status"] in ["pending", "running", "completed", "failed"], (
            f"Unexpected job status: {job_status['status']}"
        )

    def test_tenant_cannot_access_other_tenant_job(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify tenant cannot access another tenant's evaluation job."""
        tenant_a_data = multi_tenant_setup["tenant_a"]
        tenant_b_data = multi_tenant_setup["tenant_b"]

        # Create job in tenant-a
        tenant_a_job = create_evalhub_evaluation_job(
            host=evalhub_route.host,
            token=tenant_a_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_a_data["namespace"].name,
            model_url="http://private-model.tenant-a.svc.cluster.local:8000/v1",
            model_name="tenant-a/private-model",
            benchmarks=[
                {
                    "id": "arc_challenge",
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        )

        # Tenant-b tries to access tenant-a's job
        with pytest.raises(requests.HTTPError) as exc_info:
            get_evalhub_evaluation_job(
                host=evalhub_route.host,
                token=tenant_b_data["token"],
                ca_bundle_file=evalhub_ca_bundle_file,
                tenant=tenant_b_data["namespace"].name,
                job_id=tenant_a_job["id"],
            )

        # Should get 404 or 403 when trying to access another tenant's job
        assert exc_info.value.response.status_code in [403, 404], (
            f"Expected 403/404 when accessing other tenant's job, got {exc_info.value.response.status_code}"
        )

    def test_multi_benchmark_evaluation_job(
        self,
        evalhub_route: Route,
        evalhub_ca_bundle_file: str,
        multi_tenant_setup: dict,
    ):
        """Verify tenant can create evaluation job with multiple benchmarks."""
        tenant_data = multi_tenant_setup["tenant_a"]

        # Create job with multiple benchmarks
        job = create_evalhub_evaluation_job(
            host=evalhub_route.host,
            token=tenant_data["token"],
            ca_bundle_file=evalhub_ca_bundle_file,
            tenant=tenant_data["namespace"].name,
            model_url="http://multi-bench-model.tenant-a.svc.cluster.local:8000/v1",
            model_name="test/multi-benchmark-model",
            benchmarks=[
                {
                    "id": "mmlu",
                    "provider_id": "lm_evaluation_harness",
                },
                {
                    "id": "arc_easy",
                    "provider_id": "lm_evaluation_harness",
                },
                {
                    "id": "hellaswag",
                    "provider_id": "lm_evaluation_harness",
                },
            ],
        )

        assert len(job["benchmarks"]) == 3, "Expected 3 benchmarks in the job"
        benchmark_ids = [b["id"] for b in job["benchmarks"]]
        assert "mmlu" in benchmark_ids, "MMLU benchmark not found"
        assert "arc_easy" in benchmark_ids, "ARC Easy benchmark not found"
        assert "hellaswag" in benchmark_ids, "HellaSwag benchmark not found"
