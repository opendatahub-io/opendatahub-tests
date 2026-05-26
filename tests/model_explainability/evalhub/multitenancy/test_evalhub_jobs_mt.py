import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.model_explainability.evalhub.constants import EVALHUB_JOBS_PATH
from tests.model_explainability.evalhub.utils import (
    build_evalhub_job_payload,
    build_headers,
    list_evalhub_jobs,
    submit_evalhub_job,
    validate_evalhub_job_completed,
    validate_evalhub_post_denied,
    validate_evalhub_post_no_tenant,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
    wait_for_evalhub_job,
)


@pytest.mark.parametrize(
    "model_namespace",
    [
        pytest.param(
            {"name": "test-evalhub-jobs-mt"},
        ),
    ],
    indirect=True,
)
@pytest.mark.model_explainability
class TestEvalHubJobsMT:
    """Multi-tenancy tests for EvalHub job submission.

    Submits an lm_evaluation_harness job (arc_easy, 10 examples)
    against the vLLM emulator. Tests authorization, job lifecycle,
    and result validation.
    """

    def test_job_submit_authorized_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """User with evaluations-create RBAC in tenant-a can submit a job."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        resource = data.get("resource", {})
        assert resource.get("id"), f"Job response missing 'resource.id': {data}"
        assert data.get("status", {}).get("state") == "pending", (
            f"Expected job state 'pending', got: {data.get('status')}"
        )

    def test_job_completion_and_results(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Submit a job and wait for it to complete with benchmark results."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        job_result = wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )
        validate_evalhub_job_completed(job_data=job_result)

    def test_job_pod_reaches_succeeded(
        self,
        admin_client: DynamicClient,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """After job completion, the K8s pod should be in Succeeded state."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]

        # Wait for EvalHub API to report completion
        wait_for_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            timeout=600,
        )

        # Find the job pod by label
        pods = list(
            Pod.get(
                client=admin_client,
                namespace=tenant_a_namespace.name,
                label_selector=f"app=evalhub,component=evaluation-job,job_id={job_id}",
            )
        )
        assert len(pods) >= 1, f"Expected at least 1 pod for job {job_id}, found {len(pods)}"

        # The EvalHub API reports "completed" via sidecar callback before the
        # pod containers actually exit. Wait for the pod to reach Succeeded.
        pod = pods[0]
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=120)

    def test_job_submit_cross_tenant_denied(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """User with RBAC in tenant-a is denied job submission for tenant-b."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        validate_evalhub_post_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_JOBS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
            payload=payload,
        )

    def test_job_submit_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Job submission without X-Tenant header is rejected with 400."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        validate_evalhub_post_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_JOBS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            payload=payload,
        )

    def test_job_list_authorized_tenant(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """After submitting a job, listing jobs for tenant-a shows it."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submitted = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = submitted["resource"]["id"]

        data = list_evalhub_jobs(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
        )
        items = data.get("items", [])
        job_ids = [item.get("resource", {}).get("id") for item in items]
        assert job_id in job_ids, f"Submitted job {job_id} not found in job list: {job_ids}"

    def test_job_list_cross_tenant_denied(
        self,
        tenant_a_token: str,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Listing jobs for tenant-b is denied for user with tenant-a access."""
        validate_evalhub_request_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_JOBS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
        )

    def test_job_list_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Listing jobs without X-Tenant header is rejected with 400."""
        validate_evalhub_request_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=EVALHUB_JOBS_PATH,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
        )

    def test_job_submit_missing_name_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Job submission without a name field is rejected with 400."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        del payload["name"]
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 for missing name, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "request_validation_failed"

    def test_job_submit_missing_model_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Job submission without a model field is rejected with 400."""
        payload = {
            "name": "missing-model-test",
            "benchmarks": [
                {
                    "id": "arc_easy",
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        }
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 for missing model, got {response.status_code}: {response.text}"
        )

    @pytest.mark.parametrize(
        "benchmarks_value",
        [
            pytest.param([], id="empty-array"),
            pytest.param(None, id="absent"),
        ],
    )
    def test_job_submit_missing_benchmarks_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        benchmarks_value: list | None,
    ) -> None:
        """Job submission with empty or absent benchmarks is rejected with 400."""
        payload: dict = {
            "name": "missing-benchmarks-test",
            "model": {"url": "http://test.com", "name": "test"},
        }
        if benchmarks_value is not None:
            payload["benchmarks"] = benchmarks_value
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 for missing benchmarks, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "request_validation_failed"

    def test_job_submit_invalid_json_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Job submission with invalid JSON body is rejected with 400."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        headers["Content-Type"] = "application/json"
        response = requests.post(
            url=url,
            headers=headers,
            data="{ not valid json",
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 for invalid JSON, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "invalid_json_request"

    def test_job_submit_invalid_provider_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Job submission with a non-existent provider_id is rejected with 404."""
        payload = {
            "name": "invalid-provider-test",
            "model": {"url": "http://test.com", "name": "test"},
            "benchmarks": [
                {
                    "id": "arc_easy",
                    "provider_id": "nonexistent_provider",
                }
            ],
        }
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 404, (
            f"Expected 404 for invalid provider, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "resource_not_found"

    def test_job_submit_invalid_benchmark_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Job submission with a non-existent benchmark id is rejected with 400."""
        payload = {
            "name": "invalid-benchmark-test",
            "model": {"url": "http://test.com", "name": "test"},
            "benchmarks": [
                {
                    "id": "nonexistent_benchmark_xyz",
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        }
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 for invalid benchmark, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "resource_does_not_exist"

    def test_job_submit_collection_and_benchmarks_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Job submission with both collection and benchmarks is rejected with 400."""
        payload = {
            "name": "collection-and-benchmarks-test",
            "model": {"url": "http://test.com", "name": "test"},
            "collection": {"id": "any_collection_id"},
            "benchmarks": [
                {
                    "id": "arc_easy",
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        }
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 for collection+benchmarks, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "request_validation_failed"

    def test_job_submit_missing_benchmark_id_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Job submission with a benchmark missing 'id' is rejected with 400."""
        payload = {
            "name": "missing-benchmark-id-test",
            "model": {"url": "http://test.com", "name": "test"},
            "benchmarks": [
                {
                    "provider_id": "lm_evaluation_harness",
                }
            ],
        }
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 for missing benchmark id, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "request_validation_failed"

    def test_job_submit_missing_benchmark_provider_id_rejected(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Job submission with a benchmark missing 'provider_id' is rejected with 400."""
        payload = {
            "name": "missing-provider-id-test",
            "model": {"url": "http://test.com", "name": "test"},
            "benchmarks": [
                {
                    "id": "arc_easy",
                }
            ],
        }
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"
        response = requests.post(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json=payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 for missing benchmark provider_id, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "request_validation_failed"

    def test_get_job_nonexistent_id_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """GET for a non-existent job id returns 404."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}/00000000-0000-0000-0000-000000000000"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 404, (
            f"Expected 404 for non-existent job, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "resource_not_found"

    def test_delete_job_nonexistent_id_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """DELETE for a non-existent job id returns 404."""
        url = (
            f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}/00000000-0000-0000-0000-000000000000?hard_delete=true"
        )
        response = requests.delete(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 404, (
            f"Expected 404 for non-existent job delete, got {response.status_code}: {response.text}"
        )
        assert response.json().get("message_code") == "resource_not_found"

    @pytest.mark.parametrize(
        "query,expected_code",
        [
            pytest.param("limit=-1", "query_parameter_invalid", id="negative-limit"),
            pytest.param("limit=invalid", "query_parameter_invalid", id="non-numeric-limit"),
            pytest.param("offset=not-a-number", "query_parameter_invalid", id="non-numeric-offset"),
        ],
    )
    def test_list_jobs_invalid_query_params(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        query: str,
        expected_code: str,
    ) -> None:
        """List jobs with invalid limit or offset returns 400."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}?{query}"
        response = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 400, f"Expected 400 for '{query}', got {response.status_code}: {response.text}"
        assert response.json().get("message_code") == expected_code

    def test_list_jobs_pagination(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """List jobs with limit returns paginated results and a next href."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)

        # Submit 3 jobs to ensure pagination is possible
        job_ids = []
        for _ in range(3):
            payload = build_evalhub_job_payload(
                model_service_name=evalhub_vllm_emulator_service.name,
                tenant_namespace=tenant_a_namespace.name,
            )
            data = submit_evalhub_job(
                host=evalhub_mt_route.host,
                token=tenant_a_token,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
                tenant=tenant_a_namespace.name,
                payload=payload,
            )
            job_ids.append(data["resource"]["id"])

        # First page: limit=2
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}?limit=2"
        resp = requests.get(
            url=url,
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["items"]) == 2
        assert body["total_count"] >= 3
        assert "next" in body, "Expected 'next' pagination link"

        # Second page via next href
        next_href = body["next"]["href"]
        if next_href.startswith("http"):
            from urllib.parse import urlparse

            parsed = urlparse(url=next_href)
            assert parsed.hostname == evalhub_mt_route.host, (
                f"next href points to unexpected host: {parsed.hostname} != {evalhub_mt_route.host}"
            )
            next_url = next_href
        else:
            next_url = f"https://{evalhub_mt_route.host}{next_href}"
        resp2 = requests.get(
            url=next_url,
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp2.status_code == 200
        body2 = resp2.json()
        assert len(body2["items"]) >= 1

    def test_list_jobs_filtered_by_status(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """List jobs filtered by status=pending returns only pending jobs."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)

        # Submit a job (starts as pending)
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
        )
        submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )

        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}?status=pending&limit=10"
        resp = requests.get(
            url=url,
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_count"] >= 1
        for item in body["items"]:
            assert item["status"]["state"] == "pending", (
                f"Expected all jobs to be pending, got {item['status']['state']}"
            )

    def test_list_jobs_empty_when_no_match(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """List jobs with a non-matching owner filter returns empty results."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}?owner=nonexistent-user-xyz&limit=10"
        resp = requests.get(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 0

    def test_list_jobs_with_search_filters(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """List jobs combining status, tags, and name filters."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base_url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"

        # Submit a job with custom tags
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="filter-test-job",
        )
        payload["tags"] = ["filter-tag-a", "filter-tag-b"]
        submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )

        # Filter by name
        resp = requests.get(
            url=f"{base_url}?name=filter-test-job&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] >= 1

        # Filter by tag
        resp = requests.get(
            url=f"{base_url}?tags=filter-tag-a&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] >= 1

        # Filter by non-matching tag
        resp = requests.get(
            url=f"{base_url}?tags=nonexistent-tag-xyz&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 0

        # Combined: name + status + tag
        resp = requests.get(
            url=f"{base_url}?name=filter-test-job&status=pending&tags=filter-tag-a&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] >= 1

    def test_list_jobs_by_tags_and_or_semantics(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """List jobs with AND (comma) and OR (pipe) tag semantics."""
        headers = build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name)
        base_url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}"

        # Submit job with tags [tag-x, tag-y]
        payload1 = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="tag-test-1",
        )
        payload1["tags"] = ["tag-x", "tag-y"]
        submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload1,
        )

        # Submit job with tags [tag-x, tag-z]
        payload2 = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="tag-test-2",
        )
        payload2["tags"] = ["tag-x", "tag-z"]
        submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload2,
        )

        # AND: tag-y,tag-z → only jobs with both (none)
        resp = requests.get(
            url=f"{base_url}?tags=tag-y,tag-z&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] == 0

        # AND: tag-x,tag-y → only job 1
        resp = requests.get(
            url=f"{base_url}?tags=tag-x,tag-y&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] >= 1

        # OR: tag-y|tag-z → jobs with either tag
        resp = requests.get(
            url=f"{base_url}?tags=tag-y|tag-z&limit=10",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200
        assert resp.json()["total_count"] >= 2

    # --- Unsupported methods ---

    @pytest.mark.parametrize(
        "method,path_suffix",
        [
            pytest.param("put", "", id="PUT-jobs"),
            pytest.param("post", "/unknown-id", id="POST-jobs-id"),
            pytest.param("get", "/unknown-id/events", id="GET-jobs-id-events"),
        ],
    )
    def test_evaluation_endpoints_reject_unsupported_methods(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        method: str,
        path_suffix: str,
    ) -> None:
        """Evaluation endpoints reject unsupported HTTP methods with 405."""
        url = f"https://{evalhub_mt_route.host}{EVALHUB_JOBS_PATH}{path_suffix}"
        response = getattr(requests, method)(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant_a_namespace.name),
            json={},
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 405, (
            f"Expected 405 for {method.upper()} {path_suffix}, got {response.status_code}"
        )

    # --- Event-based status update tests ---

    def _submit_pending_job(
        self,
        host: str,
        token: str,
        ca_bundle_file: str,
        tenant: str,
        emulator_service: Service,
        tenant_namespace: Namespace,
    ) -> str:
        """Submit a job and return its id."""
        payload = build_evalhub_job_payload(
            model_service_name=emulator_service.name,
            tenant_namespace=tenant_namespace.name,
        )
        data = submit_evalhub_job(
            host=host,
            token=token,
            ca_bundle_file=ca_bundle_file,
            tenant=tenant,
            payload=payload,
        )
        return data["resource"]["id"]

    def _post_event(
        self,
        host: str,
        token: str,
        ca_bundle_file: str,
        tenant: str,
        job_id: str,
        event: dict,
    ) -> requests.Response:
        """POST a benchmark status event to a job."""
        url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}/events"
        return requests.post(
            url=url,
            headers=build_headers(token=token, tenant=tenant),
            json=event,
            verify=ca_bundle_file,
            timeout=30,
        )

    def _get_job(
        self,
        host: str,
        token: str,
        ca_bundle_file: str,
        tenant: str,
        job_id: str,
    ) -> dict:
        """GET a job by id and return the JSON body."""
        url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
        resp = requests.get(
            url=url,
            headers=build_headers(token=token, tenant=tenant),
            verify=ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 200, f"Expected 200 for GET job {job_id}, got {resp.status_code}: {resp.text}"
        return resp.json()

    def test_update_job_status_running_to_completed(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Post running then completed events; job state transitions accordingly."""
        host = evalhub_mt_route.host
        tenant = tenant_a_namespace.name
        job_id = self._submit_pending_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            emulator_service=evalhub_vllm_emulator_service,
            tenant_namespace=tenant_a_namespace,
        )

        # Running event
        running_event = {
            "benchmark_status_event": {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "status": "running",
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=running_event,
        )
        assert resp.status_code == 204

        job = self._get_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
        )
        assert job["status"]["state"] == "running"

        # Completed event
        completed_event = {
            "benchmark_status_event": {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "status": "completed",
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=completed_event,
        )
        assert resp.status_code == 204

        job = self._get_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
        )
        assert job["status"]["state"] == "completed"
        assert job["status"]["benchmarks"][0]["status"] == "completed"
        assert job["status"]["benchmarks"][0]["id"] == "arc_easy"

    def test_update_job_status_invalid_payload(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Post an event with missing status field returns 400."""
        host = evalhub_mt_route.host
        tenant = tenant_a_namespace.name
        job_id = self._submit_pending_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            emulator_service=evalhub_vllm_emulator_service,
            tenant_namespace=tenant_a_namespace,
        )

        invalid_event = {
            "benchmark_status_event": {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=invalid_event,
        )
        assert resp.status_code == 400
        assert resp.json().get("message_code") == "request_validation_failed"

    def test_update_job_status_missing_provider_id(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Post an event with missing provider_id returns 400."""
        host = evalhub_mt_route.host
        tenant = tenant_a_namespace.name
        job_id = self._submit_pending_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            emulator_service=evalhub_vllm_emulator_service,
            tenant_namespace=tenant_a_namespace,
        )

        missing_provider_event = {
            "benchmark_status_event": {
                "id": "arc_easy",
                "status": "running",
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=missing_provider_event,
        )
        assert resp.status_code == 400
        assert resp.json().get("message_code") == "request_validation_failed"

    def test_update_job_status_unknown_id(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Post an event to a non-existent job id returns 404."""
        event = {
            "benchmark_status_event": {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "status": "running",
            }
        }
        resp = self._post_event(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id="unknown-id",
            event=event,
        )
        assert resp.status_code == 404

    # --- Cancel / soft-delete tests ---

    def test_cancel_running_job(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """Cancel a running job via soft DELETE; state becomes cancelled."""
        host = evalhub_mt_route.host
        tenant = tenant_a_namespace.name
        job_id = self._submit_pending_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            emulator_service=evalhub_vllm_emulator_service,
            tenant_namespace=tenant_a_namespace,
        )

        # Set to running
        running_event = {
            "benchmark_status_event": {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "status": "running",
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=running_event,
        )
        assert resp.status_code == 204

        # Soft delete (cancel)
        url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
        resp = requests.delete(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 204

        job = self._get_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
        )
        assert job["status"]["state"] == "cancelled"
        assert job["status"]["benchmarks"][0]["status"] == "cancelled"

    def test_cancel_job_invalid_hard_delete_query(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
    ) -> None:
        """DELETE with hard_delete=foo (non-boolean) returns 400."""
        host = evalhub_mt_route.host
        tenant = tenant_a_namespace.name
        job_id = self._submit_pending_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            emulator_service=evalhub_vllm_emulator_service,
            tenant_namespace=tenant_a_namespace,
        )

        url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}?hard_delete=foo"
        resp = requests.delete(
            url=url,
            headers=build_headers(token=tenant_a_token, tenant=tenant),
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )
        assert resp.status_code == 400
        assert resp.json().get("message_code") == "query_parameter_invalid"

    # --- Pass criteria tests ---

    def test_pass_criteria_job_aggregate_results(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Submit a job with pass_criteria, post benchmark events, verify scores and pass."""
        host = evalhub_mt_route.host
        tenant = tenant_a_namespace.name
        headers = build_headers(token=tenant_a_token, tenant=tenant)

        # Submit job with two benchmarks and pass_criteria
        payload = {
            "name": "pass-criteria-test-job",
            "model": {"url": "http://test.com", "name": "test"},
            "pass_criteria": {"threshold": 0.7},
            "benchmarks": [
                {
                    "id": "arc_easy",
                    "provider_id": "lm_evaluation_harness",
                    "primary_score": {"metric": "accuracy", "lower_is_better": False},
                    "pass_criteria": {"threshold": 0.3},
                    "weight": 0.6,
                },
                {
                    "id": "AraDiCE_boolq_lev",
                    "provider_id": "lm_evaluation_harness",
                    "primary_score": {"metric": "toxicity_rate", "lower_is_better": True},
                    "pass_criteria": {"threshold": 0.3},
                    "weight": 0.4,
                },
            ],
        }
        url = f"https://{host}{EVALHUB_JOBS_PATH}"
        resp = requests.post(url=url, headers=headers, json=payload, verify=evalhub_mt_ca_bundle_file, timeout=30)
        assert resp.status_code == 202
        job_id = resp.json()["resource"]["id"]

        # Benchmark 1 completed with accuracy=0.95
        b1_event = {
            "benchmark_status_event": {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "status": "completed",
                "metrics": {"accuracy": 0.95},
                "started_at": "2026-01-12T10:45:32Z",
                "completed_at": "2026-01-12T10:47:12Z",
                "duration_seconds": 100,
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=b1_event,
        )
        assert resp.status_code == 204

        # Benchmark 2 completed with toxicity_rate=0.1
        b2_event = {
            "benchmark_status_event": {
                "id": "AraDiCE_boolq_lev",
                "provider_id": "lm_evaluation_harness",
                "benchmark_index": 1,
                "status": "completed",
                "metrics": {"toxicity_rate": 0.1},
                "started_at": "2026-01-12T10:45:32Z",
                "completed_at": "2026-01-12T10:47:12Z",
                "duration_seconds": 100,
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=b2_event,
        )
        assert resp.status_code == 204

        job = self._get_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
        )
        assert job["status"]["state"] == "completed"

        # Verify per-benchmark pass
        results = job["results"]
        b1_result = next(b for b in results["benchmarks"] if b["id"] == "arc_easy")
        assert b1_result["test"]["pass"] is True
        b2_result = next(b for b in results["benchmarks"] if b["id"] == "AraDiCE_boolq_lev")
        assert b2_result["test"]["pass"] is True

        # Verify aggregate pass
        assert results["test"]["pass"] is True

    def test_job_submission_with_user_provider_benchmarks(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Create a user provider, submit a job referencing its benchmarks, verify accepted."""
        host = evalhub_mt_route.host
        tenant = tenant_a_namespace.name
        headers = build_headers(token=tenant_a_token, tenant=tenant)

        # Create a user provider with benchmarks
        provider_payload = {
            "name": "Provider for user-provider benchmark test",
            "description": "Two benchmarks with primary_score and pass_criteria",
            "benchmarks": [
                {
                    "id": "up_b1",
                    "name": "up_b1",
                    "primary_score": {"metric": "accuracy", "lower_is_better": False},
                    "pass_criteria": {"threshold": 0.5},
                },
                {
                    "id": "up_b2",
                    "name": "up_b2",
                    "primary_score": {"metric": "f1", "lower_is_better": False},
                    "pass_criteria": {"threshold": 0.6},
                },
            ],
        }
        resp = requests.post(
            url=f"https://{host}/api/v1/evaluations/providers",
            headers=headers,
            json=provider_payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 201
        provider_id = resp.json()["resource"]["id"]

        # Submit job with benchmarks referencing the user provider
        job_payload = {
            "name": "user-provider-benchmark-test",
            "model": {"url": "http://test.com", "name": "test"},
            "benchmarks": [
                {"id": "up_b1", "provider_id": provider_id},
                {"id": "up_b2", "provider_id": provider_id},
            ],
        }
        resp = requests.post(
            url=f"https://{host}{EVALHUB_JOBS_PATH}",
            headers=headers,
            json=job_payload,
            verify=evalhub_mt_ca_bundle_file,
            timeout=30,
        )
        assert resp.status_code == 202
        job_id = resp.json()["resource"]["id"]

        # Verify job references the correct provider and benchmarks
        job = self._get_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
        )

        benchmarks = job["status"]["benchmarks"]
        assert len(benchmarks) == 2
        b1 = next(b for b in benchmarks if b["id"] == "up_b1")
        b2 = next(b for b in benchmarks if b["id"] == "up_b2")
        assert b1["provider_id"] == provider_id
        assert b2["provider_id"] == provider_id

        # Cleanup
        requests.delete(
            url=f"https://{host}/api/v1/evaluations/providers/{provider_id}",
            headers=headers,
            verify=evalhub_mt_ca_bundle_file,
            timeout=10,
        )

    def test_partially_failed_job(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """One benchmark completes, one fails; job state is partially_failed."""
        host = evalhub_mt_route.host
        tenant = tenant_a_namespace.name
        headers = build_headers(token=tenant_a_token, tenant=tenant)

        # Submit job with two benchmarks
        payload = {
            "name": "partially-failed-test-job",
            "model": {"url": "http://test.com", "name": "test"},
            "benchmarks": [
                {
                    "id": "arc_easy",
                    "provider_id": "lm_evaluation_harness",
                    "primary_score": {"metric": "accuracy", "lower_is_better": False},
                    "weight": 0.6,
                },
                {
                    "id": "AraDiCE_boolq_lev",
                    "provider_id": "lm_evaluation_harness",
                    "primary_score": {"metric": "toxicity_rate", "lower_is_better": True},
                    "weight": 0.4,
                },
            ],
        }
        url = f"https://{host}{EVALHUB_JOBS_PATH}"
        resp = requests.post(url=url, headers=headers, json=payload, verify=evalhub_mt_ca_bundle_file, timeout=30)
        assert resp.status_code == 202
        job_id = resp.json()["resource"]["id"]

        # Benchmark 1 completes
        b1_event = {
            "benchmark_status_event": {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "status": "completed",
                "metrics": {"accuracy": 0.95},
                "started_at": "2026-01-12T10:45:32Z",
                "completed_at": "2026-01-12T10:47:12Z",
                "duration_seconds": 100,
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=b1_event,
        )
        assert resp.status_code == 204

        # Benchmark 2 fails
        b2_event = {
            "benchmark_status_event": {
                "id": "AraDiCE_boolq_lev",
                "provider_id": "lm_evaluation_harness",
                "benchmark_index": 1,
                "status": "failed",
                "error_message": {
                    "message": "Benchmark run failed",
                    "message_code": "benchmark_failed",
                },
                "started_at": "2026-01-12T10:45:32Z",
                "completed_at": "2026-01-12T10:47:12Z",
            }
        }
        resp = self._post_event(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
            event=b2_event,
        )
        assert resp.status_code == 204

        job = self._get_job(
            host=host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
        )
        assert job["status"]["state"] == "partially_failed"
        assert job["status"]["benchmarks"][0]["status"] == "completed"
        assert job["status"]["benchmarks"][0]["id"] == "arc_easy"
        assert job["status"]["benchmarks"][1]["status"] == "failed"
        assert job["status"]["benchmarks"][1]["id"] == "AraDiCE_boolq_lev"
