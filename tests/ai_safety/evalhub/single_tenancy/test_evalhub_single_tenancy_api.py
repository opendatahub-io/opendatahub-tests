"""API behaviour tests for EvalHub single-tenancy mode.

Three test classes:

  TestEvalHubSingleTenancyHealth
      Verifies the /api/v1/health endpoint is accessible without tenant context.

  TestEvalHubSingleTenancyAPIAccess
      Verifies a user SA with the evalhub-user Role can list providers,
      list collections, submit a job, and list submitted jobs — all using
      X-Tenant: {own_namespace}.

  TestEvalHubSingleTenancyCrossNamespaceRejection
      Verifies that requests with X-Tenant pointing to a different namespace
      (where the caller has no Role) are rejected, and that requests without
      an X-Tenant header are also rejected.

Run in isolation:
    pytest tests/ai_safety/evalhub/single_tenancy/test_evalhub_single_tenancy_api.py -m ai_safety
"""

from __future__ import annotations

from typing import Literal, TypedDict

import pytest
import requests
import structlog
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.ai_safety.evalhub.constants import (
    EVALHUB_COLLECTIONS_PATH,
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
    EVALHUB_JOBS_PATH,
    EVALHUB_PROVIDERS_PATH,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from tests.ai_safety.evalhub.utils import build_headers

LOGGER = structlog.get_logger(name=__name__)


class _JobBenchmarkParameters(TypedDict):
    num_examples: int
    tokenizer: Literal["google/flan-t5-small"]


class _JobBenchmark(TypedDict):
    id: Literal["arc_easy"]
    provider_id: Literal["lm_evaluation_harness"]
    parameters: _JobBenchmarkParameters


class _JobModel(TypedDict):
    url: str
    name: Literal["emulatedModel"]


class _JobPayload(TypedDict):
    name: str
    model: _JobModel
    benchmarks: list[_JobBenchmark]


def _minimal_job_payload(
    tenant_namespace: str,
    job_name: str = "evalhub-st-api-test-job",
) -> _JobPayload:
    """Minimal job payload that targets a non-existent model URL.

    The job will be accepted (202) and then fail when it tries to reach the
    model — but we only assert on the submission response code, not completion.
    """
    model_url = f"http://evalhub-st-emulator.{tenant_namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"
    return {
        "name": job_name,
        "model": {
            "url": model_url,
            "name": "emulatedModel",
        },
        "benchmarks": [
            {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "parameters": {
                    "num_examples": 1,
                    "tokenizer": "google/flan-t5-small",
                },
            }
        ],
    }


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-health"})],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyHealth:
    """Health endpoint is unauthenticated and always returns healthy."""

    def test_health_endpoint_returns_healthy(
        self,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
    ) -> None:
        """Given: a ready single-tenancy EvalHub instance with a public route.

        When: GET /api/v1/health is called without authentication.

        Then: response is 200 with status "healthy".
        """
        url = f"https://{evalhub_st_route.host}{EVALHUB_HEALTH_PATH}"
        response = requests.get(url=url, verify=evalhub_st_ca_bundle_file, timeout=10)
        assert response.status_code == 200, (
            f"Expected 200 from health endpoint, got {response.status_code}: {response.text}"
        )
        data = response.json()
        assert data.get("status") == EVALHUB_HEALTH_STATUS_HEALTHY, (
            f"Expected status='{EVALHUB_HEALTH_STATUS_HEALTHY}', got: {data}"
        )


# ---------------------------------------------------------------------------
# Authorised API access using own namespace as X-Tenant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-api"})],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyAPIAccess:
    """A user SA bound to evalhub-user Role can access the API with X-Tenant: own namespace."""

    def test_list_providers_with_own_namespace_tenant(
        self,
        model_namespace: Namespace,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
    ) -> None:
        """Given: a user SA bound to evalhub-user Role in the workload namespace.

        When: GET /api/v1/evaluations/providers with X-Tenant set to the caller's own namespace.

        Then: response is 200.
        """
        url = f"https://{evalhub_st_route.host}{EVALHUB_PROVIDERS_PATH}"
        response = requests.get(
            url=url,
            headers=build_headers(token=evalhub_st_user_token, tenant=model_namespace.name),
            verify=evalhub_st_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, (
            f"Expected 200 from providers endpoint, got {response.status_code}: {response.text}"
        )

    def test_list_collections_with_own_namespace_tenant(
        self,
        model_namespace: Namespace,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
    ) -> None:
        """Given: a user SA bound to evalhub-user Role in the workload namespace.

        When: GET /api/v1/evaluations/collections with X-Tenant set to the caller's own namespace.

        Then: response is 200.
        """
        url = f"https://{evalhub_st_route.host}{EVALHUB_COLLECTIONS_PATH}"
        response = requests.get(
            url=url,
            headers=build_headers(token=evalhub_st_user_token, tenant=model_namespace.name),
            verify=evalhub_st_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, (
            f"Expected 200 from collections endpoint, got {response.status_code}: {response.text}"
        )

    def test_submit_job_with_own_namespace_tenant(
        self,
        model_namespace: Namespace,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
    ) -> None:
        """Given: a user SA bound to evalhub-user Role in the workload namespace.

        When: POST /api/v1/evaluations/jobs with X-Tenant set to the caller's own namespace.

        Then: response is 202 with resource.id in the body.
        """
        url = f"https://{evalhub_st_route.host}{EVALHUB_JOBS_PATH}"
        payload = _minimal_job_payload(tenant_namespace=model_namespace.name)
        response = requests.post(
            url=url,
            headers=build_headers(token=evalhub_st_user_token, tenant=model_namespace.name),
            json=payload,
            verify=evalhub_st_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 202, f"Expected 202 from job submit, got {response.status_code}: {response.text}"
        data = response.json()
        job_id = (data.get("resource") or {}).get("id")
        assert job_id, f"Expected resource.id in 202 response body, got: {data}"

    def test_list_jobs_after_submit(
        self,
        model_namespace: Namespace,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
    ) -> None:
        """Given: a user SA that has submitted a job in their own namespace.

        When: GET /api/v1/evaluations/jobs with X-Tenant set to the caller's own namespace.

        Then: response is 200 and the submitted job appears in items.
        """
        # Submit a job first so the list is non-empty
        post_url = f"https://{evalhub_st_route.host}{EVALHUB_JOBS_PATH}"
        payload = _minimal_job_payload(
            tenant_namespace=model_namespace.name,
            job_name="evalhub-st-list-check-job",
        )
        post_resp = requests.post(
            url=post_url,
            headers=build_headers(token=evalhub_st_user_token, tenant=model_namespace.name),
            json=payload,
            verify=evalhub_st_ca_bundle_file,
            timeout=30,
        )
        assert post_resp.status_code == 202, (
            f"Pre-condition: job submission failed with {post_resp.status_code}: {post_resp.text}"
        )
        submitted_id = (post_resp.json().get("resource") or {}).get("id")

        list_url = f"https://{evalhub_st_route.host}{EVALHUB_JOBS_PATH}"
        list_resp = requests.get(
            url=list_url,
            headers=build_headers(token=evalhub_st_user_token, tenant=model_namespace.name),
            verify=evalhub_st_ca_bundle_file,
            timeout=10,
        )
        assert list_resp.status_code == 200, (
            f"Expected 200 from jobs list, got {list_resp.status_code}: {list_resp.text}"
        )
        data = list_resp.json()
        items = data.get("items") or []
        job_ids = [(item.get("resource") or {}).get("id") for item in items]
        assert submitted_id in job_ids, f"Submitted job '{submitted_id}' not found in jobs list: {job_ids}"


# ---------------------------------------------------------------------------
# Cross-namespace and missing-tenant-header rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-xns"})],
    indirect=True,
)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyCrossNamespaceRejection:
    """Requests with an unauthorized X-Tenant or no X-Tenant header are rejected.

    In single-tenancy mode, kube-rbac-proxy still performs a SubjectAccessReview
    against the X-Tenant namespace. If the caller's SA has no Role in that namespace,
    the SAR returns NoOpinion → 400 unable_to_authorize_request. If no X-Tenant
    header is sent at all, the server returns 400 immediately.
    """

    def test_cross_namespace_x_tenant_rejected(
        self,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
        second_namespace: Namespace,
    ) -> None:
        """Given: a user SA with no Role in a second namespace.

        When: POST /api/v1/evaluations/jobs with X-Tenant pointing to that namespace.

        Then: response is 400 or 403 (kube-rbac-proxy Forbidden in single-tenancy mode).
        """
        payload = _minimal_job_payload(tenant_namespace=second_namespace.name)
        response = requests.post(
            url=f"https://{evalhub_st_route.host}{EVALHUB_JOBS_PATH}",
            headers=build_headers(token=evalhub_st_user_token, tenant=second_namespace.name),
            json=payload,
            verify=evalhub_st_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code in (400, 403), (
            f"Expected 400 or 403 for cross-tenant POST, got {response.status_code}: {response.text}"
        )

    def test_missing_x_tenant_header_rejected(
        self,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
        model_namespace: Namespace,
    ) -> None:
        """Given: an authenticated user SA in single-tenancy mode.

        When: POST /api/v1/evaluations/jobs without an X-Tenant header.

        Then: response is 400 Bad Request.
        """
        payload = _minimal_job_payload(tenant_namespace=model_namespace.name)
        response = requests.post(
            url=f"https://{evalhub_st_route.host}{EVALHUB_JOBS_PATH}",
            headers=build_headers(token=evalhub_st_user_token, tenant=None),
            json=payload,
            verify=evalhub_st_ca_bundle_file,
            timeout=30,
        )
        assert response.status_code == 400, (
            f"Expected 400 Bad Request for missing X-Tenant, got {response.status_code}: {response.text}"
        )

    def test_get_providers_cross_namespace_rejected(
        self,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
        second_namespace: Namespace,
    ) -> None:
        """Given: a user SA with no Role in a second namespace.

        When: GET /api/v1/evaluations/providers with X-Tenant pointing to that namespace.

        Then: response is 400 or 403.
        """
        response = requests.get(
            url=f"https://{evalhub_st_route.host}{EVALHUB_PROVIDERS_PATH}",
            headers=build_headers(token=evalhub_st_user_token, tenant=second_namespace.name),
            verify=evalhub_st_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code in (400, 403), (
            f"Expected 400 or 403 for cross-tenant GET providers, got {response.status_code}: {response.text}"
        )

    def test_get_providers_no_tenant_rejected(
        self,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
    ) -> None:
        """Given: an authenticated user SA in single-tenancy mode.

        When: GET /api/v1/evaluations/providers without an X-Tenant header.

        Then: response is 400 Bad Request.
        """
        response = requests.get(
            url=f"https://{evalhub_st_route.host}{EVALHUB_PROVIDERS_PATH}",
            headers=build_headers(token=evalhub_st_user_token, tenant=None),
            verify=evalhub_st_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 400, (
            f"Expected 400 Bad Request for missing X-Tenant, got {response.status_code}: {response.text}"
        )
