"""Positive API behaviour tests for EvalHub single-tenancy mode.

Two test classes:

  TestEvalHubSingleTenancyHealth
      Verifies the /api/v1/health endpoint is accessible without tenant context.

  TestEvalHubSingleTenancyAPIAccess
      Verifies a user SA with the evalhub-user Role can list providers,
      list collections, submit a job, and list submitted jobs — all using
      X-Tenant: {own_namespace}.

Negative tenant-header cases live in test_evalhub_single_tenancy_api_negative.py.

Run in isolation:
    pytest tests/ai_safety/evalhub/single_tenancy/test_evalhub_single_tenancy_api.py -m ai_safety
"""

import pytest
import requests
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route

from tests.ai_safety.evalhub.constants import (
    EVALHUB_COLLECTIONS_PATH,
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
    EVALHUB_JOBS_PATH,
    EVALHUB_PROVIDERS_PATH,
)
from tests.ai_safety.evalhub.utils import build_evalhub_job_payload, build_headers, submit_evalhub_job


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


@pytest.fixture(scope="class")
def evalhub_st_submitted_job_id(
    model_namespace: Namespace,
    evalhub_st_ready: None,
    evalhub_st_route: Route,
    evalhub_st_ca_bundle_file: str,
    evalhub_st_user_token: str,
) -> str:
    """Minimal evaluation job submitted once per API access test class."""
    payload = build_evalhub_job_payload(
        model_service_name="evalhub-st-emulator",
        tenant_namespace=model_namespace.name,
        job_name="evalhub-st-api-test-job",
    )
    data = submit_evalhub_job(
        host=evalhub_st_route.host,
        token=evalhub_st_user_token,
        ca_bundle_file=evalhub_st_ca_bundle_file,
        tenant=model_namespace.name,
        payload=payload,
    )
    job_id = (data.get("resource") or {}).get("id")
    assert job_id, f"Expected resource.id in 202 response body, got: {data}"
    return job_id


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "test-evalhub-st-api"})],
    indirect=True,
)
@pytest.mark.tier1
@pytest.mark.ai_safety
class TestEvalHubSingleTenancyAPIAccess:
    """A user SA bound to evalhub-user Role can access the API with X-Tenant: own namespace."""

    @pytest.mark.parametrize(
        ("api_path", "endpoint_label"),
        [
            pytest.param(EVALHUB_PROVIDERS_PATH, "providers", id="test_providers"),
            pytest.param(EVALHUB_COLLECTIONS_PATH, "collections", id="test_collections"),
        ],
    )
    def test_list_resource_with_own_namespace_tenant(
        self,
        api_path: str,
        endpoint_label: str,
        model_namespace: Namespace,
        evalhub_st_ready: None,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
    ) -> None:
        """Given: a user SA bound to evalhub-user Role in the workload namespace.

        When: GET a list endpoint with X-Tenant set to the caller's own namespace.

        Then: response is 200.
        """
        url = f"https://{evalhub_st_route.host}{api_path}"
        response = requests.get(
            url=url,
            headers=build_headers(token=evalhub_st_user_token, tenant=model_namespace.name),
            verify=evalhub_st_ca_bundle_file,
            timeout=10,
        )
        assert response.status_code == 200, (
            f"Expected 200 from {endpoint_label} endpoint, got {response.status_code}: {response.text}"
        )

    def test_submit_job_with_own_namespace_tenant(
        self,
        evalhub_st_submitted_job_id: str,
    ) -> None:
        """Given: a user SA bound to evalhub-user Role in the workload namespace.

        When: POST /api/v1/evaluations/jobs with X-Tenant set to the caller's own namespace.

        Then: response is 202 with resource.id in the body.
        """
        assert evalhub_st_submitted_job_id, "Expected non-empty job id from class-scoped submission fixture"

    def test_list_jobs_after_submit(
        self,
        model_namespace: Namespace,
        evalhub_st_route: Route,
        evalhub_st_ca_bundle_file: str,
        evalhub_st_user_token: str,
        evalhub_st_submitted_job_id: str,
    ) -> None:
        """Given: a user SA that has submitted a job in their own namespace.

        When: GET /api/v1/evaluations/jobs with X-Tenant set to the caller's own namespace.

        Then: response is 200 and the submitted job appears in items.
        """
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
        assert evalhub_st_submitted_job_id in job_ids, (
            f"Submitted job '{evalhub_st_submitted_job_id}' not found in jobs list: {job_ids}"
        )
