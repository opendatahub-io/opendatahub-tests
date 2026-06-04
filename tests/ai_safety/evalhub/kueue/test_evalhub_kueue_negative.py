"""Negative tests for EvalHub Kueue integration.

This module contains negative test cases that validate error handling and
edge cases for EvalHub when integrated with Kueue admission control.
"""

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.utils import build_evalhub_job_payload, get_evalhub_job_http
from utilities.kueue_utils import LocalQueue, Workload


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueNegative:
    """Negative tests for EvalHub Kueue integration."""

    def test_nonexistent_queue_name(
        self,
        evalhub_job_with_nonexistent_queue: dict,
    ) -> None:
        """TC-NEG-001: Verify error when submitting job with non-existent queue name.

        Given a Kueue-enabled cluster with no LocalQueue named 'nonexistent-queue',
        When a job is submitted referencing that queue,
        Then the job is accepted but shows admission failure or pending state.
        """
        job_id = evalhub_job_with_nonexistent_queue["job_id"]

        # Verify job status reflects the invalid queue
        status_response = get_evalhub_job_http(
            host=evalhub_job_with_nonexistent_queue["host"],
            token=evalhub_job_with_nonexistent_queue["token"],
            ca_bundle_file=evalhub_job_with_nonexistent_queue["ca_bundle_file"],
            tenant=evalhub_job_with_nonexistent_queue["tenant"],
            job_id=job_id,
        )
        status_response.raise_for_status()
        status_data = status_response.json()

        state = status_data.get("status", {}).get("state")
        assert state in ("pending", "failed"), f"Job with invalid queue should be pending or failed, got {state}"

    def test_submit_without_queue_spec(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_job_without_queue_spec: dict,
    ) -> None:
        """TC-NEG-002: Verify job without queue spec runs without Kueue (backwards compatibility).

        Given EvalHub deployed with or without Kueue,
        When a job is submitted without the queue field,
        Then the job is accepted (202) and runs without Kueue management.
        """
        # Verify no Kueue Workload was created for this job
        workloads = list(Workload.get(client=admin_client, namespace=evalhub_kueue_namespace.name))
        job_workloads = [
            wl for wl in workloads if wl.instance.get("metadata", {}).get("name", "").startswith("tc-neg-002")
        ]
        assert len(job_workloads) == 0, "No Workload should be created for job without queue spec"

    @pytest.mark.parametrize(
        "test_case,expected_status,method,use_invalid_token,job_id",
        [
            ("TC-NEG-003: unauthorized POST", 401, "POST", True, None),
            ("TC-NEG-005: GET nonexistent job", 404, "GET", False, "00000000-0000-0000-0000-000000000000"),
        ],
        ids=["unauthorized_401", "nonexistent_job_404"],
    )
    def test_error_responses(
        self,
        test_case: str,
        expected_status: int,
        method: str,
        use_invalid_token: bool,
        job_id: str | None,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
        evalhub_kueue_user_token: str,
    ) -> None:
        """Parameterized test for HTTP error responses.

        Tests both unauthorized access (401) and non-existent resource (404) scenarios.
        """
        if method == "POST":
            # Test unauthorized POST request
            payload = build_evalhub_job_payload(
                model_service_name=evalhub_kueue_vllm_service.name,
                tenant_namespace=evalhub_kueue_namespace.name,
                job_name="tc-neg-003-unauth",
            )
            payload["queue"] = {"kind": "kueue", "name": evalhub_kueue_multi_job_local_queue.name}

            url = f"https://{evalhub_kueue_route.host}/api/v1/evaluations/jobs"
            headers = {
                "Authorization": "Bearer invalid-token-12345",
                "X-Tenant": evalhub_kueue_namespace.name,
                "Content-Type": "application/json",
            }

            response = requests.post(
                url=url,
                headers=headers,
                json=payload,
                verify=evalhub_kueue_ca_bundle_file,
                timeout=30,
            )
        else:
            # Test GET for non-existent job
            response = get_evalhub_job_http(
                host=evalhub_kueue_route.host,
                token=evalhub_kueue_user_token,
                ca_bundle_file=evalhub_kueue_ca_bundle_file,
                tenant=evalhub_kueue_namespace.name,
                job_id=job_id,
            )

        assert response.status_code == expected_status, (
            f"{test_case}: Expected {expected_status}, got {response.status_code}: {response.text}"
        )

    def test_forbidden_cross_tenant_access(
        self,
        evalhub_job_for_cross_tenant_test: dict,
    ) -> None:
        """TC-NEG-004: Verify forbidden request returns 400/403 for cross-tenant access.

        Given a valid user with access to one namespace,
        When the user attempts to access a different tenant namespace,
        Then the API returns 400 or 403 Forbidden.
        """
        job_id = evalhub_job_for_cross_tenant_test["job_id"]

        # Try to access the job from a different (non-existent) tenant
        url = f"https://{evalhub_job_for_cross_tenant_test['host']}/api/v1/evaluations/jobs/{job_id}"
        headers = {
            "Authorization": f"Bearer {evalhub_job_for_cross_tenant_test['token']}",
            "X-Tenant": "unauthorized-tenant",
            "Content-Type": "application/json",
        }

        response = requests.get(
            url=url,
            headers=headers,
            verify=evalhub_job_for_cross_tenant_test["ca_bundle_file"],
            timeout=10,
        )

        assert response.status_code in (400, 403), (
            f"Expected 400 or 403 for cross-tenant access, got {response.status_code}: {response.text}"
        )
