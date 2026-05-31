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

from tests.ai_safety.evalhub.utils import (
    _get_job_status,
    build_evalhub_job_payload,
    delete_evalhub_job,
    get_evalhub_job_http,
    submit_evalhub_job,
)
from utilities.kueue_utils import LocalQueue, Workload


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueNegative:
    """Negative tests for EvalHub Kueue integration."""

    def test_nonexistent_queue_name(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-NEG-001: Verify error when submitting job with non-existent queue name.

        Given a Kueue-enabled cluster with no LocalQueue named 'nonexistent-queue',
        When a job is submitted referencing that queue,
        Then the job is accepted but shows admission failure or pending state.
        """
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
            job_name="tc-neg-001-invalid-queue",
        )
        # Add non-existent queue to payload
        payload["queue"] = {"kind": "kueue", "name": "nonexistent-queue"}

        data = submit_evalhub_job(
            host=evalhub_kueue_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_kueue_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            payload=payload,
        )

        job_id = data["resource"]["id"]

        # Verify job status reflects the invalid queue
        status_response = _get_job_status(
            host=evalhub_kueue_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_kueue_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            job_id=job_id,
        )

        state = status_response.get("status", {}).get("state")
        assert state in ("pending", "failed"), f"Job with invalid queue should be pending or failed, got {state}"

        # Cleanup
        delete_evalhub_job(
            host=evalhub_kueue_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_kueue_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            job_id=job_id,
            hard_delete=True,
        )

    def test_submit_without_queue_spec(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-NEG-002: Verify job without queue spec runs without Kueue (backwards compatibility).

        Given EvalHub deployed with or without Kueue,
        When a job is submitted without the queue field,
        Then the job is accepted (202) and runs without Kueue management.
        """
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
            job_name="tc-neg-002-no-queue",
        )
        # Ensure no queue field is present
        payload.pop("queue", None)

        data = submit_evalhub_job(
            host=evalhub_kueue_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_kueue_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            payload=payload,
        )

        job_id = data["resource"]["id"]

        # Verify no Kueue Workload was created for this job
        workloads = list(Workload.get(client=admin_client, namespace=evalhub_kueue_namespace.name))
        job_workloads = [
            wl for wl in workloads if wl.instance.get("metadata", {}).get("name", "").startswith("tc-neg-002")
        ]
        assert len(job_workloads) == 0, "No Workload should be created for job without queue spec"

        # Cleanup
        delete_evalhub_job(
            host=evalhub_kueue_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_kueue_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            job_id=job_id,
            hard_delete=True,
        )

    def test_unauthorized_returns_401(
        self,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-NEG-003: Verify unauthorized request returns 401.

        Given EvalHub requires authentication,
        When a request is made without a valid bearer token,
        Then the API returns 401 Unauthorized.
        """
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
            job_name="tc-neg-003-unauth",
        )
        payload["queue"] = evalhub_kueue_multi_job_local_queue.name

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

        assert response.status_code == 401, f"Expected 401 Unauthorized, got {response.status_code}: {response.text}"

    def test_get_nonexistent_job_returns_404(
        self,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-NEG-005: Verify GET non-existent job returns 404.

        Given no job exists with a specific ID,
        When a GET request is made for that job ID,
        Then the API returns 404 Not Found.
        """
        fake_job_id = "00000000-0000-abc"
        response = get_evalhub_job_http(
            host=evalhub_kueue_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_kueue_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            job_id=fake_job_id,
        )

        assert response.status_code == 404, f"Expected 404 Not Found, got {response.status_code}: {response.text}"

    def test_forbidden_cross_tenant_access(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-NEG-004: Verify forbidden request returns 400/403 for cross-tenant access.

        Given a valid user with access to one namespace,
        When the user attempts to access a different tenant namespace,
        Then the API returns 400 or 403 Forbidden.
        """
        # Submit a job in the correct tenant
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
            job_name="tc-neg-004-cross-tenant",
        )
        payload["queue"] = {"kind": "kueue", "name": evalhub_kueue_multi_job_local_queue.name}

        data = submit_evalhub_job(
            host=evalhub_kueue_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_kueue_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            payload=payload,
        )

        job_id = data["resource"]["id"]

        # Try to access the job from a different (non-existent) tenant
        import requests

        url = f"https://{evalhub_kueue_route.host}/api/v1/evaluations/jobs/{job_id}"
        headers = {
            "Authorization": f"Bearer {evalhub_kueue_user_token}",
            "X-Tenant": "unauthorized-tenant",
            "Content-Type": "application/json",
        }

        response = requests.get(
            url=url,
            headers=headers,
            verify=evalhub_kueue_ca_bundle_file,
            timeout=10,
        )

        assert response.status_code in (400, 403), (
            f"Expected 400 or 403 for cross-tenant access, got {response.status_code}: {response.text}"
        )

        # Cleanup
        delete_evalhub_job(
            host=evalhub_kueue_route.host,
            token=evalhub_kueue_user_token,
            ca_bundle_file=evalhub_kueue_ca_bundle_file,
            tenant=evalhub_kueue_namespace.name,
            job_id=job_id,
            hard_delete=True,
        )
