"""Negative tests for EvalHub Kueue integration.

This module contains negative test cases that validate error handling and
edge cases for EvalHub when integrated with Kueue admission control.
"""

import pytest
import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.job import Job
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError

from tests.ai_safety.evalhub.constants import EVALHUB_JOBS_PATH
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    build_headers,
    evalhub_runtime_label_selector,
    get_evalhub_job_http,
    submit_evalhub_job,
    wait_for_evalhub_job,
    wait_for_evalhub_job_workload_inadmissible,
)
from utilities.kueue_utils import ClusterQueue, LocalQueue, Workload

LOGGER = structlog.get_logger(name=__name__)

KUEUE_QUEUE_LABEL = "kueue.x-k8s.io/queue-name"


def _log_job_kueue_labels(admin_client: DynamicClient, namespace: str, evalhub_job_id: str) -> None:
    """Log the Kueue labels and annotations on the Kubernetes Job created by EvalHub.

    Kueue reads the queue-name from a Job LABEL (not annotation).
    """
    selector = evalhub_runtime_label_selector(evalhub_job_id=evalhub_job_id)
    jobs = list(Job.get(client=admin_client, namespace=namespace, label_selector=selector))
    if not jobs:
        LOGGER.warning("No Kubernetes Job found for EvalHub job", evalhub_job_id=evalhub_job_id)
        return
    for job in jobs:
        labels = job.instance.metadata.labels or {}
        annotations = job.instance.metadata.annotations or {}
        queue_label = labels.get(KUEUE_QUEUE_LABEL)
        LOGGER.info(
            "Kubernetes Job kueue label check",
            job_name=job.name,
            kueue_queue_name_label=queue_label,
            has_kueue_label=queue_label is not None,
            all_labels=dict(labels),
            all_annotations=dict(annotations),
        )


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


@pytest.mark.kueue
@pytest.mark.tier2
class TestEvalHubKueueListFiltering:
    """Verify EvalHub job list filtering is accurate in a Kueue environment."""

    def test_list_jobs_filtered_by_status(
        self,
        admin_client: DynamicClient,
        evalhub_kueue_namespace: Namespace,
        evalhub_kueue_multi_job_local_queue: LocalQueue,
        evalhub_kueue_user_token: str,
        evalhub_kueue_vllm_service: Service,
        evalhub_kueue_route: Route,
        evalhub_kueue_ca_bundle_file: str,
    ) -> None:
        """TC-STATUS-001: GET /jobs?status=completed filters correctly.

        Submits two jobs: job1 runs to completion, job2 is held by a stopped
        ClusterQueue so it never reaches completed state. Verifies that
        ?status=completed returns job1 but not job2, and that every item in
        the filtered response has state == 'completed'.

        Note: Kueue-gated jobs are not exposed as 'pending' in EvalHub's API —
        EvalHub tracks state independently of Kueue admission. Testing the
        'completed' filter avoids relying on transient Kueue-specific states.
        """
        common = {
            "host": evalhub_kueue_route.host,
            "token": evalhub_kueue_user_token,
            "ca_bundle_file": evalhub_kueue_ca_bundle_file,
            "tenant": evalhub_kueue_namespace.name,
        }

        # Job1 — runs to completion
        payload1 = build_evalhub_job_payload(
            model_service_name=evalhub_kueue_vllm_service.name,
            tenant_namespace=evalhub_kueue_namespace.name,
            job_name="tc-status-001-job1",
        )
        payload1["queue"] = {"kind": "kueue", "name": evalhub_kueue_multi_job_local_queue.name}
        data1 = submit_evalhub_job(**common, payload=payload1)
        job1_id = data1["resource"]["id"]

        wait_for_evalhub_job(**common, job_id=job1_id, timeout=600)

        # Job2 — held by stopped ClusterQueue, never completes during this test
        cq_name = evalhub_kueue_multi_job_local_queue.instance.spec.clusterQueue
        cq = ClusterQueue(client=admin_client, name=cq_name)

        with ResourceEditor(patches={cq: {"spec": {"stopPolicy": "HoldAndDrain"}}}):
            payload2 = build_evalhub_job_payload(
                model_service_name=evalhub_kueue_vllm_service.name,
                tenant_namespace=evalhub_kueue_namespace.name,
                job_name="tc-status-001-job2",
            )
            payload2["queue"] = {"kind": "kueue", "name": evalhub_kueue_multi_job_local_queue.name}
            data2 = submit_evalhub_job(**common, payload=payload2)
            job2_id = data2["resource"]["id"]

            try:
                wait_for_evalhub_job_workload_inadmissible(
                    admin_client=admin_client,
                    namespace=evalhub_kueue_namespace.name,
                    evalhub_job_id=job2_id,
                )
            except TimeoutExpiredError:
                _log_job_kueue_labels(admin_client, evalhub_kueue_namespace.name, job2_id)
                raise

            # Query the list endpoint filtering for completed jobs
            url = f"https://{evalhub_kueue_route.host}{EVALHUB_JOBS_PATH}?status=completed&limit=50"
            resp = requests.get(
                url=url,
                headers=build_headers(token=evalhub_kueue_user_token, tenant=evalhub_kueue_namespace.name),
                verify=evalhub_kueue_ca_bundle_file,
                timeout=10,
            )
            assert resp.status_code == 200, f"Expected 200 from list endpoint, got {resp.status_code}: {resp.text}"

            body = resp.json()
            items = body.get("items", [])
            job_ids_in_list = [item.get("resource", {}).get("id") for item in items]

            # job1 completed → must appear in the completed filter
            assert job1_id in job_ids_in_list, (
                f"Completed job {job1_id} not found in ?status=completed results: {job_ids_in_list}"
            )
            # job2 is Kueue-gated (not completed) → must NOT appear
            assert job2_id not in job_ids_in_list, (
                f"Non-completed job {job2_id} appeared in ?status=completed results: {job_ids_in_list}"
            )
            # Every returned item must have state == 'completed'
            for item in items:
                state = item.get("status", {}).get("state")
                assert state == "completed", (
                    f"Expected all items to be completed when filtering by status=completed, got state={state}"
                )
        # Queue restored on exit — job2 runs to completion
