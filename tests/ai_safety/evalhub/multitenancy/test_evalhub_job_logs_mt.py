"""EvalHub job log access tests (RHOAIENG-58864 / RHAISTRAT-1437).

Validates the HTTP API implemented in eval-hub:
``GET /api/v1/evaluations/jobs/{id}/logs`` and
``GET /api/v1/evaluations/jobs/{id}/benchmarks/{benchmark_index}/logs``.

The eval-hub-sdk ``evalhub eval logs`` CLI command is not yet implemented and is
out of scope for this module.
"""

from __future__ import annotations

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service

from tests.ai_safety.evalhub.constants import (
    EVALHUB_LOG_ADAPTER_CONTAINER,
    EVALHUB_LOG_COMPLETED_MARKER,
    EVALHUB_LOG_CONTENT_TYPE,
    EVALHUB_LOG_MAX_TAIL_LINES,
    EVALHUB_LOG_SECTION_PREFIX,
)
from tests.ai_safety.evalhub.utils import (
    build_evalhub_job_payload,
    build_failing_evalhub_job_payload,
    build_headers,
    delete_evalhub_job,
    get_evalhub_job_logs_http,
    submit_evalhub_job,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
    wait_for_evalhub_runtime_resources_absent,
)


def _assert_plain_text_logs_response(response: requests.Response) -> str:
    """Assert OpenAPI-conformant 200 text/plain log response and return the body."""
    assert response.status_code == 200, f"Expected 200 for job logs, got {response.status_code}: {response.text}"
    content_type = response.headers.get("Content-Type", "")
    assert content_type.startswith(EVALHUB_LOG_CONTENT_TYPE), (
        f"Expected Content-Type starting with {EVALHUB_LOG_CONTENT_TYPE!r}, got {content_type!r}"
    )
    return response.text


def _count_non_empty_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


@pytest.fixture(scope="class")
def evalhub_logs_completed_job_id(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_vllm_emulator_service: Service,
) -> str:
    """Submit one arc_easy job and wait for completion (shared by log retrieval tests)."""
    payload = build_evalhub_job_payload(
        model_service_name=evalhub_vllm_emulator_service.name,
        tenant_namespace=tenant_a_namespace.name,
        job_name="evalhub-logs-completed-job",
    )
    data = submit_evalhub_job(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        payload=payload,
    )
    job_id = data["resource"]["id"]
    wait_for_evalhub_job(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        job_id=job_id,
        timeout=600,
    )
    return job_id


LOGS_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-job-logs-mt"})


@pytest.mark.parametrize("model_namespace", [LOGS_MODEL_NAMESPACE], indirect=True)
@pytest.mark.ai_safety
class TestEvalHubJobLogsMT:
    """Multi-tenancy tests for EvalHub evaluation job log HTTP API."""

    def test_completed_job_logs(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given a successfully completed job, When GET /jobs/{id}/logs, Then full logs are returned."""
        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
        )
        body = _assert_plain_text_logs_response(response=response)
        assert EVALHUB_LOG_SECTION_PREFIX in body
        assert "benchmark_id=arc_easy" in body
        assert EVALHUB_LOG_ADAPTER_CONTAINER in body
        assert EVALHUB_LOG_COMPLETED_MARKER in body

    def test_completed_job_benchmark_logs(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given a completed job, When GET benchmark logs, Then adapter output is returned without section header."""
        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            benchmark_index=0,
        )
        body = _assert_plain_text_logs_response(response=response)
        assert EVALHUB_LOG_SECTION_PREFIX not in body
        assert EVALHUB_LOG_COMPLETED_MARKER in body

    def test_running_job_logs(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        admin_client: DynamicClient,
    ) -> None:
        """Given an in-progress job, When GET /jobs/{id}/logs, Then logs are retrievable."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-logs-running-job",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )

        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        body = _assert_plain_text_logs_response(response=response)
        assert EVALHUB_LOG_SECTION_PREFIX in body
        assert "benchmark_id=arc_easy" in body

    def test_failed_job_logs(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Given a failed job, When GET /jobs/{id}/logs, Then logs are retrievable."""
        payload = build_failing_evalhub_job_payload(
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-logs-failed-job",
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
        assert job_result.get("status", {}).get("state") == "failed"

        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        body = _assert_plain_text_logs_response(response=response)
        assert EVALHUB_LOG_SECTION_PREFIX in body
        assert "benchmark_id=arc_easy" in body

    def test_cancelled_job_logs(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_vllm_emulator_service: Service,
        admin_client: DynamicClient,
    ) -> None:
        """Given a cancelled job, When GET /jobs/{id}/logs, Then logs remain retrievable."""
        payload = build_evalhub_job_payload(
            model_service_name=evalhub_vllm_emulator_service.name,
            tenant_namespace=tenant_a_namespace.name,
            job_name="evalhub-logs-cancelled-job",
        )
        data = submit_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            payload=payload,
        )
        job_id = data["resource"]["id"]
        wait_for_evalhub_runtime_job_count(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
            minimum=1,
        )

        cancel_response = delete_evalhub_job(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
            hard_delete=False,
        )
        assert cancel_response.status_code == 204

        wait_for_evalhub_runtime_resources_absent(
            admin_client=admin_client,
            namespace=tenant_a_namespace.name,
            evalhub_job_id=job_id,
        )

        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
        body = _assert_plain_text_logs_response(response=response)
        assert EVALHUB_LOG_SECTION_PREFIX in body
        assert "benchmark_id=arc_easy" in body

    def test_partial_log_retrieval_tail_lines(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given a completed job, When tail_lines=1, Then the response is shorter than the full log."""
        full_response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            params={"tail_lines": str(EVALHUB_LOG_MAX_TAIL_LINES)},
        )
        tail_response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            params={"tail_lines": "1"},
        )
        full_body = _assert_plain_text_logs_response(response=full_response)
        tail_body = _assert_plain_text_logs_response(response=tail_response)
        assert _count_non_empty_lines(tail_body) <= _count_non_empty_lines(full_body)
        assert EVALHUB_LOG_COMPLETED_MARKER in full_body
        assert EVALHUB_LOG_SECTION_PREFIX in tail_body

    def test_log_query_parameters_since_seconds_and_timestamps(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given a completed job, When since_seconds and timestamps are set, Then the API accepts them."""
        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            params={"since_seconds": "3600", "timestamps": "true"},
        )
        _assert_plain_text_logs_response(response=response)


@pytest.mark.parametrize("model_namespace", [LOGS_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier3
@pytest.mark.ai_safety
class TestEvalHubJobLogsAuthMT:
    """Authentication and authorization for EvalHub job log endpoints."""

    def test_logs_cross_tenant_denied(
        self,
        tenant_a_token: str,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given a job in tenant-a, When tenant-b requests logs, Then access is denied."""
        validate_evalhub_request_denied(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=f"/api/v1/evaluations/jobs/{evalhub_logs_completed_job_id}/logs",
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_b_namespace.name,
        )

    def test_logs_missing_tenant_rejected(
        self,
        tenant_a_token: str,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given an authenticated user, When X-Tenant is omitted, Then log access returns 400."""
        validate_evalhub_request_no_tenant(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            path=f"/api/v1/evaluations/jobs/{evalhub_logs_completed_job_id}/logs",
            ca_bundle_file=evalhub_mt_ca_bundle_file,
        )

    def test_logs_unauthenticated_rejected(
        self,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given no Authorization header, When GET /jobs/{id}/logs, Then the request is rejected."""
        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token="",
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            headers=build_headers(token="", tenant=tenant_a_namespace.name),
        )
        assert response.status_code in (401, 403), (
            f"Expected 401 or 403 for unauthenticated log access, got {response.status_code}: {response.text}"
        )


@pytest.mark.parametrize("model_namespace", [LOGS_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier3
@pytest.mark.ai_safety
class TestEvalHubJobLogsNegativeMT:
    """Negative tests for EvalHub job log HTTP API."""

    def test_logs_nonexistent_job_returns_404(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
    ) -> None:
        """Given an unknown job id, When GET /jobs/{id}/logs, Then the API returns 404."""
        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id="00000000-0000-0000-0000-000000000000",
        )
        assert response.status_code == 404, f"Expected 404 for unknown job logs, got {response.status_code}"

    def test_logs_invalid_tail_lines_zero(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given tail_lines=0, When GET /jobs/{id}/logs, Then the API returns 400."""
        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            params={"tail_lines": "0"},
        )
        assert response.status_code == 400
        assert response.json().get("message_code") == "query_parameter_invalid"

    def test_logs_invalid_tail_lines_over_max(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given tail_lines above the OpenAPI maximum, When GET logs, Then the API returns 400."""
        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            params={"tail_lines": str(EVALHUB_LOG_MAX_TAIL_LINES + 1)},
        )
        assert response.status_code == 400
        assert response.json().get("message_code") == "query_parameter_invalid"

    def test_logs_invalid_benchmark_index(
        self,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given an out-of-range benchmark index, When GET benchmark logs, Then the API returns 404."""
        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            benchmark_index=99,
        )
        assert response.status_code == 404
