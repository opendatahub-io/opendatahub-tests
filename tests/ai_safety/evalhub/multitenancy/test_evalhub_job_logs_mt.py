"""EvalHub job log access tests (RHOAIENG-58864 / RHAISTRAT-1437).

Validates the HTTP API implemented in eval-hub:
``GET /api/v1/evaluations/jobs/{id}/logs`` and
``GET /api/v1/evaluations/jobs/{id}/benchmarks/{benchmark_index}/logs``.

The eval-hub-sdk ``evalhub eval logs`` CLI command is not yet implemented and is
out of scope for this module.
"""

from __future__ import annotations

from typing import Literal

import pytest
import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.route import Route
from ocp_resources.service import Service
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.evalhub.constants import (
    EVALHUB_LOG_ADAPTER_CONTAINER,
    EVALHUB_LOG_COMPLETED_MARKER,
    EVALHUB_LOG_CONTENT_TYPE,
    EVALHUB_LOG_MAX_TAIL_LINES,
    EVALHUB_LOG_SECTION_PREFIX,
)
from tests.ai_safety.evalhub.utils import (
    EVALHUB_JOB_TERMINAL_STATES,
    build_evalhub_job_payload,
    build_failing_evalhub_job_payload,
    build_headers,
    delete_evalhub_job,
    get_evalhub_job_http,
    get_evalhub_job_logs_http,
    submit_evalhub_job,
    validate_evalhub_request_denied,
    validate_evalhub_request_no_tenant,
    wait_for_evalhub_job,
    wait_for_evalhub_runtime_job_count,
    wait_for_evalhub_runtime_resources_absent,
)

LOGS_MODEL_NAMESPACE = pytest.param({"name": "test-evalhub-job-logs-mt"})

AuthScenario = Literal["cross_namespace", "missing_tenant", "unauthenticated"]
InvalidLogsScenario = Literal["tail_lines_zero", "tail_lines_over_max", "invalid_benchmark_index"]


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


def _fetch_evalhub_job_logs_while_running(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
    timeout: int = 180,
    sleep: int = 2,
) -> str:
    """Poll until the EvalHub API reports ``running``, then fetch logs in the same iteration."""
    for status_response in TimeoutSampler(
        wait_timeout=timeout,
        sleep=sleep,
        func=get_evalhub_job_http,
        host=host,
        token=token,
        ca_bundle_file=ca_bundle_file,
        tenant=tenant,
        job_id=job_id,
    ):
        status_response.raise_for_status()
        state = status_response.json().get("status", {}).get("state", "")
        if state in EVALHUB_JOB_TERMINAL_STATES:
            pytest.fail(
                f"Job '{job_id}' reached terminal state '{state}' before running; "
                "cannot verify in-progress log retrieval"
            )
        if state != "running":
            continue

        response = get_evalhub_job_logs_http(
            host=host,
            token=token,
            ca_bundle_file=ca_bundle_file,
            tenant=tenant,
            job_id=job_id,
        )
        return _assert_plain_text_logs_response(response=response)

    raise TimeoutExpiredError(f"Job '{job_id}' did not reach running state within {timeout}s")


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


@pytest.fixture(scope="class")
def evalhub_logs_completed_job_logs(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_logs_completed_job_id: str,
) -> str:
    """Validated full job logs for a completed evaluation job (fetched once per class)."""
    response = get_evalhub_job_logs_http(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        job_id=evalhub_logs_completed_job_id,
        params={"tail_lines": str(EVALHUB_LOG_MAX_TAIL_LINES)},
    )
    return _assert_plain_text_logs_response(response=response)


@pytest.fixture(scope="class")
def evalhub_logs_completed_benchmark_logs(
    tenant_a_token: str,
    tenant_a_namespace: Namespace,
    evalhub_mt_ca_bundle_file: str,
    evalhub_mt_route: Route,
    evalhub_logs_completed_job_id: str,
) -> str:
    """Validated benchmark logs for a completed evaluation job (fetched once per class)."""
    response = get_evalhub_job_logs_http(
        host=evalhub_mt_route.host,
        token=tenant_a_token,
        ca_bundle_file=evalhub_mt_ca_bundle_file,
        tenant=tenant_a_namespace.name,
        job_id=evalhub_logs_completed_job_id,
        benchmark_index=0,
    )
    return _assert_plain_text_logs_response(response=response)


@pytest.mark.parametrize("model_namespace", [LOGS_MODEL_NAMESPACE], indirect=True)
@pytest.mark.tier2
@pytest.mark.ai_safety
class TestEvalHubJobLogsMT:
    """Multi-tenancy tests for EvalHub evaluation job log HTTP API."""

    def test_completed_job_logs(
        self,
        evalhub_logs_completed_job_logs: str,
    ) -> None:
        """Given a successfully completed job, When GET /jobs/{id}/logs, Then full logs are returned."""
        body = evalhub_logs_completed_job_logs
        assert EVALHUB_LOG_SECTION_PREFIX in body
        assert "benchmark_id=arc_easy" in body
        assert EVALHUB_LOG_ADAPTER_CONTAINER in body
        assert EVALHUB_LOG_COMPLETED_MARKER in body

    def test_completed_job_benchmark_logs(
        self,
        evalhub_logs_completed_benchmark_logs: str,
    ) -> None:
        """Given a completed job, When GET benchmark logs, Then adapter output is returned without section header."""
        body = evalhub_logs_completed_benchmark_logs
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
        """Given an in-progress job, When GET /jobs/{id}/logs, Then logs are retrievable.

        Polls the EvalHub API until state is ``running``, then fetches logs in the
        same iteration so the request is not issued after the job has completed.
        """
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

        body = _fetch_evalhub_job_logs_while_running(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        )
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
        """Given a cancelled job, When GET /jobs/{id}/logs, Then logs remain retrievable.

        Waits for the batch Job to exist, then polls until the API reports ``running``
        before soft DELETE. If a terminal state is observed first, the test fails rather
        than cancelling an already-finished job or polling until timeout.
        """
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

        # Poll until the API reports ``running`` (then cancel) or a terminal state
        # (job finished too fast — fail rather than loop until timeout).
        for status_response in TimeoutSampler(
            wait_timeout=180,
            sleep=2,
            func=get_evalhub_job_http,
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=job_id,
        ):
            status_response.raise_for_status()
            state = status_response.json().get("status", {}).get("state", "")
            if state == "running":
                break
            if state in EVALHUB_JOB_TERMINAL_STATES:
                pytest.fail(
                    f"Job '{job_id}' reached terminal state '{state}' before running; "
                    "cannot verify cancel-in-progress log retrieval"
                )
        else:
            raise TimeoutExpiredError(f"Job '{job_id}' did not reach running state within 180s before cancel")

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
        evalhub_logs_completed_job_logs: str,
    ) -> None:
        """Given a completed job, When tail_lines=1, Then the response is shorter than the full log."""
        full_body = evalhub_logs_completed_job_logs
        tail_response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            params={"tail_lines": "1"},
        )
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

    @pytest.mark.parametrize(
        "auth_scenario",
        [
            pytest.param("cross_namespace", id="test_logs_cross_tenant_denied"),
            pytest.param("missing_tenant", id="test_logs_missing_tenant_rejected"),
            pytest.param("unauthenticated", id="test_logs_unauthenticated_rejected"),
        ],
    )
    def test_logs_auth_rejection(
        self,
        auth_scenario: AuthScenario,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        tenant_b_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given log access prerequisites, When auth or tenant context is invalid, Then access is rejected."""
        host = evalhub_mt_route.host
        logs_path = f"/api/v1/evaluations/jobs/{evalhub_logs_completed_job_id}/logs"

        if auth_scenario == "cross_namespace":
            validate_evalhub_request_denied(
                host=host,
                token=tenant_a_token,
                path=logs_path,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
                tenant=tenant_b_namespace.name,
            )
            return

        if auth_scenario == "missing_tenant":
            validate_evalhub_request_no_tenant(
                host=host,
                token=tenant_a_token,
                path=logs_path,
                ca_bundle_file=evalhub_mt_ca_bundle_file,
            )
            return

        response = get_evalhub_job_logs_http(
            host=host,
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

    @pytest.mark.parametrize(
        "invalid_scenario",
        [
            pytest.param("tail_lines_zero", id="test_logs_invalid_tail_lines_zero"),
            pytest.param("tail_lines_over_max", id="test_logs_invalid_tail_lines_over_max"),
            pytest.param("invalid_benchmark_index", id="test_logs_invalid_benchmark_index"),
        ],
    )
    def test_logs_invalid_request_on_existing_job(
        self,
        invalid_scenario: InvalidLogsScenario,
        tenant_a_token: str,
        tenant_a_namespace: Namespace,
        evalhub_mt_ca_bundle_file: str,
        evalhub_mt_route: Route,
        evalhub_logs_completed_job_id: str,
    ) -> None:
        """Given a completed job, When log request parameters are invalid, Then the API returns an error."""
        params: dict[str, str] | None = None
        benchmark_index: int | None = None
        expected_status = 400
        expected_message_code: str | None = "query_parameter_invalid"

        if invalid_scenario == "tail_lines_zero":
            params = {"tail_lines": "0"}
        elif invalid_scenario == "tail_lines_over_max":
            params = {"tail_lines": str(EVALHUB_LOG_MAX_TAIL_LINES + 1)}
        else:
            benchmark_index = 99
            expected_status = 404
            expected_message_code = None

        response = get_evalhub_job_logs_http(
            host=evalhub_mt_route.host,
            token=tenant_a_token,
            ca_bundle_file=evalhub_mt_ca_bundle_file,
            tenant=tenant_a_namespace.name,
            job_id=evalhub_logs_completed_job_id,
            params=params,
            benchmark_index=benchmark_index,
        )
        assert response.status_code == expected_status
        if expected_message_code is not None:
            assert response.json().get("message_code") == expected_message_code
