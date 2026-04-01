import requests
import structlog
from timeout_sampler import TimeoutSampler

from tests.model_explainability.evalhub.constants import (
    EVALHUB_COLLECTIONS_PATH,
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
    EVALHUB_JOBS_PATH,
    EVALHUB_PROVIDERS_PATH,
    EVALHUB_VLLM_EMULATOR_PORT,
)
from utilities.guardrails import get_auth_headers

LOGGER = structlog.get_logger(name=__name__)

TENANT_HEADER: str = "X-Tenant"


def build_headers(token: str, tenant: str | None = None) -> dict[str, str]:
    """Build request headers with auth and optional tenant.

    Args:
        token: Bearer token for authentication.
        tenant: Namespace for the X-Tenant header. Omitted if None.

    Returns:
        Headers dict.
    """
    headers = get_auth_headers(token=token)
    if tenant is not None:
        headers[TENANT_HEADER] = tenant
    return headers


def validate_evalhub_health(
    host: str,
    token: str,
    ca_bundle_file: str,
) -> None:
    """Validate that the EvalHub service health endpoint returns healthy status.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.

    Raises:
        AssertionError: If the health check fails.
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_HEALTH_PATH}"
    LOGGER.info(f"Checking EvalHub health at {url}")

    response = requests.get(
        url=url,
        headers=get_auth_headers(token=token),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    LOGGER.info(f"EvalHub health response: {data}")

    assert "status" in data, "Health response missing 'status' field"
    assert data["status"] == EVALHUB_HEALTH_STATUS_HEALTHY, (
        f"Expected status '{EVALHUB_HEALTH_STATUS_HEALTHY}', got '{data['status']}'"
    )
    assert "timestamp" in data, "Health response missing 'timestamp' field"


def validate_evalhub_providers(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str | None = None,
) -> dict:
    """Smoke test for the EvalHub providers endpoint."""
    url = f"https://{host}{EVALHUB_PROVIDERS_PATH}"

    response = requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    assert data.get("items"), f"Smoke test failed: Providers list is empty for tenant {tenant}"

    return data


def validate_evalhub_request_denied(
    host: str,
    token: str,
    path: str,
    ca_bundle_file: str,
    tenant: str,
) -> None:
    """Assert that a cross-tenant request is denied.

    EvalHub uses Kubernetes SubjectAccessReview for tenant authorization.
    When no RBAC rule grants access, the SAR returns DecisionNoOpinion,
    which the service maps to 400 (unable_to_authorize_request).

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for a user without access to the tenant.
        path: API path (e.g. EVALHUB_PROVIDERS_PATH).
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace the user should NOT have access to.

    Raises:
        AssertionError: If the request succeeds (2xx).
    """
    url = f"https://{host}{path}"
    LOGGER.info(f"Expecting access denied at {url} for tenant {tenant}")

    response = requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    assert response.status_code in (400, 403), (
        f"Expected 400 or 403 for cross-tenant access, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert data.get("message_code") in ("unable_to_authorize_request", "forbidden"), (
        f"Expected authorization denial, got message_code: {data.get('message_code')}"
    )


def validate_evalhub_request_no_tenant(
    host: str,
    token: str,
    path: str,
    ca_bundle_file: str,
) -> None:
    """Assert that a request without the X-Tenant header returns 400.

    The EvalHub service requires an explicit X-Tenant header on
    tenant-scoped endpoints. Omitting it is a client error.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        path: API path (e.g. EVALHUB_PROVIDERS_PATH).
        ca_bundle_file: Path to CA bundle for TLS verification.

    Raises:
        AssertionError: If the response is not 400.
    """
    url = f"https://{host}{path}"
    LOGGER.info(f"Expecting 400 Bad Request at {url} (no X-Tenant header)")

    response = requests.get(
        url=url,
        headers=build_headers(token=token, tenant=None),
        verify=ca_bundle_file,
        timeout=10,
    )
    assert response.status_code == 400, f"Expected 400 Bad Request, got {response.status_code}: {response.text}"


def submit_evalhub_job(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    payload: dict,
) -> dict:
    """Submit an evaluation job and assert 202 Accepted.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.
        payload: Job request body (model, benchmarks, etc.).

    Returns:
        Response JSON (job resource with ID and status).

    Raises:
        AssertionError: If the response is not 202.
    """
    url = f"https://{host}{EVALHUB_JOBS_PATH}"
    LOGGER.info(f"Submitting evaluation job to {url} for tenant {tenant}")

    response = requests.post(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )
    assert response.status_code == 202, f"Expected 202 Accepted, got {response.status_code}: {response.text}"

    data = response.json()
    LOGGER.info(f"Job submitted: {data.get('resource', {}).get('id', 'unknown')}")
    return data


def validate_evalhub_post_denied(
    host: str,
    token: str,
    path: str,
    ca_bundle_file: str,
    tenant: str,
    payload: dict,
) -> None:
    """Assert that a POST request is denied for cross-tenant access.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for a user without access to the tenant.
        path: API path (e.g. EVALHUB_JOBS_PATH).
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace the user should NOT have access to.
        payload: Request body.

    Raises:
        AssertionError: If the request succeeds.
    """
    url = f"https://{host}{path}"
    LOGGER.info(f"Expecting POST denied at {url} for tenant {tenant}")

    response = requests.post(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )
    assert response.status_code in (400, 403), (
        f"Expected 400 or 403 for cross-tenant POST, got {response.status_code}: {response.text}"
    )


def validate_evalhub_post_no_tenant(
    host: str,
    token: str,
    path: str,
    ca_bundle_file: str,
    payload: dict,
) -> None:
    """Assert that a POST without X-Tenant header returns 400.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        path: API path (e.g. EVALHUB_JOBS_PATH).
        ca_bundle_file: Path to CA bundle for TLS verification.
        payload: Request body.

    Raises:
        AssertionError: If the response is not 400.
    """
    url = f"https://{host}{path}"
    LOGGER.info(f"Expecting 400 for POST at {url} (no X-Tenant header)")

    response = requests.post(
        url=url,
        headers=build_headers(token=token, tenant=None),
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )
    assert response.status_code == 400, f"Expected 400 Bad Request, got {response.status_code}: {response.text}"


# ---------------------------------------------------------------------------
# Job state constants
# ---------------------------------------------------------------------------

EVALHUB_JOB_TERMINAL_STATES: set[str] = {
    "completed",
    "failed",
    "cancelled",
    "partially_failed",
}


# ---------------------------------------------------------------------------
# Job polling
# ---------------------------------------------------------------------------


def _get_job_status(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
) -> dict:
    """Fetch current job status from the EvalHub API."""
    url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
    response = requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def wait_for_evalhub_job(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
    timeout: int = 600,
    sleep: int = 10,
) -> dict:
    """Poll a job until it reaches a terminal state.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.
        job_id: ID of the job to poll.
        timeout: Maximum seconds to wait (default 10 minutes).
        sleep: Seconds between polls (default 10).

    Returns:
        Final job response dict.

    Raises:
        TimeoutExpiredError: If the job does not reach a terminal state.
    """
    LOGGER.info(f"Waiting for job {job_id} to complete (timeout={timeout}s)")

    for sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=sleep,
        func=_get_job_status,
        host=host,
        token=token,
        ca_bundle_file=ca_bundle_file,
        tenant=tenant,
        job_id=job_id,
    ):
        state = sample.get("status", {}).get("state", "")
        LOGGER.info(f"Job {job_id} state: {state}")
        if state in EVALHUB_JOB_TERMINAL_STATES:
            return sample

    return {}


def validate_evalhub_job_completed(job_data: dict) -> None:
    """Assert that a job completed successfully with benchmark results.

    Args:
        job_data: Job response dict from wait_for_evalhub_job.

    Raises:
        AssertionError: If the job did not complete or has no results.
    """
    state = job_data.get("status", {}).get("state")
    assert state == "completed", (
        f"Expected job state 'completed', got '{state}': {job_data.get('status', {}).get('message')}"
    )

    results = job_data.get("results", {})
    benchmarks = results.get("benchmarks", [])
    assert benchmarks, f"Job completed but has no benchmark results: {results}"

    for bench in benchmarks:
        assert bench.get("metrics"), f"Benchmark '{bench.get('id')}' completed with no metrics: {bench}"


def list_evalhub_jobs(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
) -> dict:
    """List evaluation jobs for a tenant.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.

    Returns:
        Response JSON with job list.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_JOBS_PATH}"
    response = requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def list_evalhub_collections(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
) -> dict:
    """List evaluation collections for a tenant."""
    url = f"https://{host}{EVALHUB_COLLECTIONS_PATH}"
    response = requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def delete_evalhub_job(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
) -> int:
    """Delete (cancel) an evaluation job. Returns the HTTP status code."""
    url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
    response = requests.delete(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    return response.status_code


def validate_evalhub_delete_denied(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
) -> None:
    """Assert that a DELETE request is denied for cross-tenant access."""
    status = delete_evalhub_job(
        host=host,
        token=token,
        ca_bundle_file=ca_bundle_file,
        tenant=tenant,
        job_id=job_id,
    )
    assert status in (400, 403), f"Expected 400 or 403 for cross-tenant DELETE, got {status}"


def validate_evalhub_delete_no_tenant(
    host: str,
    token: str,
    ca_bundle_file: str,
    job_id: str,
) -> None:
    """Assert that a DELETE without X-Tenant header returns 400."""
    url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
    response = requests.delete(
        url=url,
        headers=build_headers(token=token, tenant=None),
        verify=ca_bundle_file,
        timeout=10,
    )
    assert response.status_code == 400, f"Expected 400 Bad Request, got {response.status_code}: {response.text}"


# ---------------------------------------------------------------------------
# Shared job and collection payloads
# ---------------------------------------------------------------------------


def build_evalhub_job_payload(
    model_service_name: str,
    tenant_namespace: str,
    job_name: str = "evalhub-mt-test-job",
) -> dict:
    """Build an EvalHub job payload targeting the vLLM emulator.

    Args:
        model_service_name: Kubernetes Service name for the vLLM emulator.
        tenant_namespace: Namespace where the service runs.
        job_name: Name for the evaluation job.

    Returns:
        Job request body dict.
    """
    model_url = f"http://{model_service_name}.{tenant_namespace}.svc.cluster.local:{EVALHUB_VLLM_EMULATOR_PORT}/v1"
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
                    "num_examples": 10,
                    "tokenizer": "google/flan-t5-small",
                },
            }
        ],
    }


def submit_evalhub_collection(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    payload: dict,
) -> requests.Response:
    """POST a collection creation request.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.
        payload: Collection config body.

    Returns:
        Raw response (caller decides which status to assert).
    """
    url = f"https://{host}{EVALHUB_COLLECTIONS_PATH}"
    return requests.post(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )
