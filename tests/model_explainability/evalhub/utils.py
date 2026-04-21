import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.job import Job
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_explainability.evalhub.constants import (
    EVALHUB_COLLECTIONS_PATH,
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
    EVALHUB_JOB_CONFIG_CLUSTERROLE,
    EVALHUB_JOBS_PATH,
    EVALHUB_JOBS_WRITER_CLUSTERROLE,
    EVALHUB_K8S_LABEL_APP,
    EVALHUB_K8S_LABEL_APP_VALUE,
    EVALHUB_K8S_LABEL_COMPONENT,
    EVALHUB_K8S_LABEL_COMPONENT_VALUE,
    EVALHUB_K8S_LABEL_JOB_ID,
    EVALHUB_MT_CR_NAME,
    EVALHUB_PROVIDERS_PATH,
    EVALHUB_VLLM_EMULATOR_PORT,
    GARAK_JOB_POLL_INTERVAL,
    GARAK_JOB_TIMEOUT,
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
    tenant_namespace: str,
    expected_providers: list[str] | None = None,
) -> dict:
    """Validate that the EvalHub providers endpoint returns the expected providers."""
    url = f"https://{host}{EVALHUB_PROVIDERS_PATH}"
    LOGGER.info(f"Checking EvalHub providers at {url}")

    response = requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant_namespace),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    LOGGER.info(f"EvalHub providers response: {data}")

    assert data.get("items"), f"Providers list is empty for tenant {tenant_namespace}"

    if expected_providers:
        provider_ids = [item["resource"]["id"] for item in data.get("items", [])]
        for expected in expected_providers:
            assert expected in provider_ids, f"Expected provider '{expected}' not found in {provider_ids}"

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
    try:
        body = response.json()
    except ValueError:
        body = {}
    body_str = str(body).lower()
    assert any(kw in body_str for kw in ("tenant", "missing tenant header", "x-tenant")), (
        f"Expected tenant-header-related error in response body for no-tenant GET, got: {response.text}"
    )


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
    try:
        body = response.json()
    except ValueError:
        body = {}
    body_str = str(body).lower()
    assert any(kw in body_str for kw in ("unauthorized", "forbidden", "auth")), (
        f"Expected auth-related error in response body for cross-tenant POST, got: {response.text}"
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
    try:
        body = response.json()
    except ValueError:
        body = {}
    body_str = str(body).lower()
    assert any(kw in body_str for kw in ("tenant", "missing tenant header", "x-tenant")), (
        f"Expected tenant-header-related error in response body for no-tenant POST, got: {response.text}"
    )


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

    raise TimeoutExpiredError(f"Job '{job_id}' did not reach a terminal state within {timeout}s")


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

    arc_easy_benches = [b for b in benchmarks if b.get("id") == "arc_easy"]
    assert arc_easy_benches, f"Expected 'arc_easy' benchmark in results, got: {[b.get('id') for b in benchmarks]}"
    assert arc_easy_benches[0].get("metrics"), f"Benchmark 'arc_easy' completed with no metrics: {arc_easy_benches[0]}"


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
    *,
    hard_delete: bool | None = None,
) -> requests.Response:
    """Delete (cancel) an evaluation job. Returns the full HTTP response.

    Args:
        hard_delete: When ``True``, pass ``hard_delete=true`` (remove API record).
            When ``False``, pass ``hard_delete=false`` (soft cancel). When ``None``,
            omit the query param (server default: soft cancel).
    """
    url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
    params: dict[str, str] | None = None
    if hard_delete is not None:
        params = {"hard_delete": "true" if hard_delete else "false"}
    return requests.delete(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        params=params,
        verify=ca_bundle_file,
        timeout=10,
    )


def validate_evalhub_delete_denied(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
) -> None:
    """Assert that a DELETE request is denied for cross-tenant access."""
    response = delete_evalhub_job(
        host=host,
        token=token,
        ca_bundle_file=ca_bundle_file,
        tenant=tenant,
        job_id=job_id,
    )
    assert response.status_code in (400, 403), (
        f"Expected 400 or 403 for cross-tenant DELETE, got {response.status_code}: {response.text}"
    )
    try:
        body = response.json()
    except ValueError:
        body = {}
    body_str = str(body).lower()
    assert any(kw in body_str for kw in ("unauthorized", "forbidden", "auth")), (
        f"Expected auth-related error in response body for cross-tenant DELETE, got: {response.text}"
    )


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
    try:
        body = response.json()
    except ValueError:
        body = {}
    body_str = str(body).lower()
    assert any(kw in body_str for kw in ("tenant", "missing tenant header", "x-tenant")), (
        f"Expected tenant-header-related error in response body for no-tenant DELETE, got: {response.text}"
    )


# ---------------------------------------------------------------------------
# Shared job and collection payloads
# ---------------------------------------------------------------------------


def post_evalhub_job_raw(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    payload: dict,
) -> requests.Response:
    """POST /evaluations/jobs without asserting status (caller handles 202 vs errors)."""
    url = f"https://{host}{EVALHUB_JOBS_PATH}"
    return requests.post(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )


def get_evalhub_job_http(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
) -> requests.Response:
    """GET a single evaluation job by id."""
    url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
    return requests.get(
        url=url,
        headers=build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )


def evalhub_runtime_label_selector(evalhub_job_id: str) -> str:
    """Label selector for batch Jobs and spec ConfigMaps created for one EvalHub job id."""
    return (
        f"{EVALHUB_K8S_LABEL_APP}={EVALHUB_K8S_LABEL_APP_VALUE},"
        f"{EVALHUB_K8S_LABEL_COMPONENT}={EVALHUB_K8S_LABEL_COMPONENT_VALUE},"
        f"{EVALHUB_K8S_LABEL_JOB_ID}={evalhub_job_id}"
    )


def wait_for_evalhub_runtime_job_count(
    admin_client: DynamicClient,
    namespace: str,
    evalhub_job_id: str,
    *,
    minimum: int,
    timeout: int = 180,
    sleep: int = 5,
) -> list[Job]:
    """Wait until at least ``minimum`` batch Jobs exist for the EvalHub logical job id."""
    selector = evalhub_runtime_label_selector(evalhub_job_id=evalhub_job_id)

    def list_jobs() -> list[Job]:
        return list(
            Job.get(
                client=admin_client,
                namespace=namespace,
                label_selector=selector,
            )
        )

    for jobs in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=list_jobs):
        if len(jobs) >= minimum:
            return jobs
    raise TimeoutExpiredError(
        f"Expected at least {minimum} batch Job(s) for evalhub job_id={evalhub_job_id} in {namespace}"
    )


def wait_for_evalhub_runtime_resources_absent(
    admin_client: DynamicClient,
    namespace: str,
    evalhub_job_id: str,
    *,
    timeout: int = 180,
    sleep: int = 5,
) -> None:
    """Wait until no batch Job or spec ConfigMap remains for the EvalHub job id."""
    selector = evalhub_runtime_label_selector(evalhub_job_id=evalhub_job_id)

    def count_runtime_objects() -> tuple[int, int]:
        jobs = list(Job.get(client=admin_client, namespace=namespace, label_selector=selector))
        cms = list(ConfigMap.get(client=admin_client, namespace=namespace, label_selector=selector))
        return len(jobs), len(cms)

    for job_count, cm_count in TimeoutSampler(wait_timeout=timeout, sleep=sleep, func=count_runtime_objects):
        if job_count == 0 and cm_count == 0:
            return
    raise TimeoutExpiredError(
        f"Timed out waiting for runtime Job/ConfigMap cleanup for job_id={evalhub_job_id} in {namespace}"
    )


def build_evalhub_multi_benchmark_job_payload(
    model_service_name: str,
    tenant_namespace: str,
    job_name: str = "evalhub-mt-multibench-job",
) -> dict:
    """Two lm_evaluation_harness benchmarks with different parameters (distinct job.json mapping)."""
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
                    "num_examples": 8,
                    "tokenizer": "google/flan-t5-small",
                },
            },
            {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "parameters": {
                    "num_examples": 3,
                    "tokenizer": "google/flan-t5-small",
                },
            },
        ],
    }


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


# ---------------------------------------------------------------------------
# Tenant RBAC readiness check
# ---------------------------------------------------------------------------


def tenant_rbac_ready(admin_client: DynamicClient, namespace: str) -> bool:
    """Check if the operator has provisioned job RBAC for the test EvalHub instance.

    Matches by roleRef ClusterRole name rather than RoleBinding name substrings,
    because long namespace names cause normalizeDNS1123LabelValue to truncate
    the "job-config"/"job-writer" suffix out of the RoleBinding name.

    Also waits for the operator-created ServiceAccount (name contains "job") and
    service CA ConfigMap (name contains "service-ca") to be present.
    """
    rbs = list(RoleBinding.get(client=admin_client, namespace=namespace))
    has_job_config = any(
        rb.instance.roleRef.name == EVALHUB_JOB_CONFIG_CLUSTERROLE and rb.name.startswith(EVALHUB_MT_CR_NAME)
        for rb in rbs
    )
    has_job_writer = any(
        rb.instance.roleRef.name == EVALHUB_JOBS_WRITER_CLUSTERROLE and rb.name.startswith(EVALHUB_MT_CR_NAME)
        for rb in rbs
    )
    sas = list(ServiceAccount.get(client=admin_client, namespace=namespace))
    has_job_sa = any(sa.name.startswith(EVALHUB_MT_CR_NAME) and "job" in sa.name for sa in sas)
    cms = list(ConfigMap.get(client=admin_client, namespace=namespace))
    has_service_ca_cm = any(cm.name.startswith(EVALHUB_MT_CR_NAME) and "service-ca" in cm.name for cm in cms)
    return has_job_config and has_job_writer and has_job_sa and has_service_ca_cm


def tenant_rbac_absent(admin_client: DynamicClient, namespace: str) -> bool:
    """Check that all operator-managed RBAC resources have been removed.

    Returns True only when both RoleBindings, the job ServiceAccount,
    and the service-CA ConfigMap are all gone.
    """
    rbs = list(RoleBinding.get(client=admin_client, namespace=namespace))
    has_job_config = any(
        rb.instance.roleRef.name == EVALHUB_JOB_CONFIG_CLUSTERROLE and rb.name.startswith(EVALHUB_MT_CR_NAME)
        for rb in rbs
    )
    has_job_writer = any(
        rb.instance.roleRef.name == EVALHUB_JOBS_WRITER_CLUSTERROLE and rb.name.startswith(EVALHUB_MT_CR_NAME)
        for rb in rbs
    )
    sas = list(ServiceAccount.get(client=admin_client, namespace=namespace))
    has_job_sa = any(sa.name.startswith(EVALHUB_MT_CR_NAME) and "job" in sa.name for sa in sas)
    cms = list(ConfigMap.get(client=admin_client, namespace=namespace))
    has_service_ca_cm = any(cm.name.startswith(EVALHUB_MT_CR_NAME) and "service-ca" in cm.name for cm in cms)
    return not has_job_config and not has_job_writer and not has_job_sa and not has_service_ca_cm


# ---------------------------------------------------------------------------
# Garak-specific helpers
# ---------------------------------------------------------------------------


def submit_garak_job(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant_namespace: str,
    payload: dict,
) -> str:
    """Submit a garak evaluation job and return the job ID."""
    url = f"https://{host}{EVALHUB_JOBS_PATH}"
    LOGGER.info(f"Submitting garak job to {url}")

    response = requests.post(
        url=url,
        headers=build_headers(token=token, tenant=tenant_namespace),
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )
    if not response.ok:
        LOGGER.error(f"Job submission failed ({response.status_code}): {response.text}")
    response.raise_for_status()

    data = response.json()
    LOGGER.info(f"Garak job submission response: {data}")

    job_id = data.get("id") or data.get("job_id") or (data.get("resource", {}).get("id"))
    assert job_id, f"No job ID in response: {data}"
    return job_id


def wait_for_job_completion(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant_namespace: str,
    job_id: str,
    timeout: int = GARAK_JOB_TIMEOUT,
    poll_interval: int = GARAK_JOB_POLL_INTERVAL,
) -> dict:
    """Poll for garak job completion, returning the final job status."""
    result = wait_for_evalhub_job(
        host=host,
        token=token,
        ca_bundle_file=ca_bundle_file,
        tenant=tenant_namespace,
        job_id=job_id,
        timeout=timeout,
        sleep=poll_interval,
    )
    state = result.get("status", {}).get("state", "")
    assert state == "completed", f"Job {job_id} ended with status '{state}': {result}"
    return result


# ---------------------------------------------------------------------------
# ServiceAccount helpers
# ---------------------------------------------------------------------------


def wait_for_service_account(
    admin_client: DynamicClient,
    namespace: str,
    sa_name: str,
    timeout: int = 360,
) -> ServiceAccount:
    """Wait for a ServiceAccount to be created in the given namespace."""
    LOGGER.info(f"Waiting for ServiceAccount '{sa_name}' in namespace '{namespace}'")

    def _sa_exists() -> ServiceAccount | None:
        try:
            sa = ServiceAccount(client=admin_client, name=sa_name, namespace=namespace)
            if sa.exists:
                return sa
        except (
            ValueError,
            AttributeError,
        ):
            pass
        return None

    for sa in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=_sa_exists,
    ):
        if sa is not None:
            LOGGER.info(f"ServiceAccount '{sa_name}' found in namespace '{namespace}'")
            return sa

    raise TimeoutError(f"ServiceAccount '{sa_name}' not found in namespace '{namespace}' within {timeout}s")
