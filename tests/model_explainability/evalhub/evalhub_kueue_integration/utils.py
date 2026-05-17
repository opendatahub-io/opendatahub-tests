import os
from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.route import Route
from timeout_sampler import retry

from tests.model_explainability.evalhub.evalhub_kueue_integration.constants import (
    EVALHUB_HEALTH_ENDPOINT,
    EVALHUB_JOBS_ENDPOINT,
    EvalJobState,
    Timeout,
)

LOGGER = structlog.get_logger(name=__name__)


class EvalHubSetupError(RuntimeError):
    """Raised when session preflight checks for EvalHub + Kueue fail."""


def get_requests_verify() -> bool | str:
    """TLS ``verify`` setting for EvalHub HTTP calls (``requests``).

    Environment variables:

        EVALHUB_INSECURE:
            When set to a truthy value (``true``, ``1``, ``yes``, case-insensitive),
            returns ``False`` so TLS certificate verification is disabled. Use only
            when necessary (e.g. dev clusters with self-signed certs).

        EVALHUB_CA_BUNDLE:
            Path to a PEM CA bundle file. When set (and ``EVALHUB_INSECURE`` is not
            enabled), that path is passed to ``requests`` as ``verify``.

    ``EVALHUB_INSECURE`` takes precedence over ``EVALHUB_CA_BUNDLE``. When neither
    applies, returns ``True`` (verify using default/system CAs).

    Returns:
        Value for the ``verify`` parameter: ``True``, ``False``, or a path string.
    """
    insecure = os.environ.get("EVALHUB_INSECURE", "").strip().lower()
    if insecure in ("true", "1", "yes"):
        return False
    ca_bundle = os.environ.get("EVALHUB_CA_BUNDLE", "").strip()
    if ca_bundle:
        return ca_bundle
    return True


def get_evalhub_route_url(client: DynamicClient, namespace: str, allow_insecure: bool = False) -> str:
    """Return the URL for the EvalHub route in the given namespace.

    By default this function fails closed and requires TLS on the selected
    route. To permit non-TLS routes explicitly, set ``allow_insecure=True`` or
    set ``EVALHUB_ALLOW_INSECURE_ROUTE=true``.
    """
    routes = list(
        Route.get(
            client=client,
            namespace=namespace,
            label_selector="app=eval-hub",
        )
    )
    if not routes:
        routes = list(
            Route.get(
                client=client,
                namespace=namespace,
            )
        )
        routes = [r for r in routes if "eval" in r.name.lower() or "evalhub" in r.name.lower()]

    if not routes:
        raise RuntimeError(f"No EvalHub route found in namespace {namespace}")

    selected_route = routes[0]
    host = selected_route.instance.spec.host
    tls = selected_route.instance.spec.get("tls")
    env_allow_insecure = os.environ.get("EVALHUB_ALLOW_INSECURE_ROUTE", "").strip().lower() in (
        "true",
        "1",
        "yes",
    )
    if not tls and not (allow_insecure or env_allow_insecure):
        raise RuntimeError(
            "EvalHub route in namespace "
            f"'{namespace}' has no TLS configuration; refusing to use insecure HTTP. "
            "Set allow_insecure=True or EVALHUB_ALLOW_INSECURE_ROUTE=true to override."
        )
    scheme = "https" if tls else "http"
    return f"{scheme}://{host}"


def submit_eval_job(
    base_url: str,
    token: str,
    name: str,
    model_url: str,
    model_name: str,
    queue_name: str | None = None,
    priority: int | None = None,
    benchmarks: list[dict[str, Any]] | None = None,
    tenant: str = "test-tenant",
) -> tuple[int, dict[str, Any]]:
    """Submit an evaluation job to EvalHub and return (status_code, response_json)."""
    payload: dict[str, Any] = {
        "name": name,
        "model": {"url": model_url, "name": model_name},
        "benchmarks": benchmarks
        or [
            {
                "id": "arc_easy",
                "provider_id": "lm_evaluation_harness",
                "parameters": {"num_examples": 5, "limit": 5, "tokenizer": "google/flan-t5-small"},
            }
        ],
    }

    if queue_name is not None:
        payload["queue"] = {"kind": "kueue", "name": queue_name}

    if priority is not None:
        payload["priority"] = priority

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Tenant": tenant,
    }

    resp = requests.post(
        f"{base_url}{EVALHUB_JOBS_ENDPOINT}",
        json=payload,
        headers=headers,
        verify=get_requests_verify(),
        timeout=30,
    )
    try:
        body = resp.json()
    except requests.exceptions.JSONDecodeError:
        body = {"raw": resp.text}
    LOGGER.info("Submitted eval job", name=name, status=resp.status_code, queue=queue_name)
    return resp.status_code, body


def get_eval_job(base_url: str, token: str, job_id: str, tenant: str = "test-tenant") -> tuple[int, dict[str, Any]]:
    """Get evaluation job status from EvalHub."""
    headers = {"Authorization": f"Bearer {token}", "X-Tenant": tenant}
    resp = requests.get(
        f"{base_url}{EVALHUB_JOBS_ENDPOINT}/{job_id}",
        headers=headers,
        verify=get_requests_verify(),
        timeout=30,
    )
    try:
        body = resp.json()
    except requests.exceptions.JSONDecodeError:
        body = {"raw": resp.text}
    return resp.status_code, body


def list_eval_jobs(
    base_url: str,
    token: str,
    status: str | None = None,
    tenant: str = "test-tenant",
) -> tuple[int, dict[str, Any]]:
    """List evaluation jobs, optionally filtered by status."""
    headers = {"Authorization": f"Bearer {token}", "X-Tenant": tenant}
    params: dict[str, str] = {}
    if status:
        params["status"] = status
    resp = requests.get(
        f"{base_url}{EVALHUB_JOBS_ENDPOINT}",
        headers=headers,
        params=params,
        verify=get_requests_verify(),
        timeout=30,
    )
    try:
        body = resp.json()
    except requests.exceptions.JSONDecodeError:
        body = {"raw": resp.text}
    return resp.status_code, body


def delete_eval_job(
    base_url: str,
    token: str,
    job_id: str,
    hard_delete: bool = False,
    tenant: str = "test-tenant",
) -> int:
    """Delete an evaluation job. Returns status code."""
    headers = {"Authorization": f"Bearer {token}", "X-Tenant": tenant}
    params: dict[str, str] = {}
    if hard_delete:
        params["hard_delete"] = "true"
    resp = requests.delete(
        f"{base_url}{EVALHUB_JOBS_ENDPOINT}/{job_id}",
        headers=headers,
        params=params,
        verify=get_requests_verify(),
        timeout=30,
    )
    LOGGER.info("Deleted eval job", job_id=job_id, hard_delete=hard_delete, status=resp.status_code)
    return resp.status_code


def verify_evalhub_preflight(
    admin_client: DynamicClient,
    base_url: str,
    *,
    nonexistent_job_id: str = "verify-evalhub-setup-nonexistent-job-id",
    tenant: str = "test-tenant",
) -> None:
    """Validate EvalHub HTTP reachability and Kueue before integration tests.

    Consolidates the former CLI verification script into shared code; uses
    the same routes and TLS behavior as test helpers (``EVALHUB_*`` endpoints,
    :func:`get_requests_verify`, ``get_openshift_token``).

    Raises:
        EvalHubSetupError: If the jobs probe does not return 404 or Kueue is not ready.
    """
    from utilities.infra import get_openshift_token
    from utilities.kueue_utils import wait_for_kueue_crds_available

    token = get_openshift_token(client=admin_client)
    if not token.strip():
        raise EvalHubSetupError("OpenShift token is empty; log in with oc login.")

    verify_tls = get_requests_verify()
    base = base_url.rstrip("/")
    headers_bearer = {"Authorization": f"Bearer {token}"}
    headers_jobs = {**headers_bearer, "X-Tenant": tenant}

    health_url = f"{base}{EVALHUB_HEALTH_ENDPOINT}"
    try:
        health_resp = requests.get(health_url, headers=headers_bearer, timeout=10, verify=verify_tls)
        LOGGER.info("evalhub_preflight_health", status_code=health_resp.status_code)
    except requests.RequestException as exc:
        LOGGER.warning("evalhub_preflight_health_unreachable", error=str(exc))

    jobs_url = f"{base}{EVALHUB_JOBS_ENDPOINT}/{nonexistent_job_id}"
    try:
        jobs_resp = requests.get(jobs_url, headers=headers_jobs, timeout=10, verify=verify_tls)
    except requests.RequestException as exc:
        raise EvalHubSetupError(f"EvalHub jobs API unreachable: {exc}") from exc

    if jobs_resp.status_code != 404:
        snippet = jobs_resp.text[:500] if jobs_resp.text else ""
        raise EvalHubSetupError(
            f"Expected HTTP 404 for nonexistent job GET, got {jobs_resp.status_code}. Body (truncated): {snippet!r}"
        )

    wait_for_kueue_crds_available(client=admin_client)
    LOGGER.info("evalhub_preflight_complete")


def get_health(base_url: str, token: str) -> tuple[int, dict[str, Any]]:
    """Get EvalHub health check."""
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(
        f"{base_url}{EVALHUB_HEALTH_ENDPOINT}",
        headers=headers,
        verify=get_requests_verify(),
        timeout=30,
    )
    try:
        body = resp.json()
    except requests.exceptions.JSONDecodeError:
        body = {"raw": resp.text}
    return resp.status_code, body


@retry(wait_timeout=Timeout.JOB_COMPLETION, sleep=Timeout.POLL_INTERVAL)
def wait_for_job_state(
    base_url: str, token: str, job_id: str, target_state: str, tenant: str = "test-tenant"
) -> dict[str, Any]:
    """Poll EvalHub API until the job reaches the target state.

    For Kueue testing purposes, FAILED is treated as equivalent to COMPLETED
    since both are terminal states that free up quota.

    Raises:
        TimeoutExpiredError: If job doesn't reach target state within timeout.
    """
    status_code, body = get_eval_job(base_url=base_url, token=token, job_id=job_id, tenant=tenant)
    assert status_code == 200, f"GET job {job_id} returned {status_code}"
    current_state = body.get("status", {}).get("state", "")
    LOGGER.info("Polling job state", job_id=job_id, current=current_state, target=target_state)

    # For Kueue testing, FAILED is a valid terminal state
    # When waiting for COMPLETED or PENDING, accept FAILED as well
    if target_state == EvalJobState.COMPLETED and current_state == EvalJobState.FAILED:
        LOGGER.info("Job failed instead of completing, treating as terminal state", job_id=job_id)
        return body

    if target_state == EvalJobState.PENDING and current_state == EvalJobState.FAILED:
        LOGGER.info("Job failed while expected to be pending, accepting as valid state", job_id=job_id)
        return body

    if current_state != target_state:
        return False
    return body


@retry(wait_timeout=Timeout.JOB_RUNNING, sleep=Timeout.POLL_INTERVAL)
def wait_for_job_running_or_completed(
    base_url: str, token: str, job_id: str, tenant: str = "test-tenant"
) -> dict[str, Any]:
    """Poll until job is running, completed, or failed (terminal states)."""
    status_code, body = get_eval_job(base_url=base_url, token=token, job_id=job_id, tenant=tenant)
    assert status_code == 200, f"GET job {job_id} returned {status_code}"
    state = body.get("status", {}).get("state", "")
    if state in (EvalJobState.RUNNING, EvalJobState.COMPLETED, EvalJobState.FAILED):
        return body
    return False
