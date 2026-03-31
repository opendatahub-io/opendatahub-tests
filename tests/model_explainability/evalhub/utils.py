import requests
import structlog

from tests.model_explainability.evalhub.constants import (
    EVALHUB_BENCHMARKS_PATH,
    EVALHUB_COLLECTIONS_PATH,
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
    EVALHUB_JOBS_PATH,
    EVALHUB_PROVIDERS_PATH,
)
from utilities.guardrails import get_auth_headers

LOGGER = structlog.get_logger(name=__name__)

TENANT_HEADER: str = "X-Tenant"


def _build_headers(token: str, tenant: str | None = None) -> dict[str, str]:
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
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    data = response.json()
    assert data.get("items"), f"Smoke test failed: Providers list is empty for tenant {tenant}"

    return data


def list_evalhub_providers(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
) -> dict:
    """List all available providers for a tenant.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.

    Returns:
        dict: Response containing providers list.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_PROVIDERS_PATH}"
    response = requests.get(
        url=url,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def get_evalhub_provider(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    provider_id: str,
) -> dict:
    """Get details of a specific provider.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.
        provider_id: ID of the provider to retrieve.

    Returns:
        dict: Provider details.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_PROVIDERS_PATH}/{provider_id}"
    response = requests.get(
        url=url,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def create_evalhub_evaluation_job(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    model_url: str,
    model_name: str,
    benchmarks: list[dict[str, str]],
) -> dict:
    """Create an evaluation job.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.
        model_url: URL of the model inference endpoint.
        model_name: Name of the model.
        benchmarks: List of benchmark configurations with 'id' and 'provider_id'.

    Returns:
        dict: Created evaluation job details.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_JOBS_PATH}"
    payload = {
        "model": {
            "url": model_url,
            "name": model_name,
        },
        "benchmarks": benchmarks,
    }

    LOGGER.info(f"Creating evaluation job for tenant {tenant}", payload=payload)

    response = requests.post(
        url=url,
        json=payload,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def get_evalhub_evaluation_job(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    job_id: str,
) -> dict:
    """Get evaluation job status.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.
        job_id: ID of the evaluation job.

    Returns:
        dict: Evaluation job details.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_JOBS_PATH}/{job_id}"
    response = requests.get(
        url=url,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def list_evalhub_evaluation_jobs(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
) -> dict:
    """List all evaluation jobs for a tenant.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.

    Returns:
        dict: Response containing evaluation jobs list.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_JOBS_PATH}"
    response = requests.get(
        url=url,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def list_evalhub_benchmarks(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
) -> dict:
    """List all available benchmarks for a tenant.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.

    Returns:
        dict: Response containing benchmarks list.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_BENCHMARKS_PATH}"
    response = requests.get(
        url=url,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()


def create_evalhub_collection(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str,
    name: str,
    description: str,
) -> dict:
    """Create a benchmark collection.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.
        name: Name of the collection.
        description: Description of the collection.

    Returns:
        dict: Created collection details.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_COLLECTIONS_PATH}"
    payload = {
        "name": name,
        "description": description,
    }

    LOGGER.info(f"Creating collection for tenant {tenant}", payload=payload)

    response = requests.post(
        url=url,
        json=payload,
        headers=_build_headers(token=token, tenant=tenant),
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
    """List all collections for a tenant.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header.

    Returns:
        dict: Response containing collections list.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_COLLECTIONS_PATH}"
    response = requests.get(
        url=url,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()
    return response.json()
