from typing import Any

import requests
from simple_logger.logger import get_logger

from tests.model_explainability.evalhub.constants import (
    EVALHUB_HEALTH_PATH,
    EVALHUB_HEALTH_STATUS_HEALTHY,
    EVALHUB_PROVIDERS_PATH,
)
from utilities.guardrails import get_auth_headers

LOGGER = get_logger(name=__name__)

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


def list_evalhub_providers(
    host: str,
    token: str,
    ca_bundle_file: str,
    tenant: str | None = None,
) -> dict[str, Any]:
    """List all providers from the EvalHub service.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        tenant: Namespace for the X-Tenant header. Required for SAR authorisation.

    Returns:
        Parsed JSON response containing items and pagination.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_PROVIDERS_PATH}"
    LOGGER.info(f"Listing EvalHub providers at {url}")

    response = requests.get(
        url=url,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    data: dict[str, Any] = response.json()
    LOGGER.info(f"Listed {data.get('total_count', 0)} providers")
    return data


def get_evalhub_provider(
    host: str,
    token: str,
    ca_bundle_file: str,
    provider_id: str,
    tenant: str | None = None,
) -> dict[str, Any]:
    """Get a single provider by ID from the EvalHub service.

    Args:
        host: Route host for the EvalHub service.
        token: Bearer token for authentication.
        ca_bundle_file: Path to CA bundle for TLS verification.
        provider_id: ID of the provider to retrieve.
        tenant: Namespace for the X-Tenant header. Required for SAR authorisation.

    Returns:
        Parsed JSON response containing the provider resource.

    Raises:
        requests.HTTPError: If the request fails.
    """
    url = f"https://{host}{EVALHUB_PROVIDERS_PATH}/{provider_id}"
    LOGGER.info(f"Getting EvalHub provider {provider_id}")

    response = requests.get(
        url=url,
        headers=_build_headers(token=token, tenant=tenant),
        verify=ca_bundle_file,
        timeout=10,
    )
    response.raise_for_status()

    data: dict[str, Any] = response.json()
    return data
