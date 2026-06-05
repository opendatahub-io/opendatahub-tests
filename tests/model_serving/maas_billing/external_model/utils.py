from __future__ import annotations

from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

LOGGER = structlog.get_logger(name=__name__)

EXTERNAL_MODEL_NAME = "e2e-external-model"
EXTERNAL_ENDPOINT = "httpbin.org"
EXTERNAL_AUTH_POLICY_NAME = "e2e-external-access"
EXTERNAL_SUBSCRIPTION_NAME = "e2e-external-subscription"
EXTERNAL_SECRET_NAME = f"{EXTERNAL_MODEL_NAME}-api-key"


def get_httproute(
    client: DynamicClient,
    name: str,
    namespace: str,
) -> dict[str, Any] | None:
    """Look up an HTTPRoute by name/namespace. Returns the resource dict or None."""
    try:
        api = client.resources.get(
            api_version="gateway.networking.k8s.io/v1",
            kind="HTTPRoute",
        )
        route = api.get(name=name, namespace=namespace)
        return route.to_dict() if route else None
    except NotFoundError, ResourceNotFoundError:
        LOGGER.debug(f"HTTPRoute {namespace}/{name} not found")
    return None


def get_service(
    client: DynamicClient,
    name: str,
    namespace: str,
) -> Service | None:
    """Look up a Service by name/namespace. Returns None if not found."""
    try:
        svc = Service(client=client, name=name, namespace=namespace)
        if svc.exists:
            return svc
    except NotFoundError, ResourceNotFoundError:
        LOGGER.debug(f"Service {namespace}/{name} not found")
    return None


def wait_for_httproute(
    client: DynamicClient,
    name: str,
    namespace: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """Poll until the HTTPRoute exists, or raise on timeout."""
    for _ in TimeoutSampler(
        wait_timeout=timeout,
        sleep=3,
        func=get_httproute,
        client=client,
        name=name,
        namespace=namespace,
    ):
        route = get_httproute(client=client, name=name, namespace=namespace)
        if route is not None:
            return route

    raise TimeoutError(f"HTTPRoute {namespace}/{name} not found within {timeout}s")


def wait_for_httproute_deleted(
    client: DynamicClient,
    name: str,
    namespace: str,
    timeout: int = 60,
) -> None:
    """Poll until the HTTPRoute no longer exists, or raise on timeout."""
    for _ in TimeoutSampler(
        wait_timeout=timeout,
        sleep=3,
        func=get_httproute,
        client=client,
        name=name,
        namespace=namespace,
    ):
        if get_httproute(client=client, name=name, namespace=namespace) is None:
            return

    raise TimeoutError(f"HTTPRoute {namespace}/{name} still exists after {timeout}s")
