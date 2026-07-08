from __future__ import annotations

from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError, ResourceNotFoundError
from ocp_resources.service import Service
from timeout_sampler import TimeoutSampler

from utilities.constants import ApiGroups
from utilities.resources.http_route import HTTPRoute

LOGGER = structlog.get_logger(name=__name__)

EXTERNAL_MODEL_NAME = "e2e-external-model"
EXTERNAL_PROVIDER_NAME = "e2e-external-provider"
EXTERNAL_ENDPOINT = "httpbin.org"
EXTERNAL_TARGET_MODEL = "gpt-3.5-turbo"
EXTERNAL_API_FORMAT = "openai-chat"
EXTERNAL_AUTH_POLICY_NAME = "e2e-external-access"
EXTERNAL_SUBSCRIPTION_NAME = "e2e-external-subscription"
EXTERNAL_SECRET_NAME = f"{EXTERNAL_MODEL_NAME}-api-key"
INFERENCE_EXTERNAL_MODEL_API_GROUP = ApiGroups.INFERENCE_OPENDATAHUB_IO


def external_provider_ref(provider_name: str, *, target_model: str = EXTERNAL_TARGET_MODEL) -> dict[str, Any]:
    """Build an externalProviderRefs entry for an ExternalModel spec."""
    return {
        "ref": {"name": provider_name},
        "targetModel": target_model,
        "apiFormat": EXTERNAL_API_FORMAT,
    }


def get_httproute(
    client: DynamicClient,
    name: str,
    namespace: str,
) -> HTTPRoute | None:
    """Look up an HTTPRoute by name/namespace. Returns the resource wrapper or None."""
    try:
        route = HTTPRoute(client=client, name=name, namespace=namespace)
        if route.exists:
            return route
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
) -> HTTPRoute:
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
