from typing import Any

from aiohttp import ClientConnectionError, ClientResponseError, ServerDisconnectedError
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel
from mr_openapi.exceptions import ServiceException, UnauthorizedException
from timeout_sampler import retry

MR_RETRY_EXCEPTIONS: dict[type[Exception], list[str]] = {
    ClientConnectionError: [],
    ServerDisconnectedError: [],
    ClientResponseError: [],
    UnauthorizedException: [],
}


@retry(wait_timeout=60, sleep=5, exceptions_dict=MR_RETRY_EXCEPTIONS)
def get_registered_model_with_retry(client: ModelRegistryClient, name: str) -> RegisteredModel | None:
    """Get a registered model, retrying on transient connection errors."""
    return client.get_registered_model(name=name)


@retry(
    wait_timeout=120,
    sleep=5,
    exceptions_dict={ServiceException: [], UnauthorizedException: []},
    print_func_args=False,
)
def get_model_registry_client_with_retry(**kwargs: Any) -> ModelRegistryClient:
    """Build a ModelRegistryClient, retrying transient warm-up errors.

    The client constructor makes a live API call, which can transiently fail
    while the Model Registry warms up:
      - HTTP 503 while the OpenShift route warms up (raised as ServiceException),
        and
      - HTTP 401 while the kube-rbac-proxy sidecar warms up and is not yet
        validating tokens, even after the CR conditions report True (raised as
        UnauthorizedException).

    ``print_func_args`` is kept False so the user token passed via kwargs is never
    logged (CWE-532).

    Args:
        **kwargs: Keyword arguments forwarded to the ModelRegistryClient
            constructor.

    Returns:
        ModelRegistryClient: The constructed client.
    """
    return ModelRegistryClient(**kwargs)
