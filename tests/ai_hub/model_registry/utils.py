from aiohttp import ClientConnectionError, ClientResponseError, ServerDisconnectedError
from model_registry import ModelRegistry as ModelRegistryClient
from model_registry.types import RegisteredModel
from timeout_sampler import retry

MR_RETRY_EXCEPTIONS: dict[type[Exception], list[str]] = {
    ClientConnectionError: [],
    ServerDisconnectedError: [],
    ClientResponseError: [],
}


@retry(wait_timeout=60, sleep=5, exceptions_dict=MR_RETRY_EXCEPTIONS)
def get_registered_model_with_retry(client: ModelRegistryClient, name: str) -> RegisteredModel | None:
    """Get a registered model, retrying on transient connection errors."""
    return client.get_registered_model(name=name)
