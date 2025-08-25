from contextlib import contextmanager
from typing import Dict, Any, Generator

from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient, APIConnectionError
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.constants import Timeout


LOGGER = get_logger(name=__name__)


@contextmanager
def create_llama_stack_distribution(
    client: DynamicClient,
    name: str,
    namespace: str,
    replicas: int,
    server: Dict[str, Any],
    teardown: bool = True,
) -> Generator[LlamaStackDistribution, Any, Any]:
    """
    Context manager to create and optionally delete a LLama Stack Distribution
    """
    with LlamaStackDistribution(
        client=client,
        name=name,
        namespace=namespace,
        replicas=replicas,
        server=server,
        wait_for_resource=True,
        teardown=teardown,
    ) as llama_stack_distribution:
        yield llama_stack_distribution


@retry(wait_timeout=Timeout.TIMEOUT_1MIN, sleep=5)
def wait_for_llama_stack_client_ready(client: LlamaStackClient) -> bool:
    try:
        client.inspect.health()
        version = client.inspect.version()
        LOGGER.info(f"Llama Stack server (v{version.version}) is available!")
        return True
    except APIConnectionError as e:
        LOGGER.debug(f"Llama Stack server not ready yet: {e}")
        return False
    except Exception as e:
        LOGGER.warning(f"Unexpected error checking Llama Stack readiness: {e}")
        return False
