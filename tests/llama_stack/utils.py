from contextlib import contextmanager
from typing import Dict, Any, Generator, List, Callable

from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient, APIConnectionError
from llama_stack_client.types.vector_store import VectorStore
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.constants import Timeout
from utilities.rag_utils import TurnExpectation, ModelInfo
from tests.llama_stack.constants import TORCHTUNE_TEST_EXPECTATIONS


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


def get_torchtune_test_expectations() -> List[TurnExpectation]:
    """
    Helper function to get the test expectations for TorchTune documentation questions.

    Returns:
        List of TurnExpectation objects for testing RAG responses
    """
    return [
        {
            "question": expectation.question,
            "expected_keywords": expectation.expected_keywords,
            "description": expectation.description,
        }
        for expectation in TORCHTUNE_TEST_EXPECTATIONS
    ]


def create_response_function(
    llama_stack_client: LlamaStackClient, llama_stack_models: ModelInfo, vector_store: VectorStore
) -> Callable:
    """
    Helper function to create a response function for testing with vector store integration.

    Args:
        llama_stack_client: The LlamaStack client instance
        llama_stack_models: The model configuration
        vector_store: The vector store instance

    Returns:
        A callable function that takes a question and returns a response
    """

    def _response_fn(*, question: str) -> str:
        response = llama_stack_client.responses.create(
            input=question,
            model=llama_stack_models.model_id,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id],
                }
            ],
        )
        return response.output_text

    return _response_fn
