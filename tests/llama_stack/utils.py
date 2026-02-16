from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, cast

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from llama_stack_client import LlamaStackClient, APIConnectionError, InternalServerError
from llama_stack_client.types.vector_store import VectorStore

from utilities.resources.llama_stack_distribution import LlamaStackDistribution
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.exceptions import UnexpectedResourceCountError


from tests.llama_stack.constants import (
    TORCHTUNE_TEST_EXPECTATIONS,
    TurnExpectation,
    ModelInfo,
    ValidationResult,
    LLS_CORE_POD_FILTER,
)

import os
import tempfile

import requests


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

    # Starting with RHOAI 3.3, pods in the 'openshift-ingress' namespace must be allowed
    # to access the llama-stack-service. This is required for the llama_stack_test_route
    # to function properly.
    network: Dict[str, Any] = {
        "allowedFrom": {
            "namespaces": ["openshift-ingress"],
        },
    }

    with LlamaStackDistribution(
        client=client,
        name=name,
        namespace=namespace,
        replicas=replicas,
        network=network,
        server=server,
        wait_for_resource=True,
        teardown=teardown,
    ) as llama_stack_distribution:
        yield llama_stack_distribution


@retry(
    wait_timeout=60,
    sleep=5,
    exceptions_dict={ResourceNotFoundError: [], UnexpectedResourceCountError: []},
)
def wait_for_unique_llama_stack_pod(client: DynamicClient, namespace: str) -> Pod:
    """Wait until exactly one LlamaStackDistribution pod is found in the
    namespace (multiple pods may indicate known bug RHAIENG-1819)."""
    pods = list(
        Pod.get(
            client=client,
            namespace=namespace,
            label_selector=LLS_CORE_POD_FILTER,
        )
    )
    if not pods:
        raise ResourceNotFoundError(f"No pods found with label selector {LLS_CORE_POD_FILTER} in namespace {namespace}")
    if len(pods) != 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly 1 pod with label selector {LLS_CORE_POD_FILTER} "
            f"in namespace {namespace}, found {len(pods)}. "
            f"(possibly due to known bug RHAIENG-1819)"
        )
    return pods[0]


@retry(wait_timeout=90, sleep=5)
def wait_for_llama_stack_client_ready(client: LlamaStackClient) -> bool:
    """Wait for LlamaStack client to be ready by checking health, version, and database access."""
    try:
        client.inspect.health()
        version = client.inspect.version()
        models = client.models.list()
        vector_stores = client.vector_stores.list()
        files = client.files.list()
        LOGGER.info(
            f"Llama Stack server is available! "
            f"(version:{version.version} "
            f"models:{len(models)} "
            f"vector_stores:{len(vector_stores.data)} "
            f"files:{len(files.data)})"
        )
        return True

    except (APIConnectionError, InternalServerError) as error:
        LOGGER.debug(f"Llama Stack server not ready yet: {error}")
        LOGGER.debug(f"Base URL: {client.base_url}, Error type: {type(error)}, Error details: {str(error)}")
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
            stream=False,
            tools=[
                {
                    "type": "file_search",
                    "vector_store_ids": [vector_store.id],
                }
            ],
        )
        return response.output_text

    return _response_fn


def validate_api_responses(
    response_fn: Callable[..., str],
    test_cases: List[TurnExpectation],
    min_keywords_required: int = 1,
) -> ValidationResult:
    """
    Validate API responses against expected keywords.

    Tests multiple questions and validates that responses contain expected keywords.
    Returns validation results with success status and detailed results for each turn.
    """
    all_results = []
    successful = 0

    for idx, test in enumerate(test_cases, 1):
        question = test["question"]
        expected_keywords = test["expected_keywords"]
        description = test.get("description", "")

        LOGGER.debug(f"\n[{idx}] Question: {question}")
        if description:
            LOGGER.debug(f"    Expectation: {description}")

        try:
            response = response_fn(question=question)
            response_lower = response.lower()

            found = [kw for kw in expected_keywords if kw.lower() in response_lower]
            missing = [kw for kw in expected_keywords if kw.lower() not in response_lower]
            success = len(found) >= min_keywords_required

            if success:
                successful += 1

            result = {
                "question": question,
                "description": description,
                "expected_keywords": expected_keywords,
                "found_keywords": found,
                "missing_keywords": missing,
                "response_content": response,
                "response_length": len(response) if isinstance(response, str) else 0,
                "event_count": len(response.events) if hasattr(response, "events") else 0,
                "success": success,
                "error": None,
            }

            all_results.append(result)

            LOGGER.debug(f"✓ Found: {found}")
            if missing:
                LOGGER.debug(f"✗ Missing: {missing}")
            LOGGER.info(f"[{idx}] Result: {'PASS' if success else 'FAIL'}")

        except Exception as e:
            all_results.append({
                "question": question,
                "description": description,
                "expected_keywords": expected_keywords,
                "found_keywords": [],
                "missing_keywords": expected_keywords,
                "response_content": "",
                "response_length": 0,
                "event_count": 0,
                "success": False,
                "error": str(e),
            })
            LOGGER.exception(f"[{idx}] ERROR")

    total = len(test_cases)
    summary = {
        "total": total,
        "passed": successful,
        "failed": total - successful,
        "success_rate": successful / total if total > 0 else 0,
    }

    LOGGER.info("\n" + "=" * 40)
    LOGGER.info("Validation Summary:")
    LOGGER.info(f"Total: {summary['total']}")
    LOGGER.info(f"Passed: {summary['passed']}")
    LOGGER.info(f"Failed: {summary['failed']}")
    LOGGER.info(f"Success rate: {summary['success_rate']:.1%}")

    return cast("ValidationResult", {"success": successful == total, "results": all_results, "summary": summary})


@retry(
    wait_timeout=240,
    sleep=15,
    exceptions_dict={requests.exceptions.RequestException: [], Exception: []},
)
def vector_store_create_file_from_url(url: str, llama_stack_client: LlamaStackClient, vector_store: Any) -> bool:
    """
    Downloads a file from URL to a temporally file and uploads it to the files provider (files.create)
    and to the vector_store (vector_stores.files.create)

    Args:
        url: The URL to download the file from
        llama_stack_client: The configured LlamaStackClient
        vector_store: The vector store to upload the file to

    Returns:
        bool: True if successful, raises exception if failed
    """
    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        path_part = url.split("/")[-1].split("?")[0]

        if content_type == "application/pdf" or path_part.lower().endswith(".pdf"):
            file_suffix = ".pdf"
        elif path_part.lower().endswith(".rst"):
            file_suffix = "_" + path_part.replace(".rst", ".txt")
        else:
            file_suffix = "_" + (path_part or "document.txt")

        with tempfile.NamedTemporaryFile(mode="wb", suffix=file_suffix, delete=False) as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

        try:
            # Upload saved file to LlamaStack (filename extension used for parsing)
            with open(temp_file_path, "rb") as file_to_upload:
                uploaded_file = llama_stack_client.files.create(file=file_to_upload, purpose="assistants")

            # Add file to vector store
            llama_stack_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=uploaded_file.id)
            return True
        finally:
            os.unlink(temp_file_path)

    except (requests.exceptions.RequestException, Exception) as e:
        LOGGER.warning(f"Failed to download and upload file {url}: {e}")
        raise
