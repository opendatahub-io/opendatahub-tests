from contextlib import contextmanager
from typing import Any, Callable, Dict, Generator, List, cast

from kubernetes.dynamic import DynamicClient
from llama_stack_client import LlamaStackClient, APIConnectionError
from llama_stack_client.types.vector_store import VectorStore
from ocp_resources.llama_stack_distribution import LlamaStackDistribution
from simple_logger.logger import get_logger
from timeout_sampler import retry

from utilities.constants import Timeout

from tests.llama_stack.constants import (
    TORCHTUNE_TEST_EXPECTATIONS,
    TurnExpectation,
    ModelInfo,
    ValidationResult,
    TurnResult,
)

from llama_stack_client import Agent, AgentEventLogger
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


@retry(wait_timeout=Timeout.TIMEOUT_2MIN, sleep=5)
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


def extract_event_content(event: Any) -> str:
    """Extract content from various event types."""
    for attr in ["content", "message", "text"]:
        if hasattr(event, attr) and getattr(event, attr):
            return str(getattr(event, attr))
    return ""


def validate_rag_agent_responses(
    rag_agent: Agent,
    session_id: str,
    turns_with_expectations: List[TurnExpectation],
    stream: bool = True,
    verbose: bool = True,
    min_keywords_required: int = 1,
    print_events: bool = False,
) -> ValidationResult:
    """
    Validate RAG agent responses against expected keywords.

    Tests multiple questions and validates that responses contain expected keywords.
    Returns validation results with success status and detailed results for each turn.
    """

    all_results = []
    total_turns = len(turns_with_expectations)
    successful_turns = 0

    for turn_idx, turn_data in enumerate(turns_with_expectations, 1):
        question = turn_data["question"]
        expected_keywords = turn_data["expected_keywords"]
        description = turn_data.get("description", "")

        if verbose:
            LOGGER.info(f"[{turn_idx}/{total_turns}] Processing: {question}")
            if description:
                LOGGER.info(f"Expected: {description}")

        # Collect response content for validation
        response_content = ""
        event_count = 0

        try:
            # Create turn with the agent
            stream_response = rag_agent.create_turn(
                messages=[{"role": "user", "content": question}],
                session_id=session_id,
                stream=stream,
            )

            # Process events
            for event in AgentEventLogger().log(stream_response):
                if print_events:
                    event.print()
                event_count += 1

                # Extract content from different event types
                response_content += extract_event_content(event)

            # Validate response content
            response_lower = response_content.lower()
            found_keywords = []
            missing_keywords = []

            for keyword in expected_keywords:
                if keyword.lower() in response_lower:
                    found_keywords.append(keyword)
                else:
                    missing_keywords.append(keyword)

            # Determine if this turn was successful
            turn_successful = (
                event_count > 0 and len(response_content) > 0 and len(found_keywords) >= min_keywords_required
            )

            if turn_successful:
                successful_turns += 1

            # Store results for this turn
            turn_result = {
                "question": question,
                "description": description,
                "expected_keywords": expected_keywords,
                "found_keywords": found_keywords,
                "missing_keywords": missing_keywords,
                "response_content": response_content,
                "response_length": len(response_content),
                "event_count": event_count,
                "success": turn_successful,
                "error": None,
            }

            all_results.append(turn_result)

            if verbose:
                LOGGER.info(f"Response length: {len(response_content)}")
                LOGGER.info(f"Events processed: {event_count}")
                LOGGER.info(f"Found keywords: {found_keywords}")

                if missing_keywords:
                    LOGGER.warning(f"Missing expected keywords: {missing_keywords}")

                if turn_successful:
                    LOGGER.info(f"✓ Successfully validated response for: {question}")
                else:
                    LOGGER.error(f"✗ Validation failed for: {question}")

                if turn_idx < total_turns:  # Don't print separator after last turn
                    LOGGER.info("-" * 50)

        except Exception as exc:
            LOGGER.exception("Error processing turn %s", question)
            turn_result = {
                "question": question,
                "description": description,
                "expected_keywords": expected_keywords,
                "found_keywords": [],
                "missing_keywords": expected_keywords,
                "response_content": "",
                "response_length": 0,
                "event_count": 0,
                "success": False,
                "error": str(exc),
            }
            all_results.append(turn_result)

    # Generate summary
    summary = {
        "total_turns": total_turns,
        "successful_turns": successful_turns,
        "failed_turns": total_turns - successful_turns,
        "success_rate": successful_turns / total_turns if total_turns > 0 else 0,
        "total_events": sum(cast(TurnResult, result)["event_count"] for result in all_results),
        "total_response_length": sum(cast(TurnResult, result)["response_length"] for result in all_results),
    }

    overall_success = successful_turns == total_turns

    if verbose:
        LOGGER.info("=" * 60)
        LOGGER.info("VALIDATION SUMMARY:")
        LOGGER.info(f"Total turns: {summary['total_turns']}")
        LOGGER.info(f"Successful: {summary['successful_turns']}")
        LOGGER.info(f"Failed: {summary['failed_turns']}")
        LOGGER.info(f"Success rate: {summary['success_rate']:.1%}")
        LOGGER.info(f"Overall result: {'✓ PASSED' if overall_success else '✗ FAILED'}")

    return cast(ValidationResult, {"success": overall_success, "results": all_results, "summary": summary})


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
    wait_timeout=Timeout.TIMEOUT_2MIN,
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

        # Save file locally first and pretend it's a txt file, not sure why this is needed
        # but it works locally without it,
        # though llama stack version is the newer one.
        file_name = url.split("/")[-1]
        local_file_name = file_name.replace(".rst", ".txt")
        with tempfile.NamedTemporaryFile(mode="wb", suffix=f"_{local_file_name}") as temp_file:
            temp_file.write(response.content)
            temp_file_path = temp_file.name

            # Upload saved file to LlamaStack
            with open(temp_file_path, "rb") as file_to_upload:
                uploaded_file = llama_stack_client.files.create(file=file_to_upload, purpose="assistants")

            # Add file to vector store
            llama_stack_client.vector_stores.files.create(vector_store_id=vector_store.id, file_id=uploaded_file.id)

        return True

    except (requests.exceptions.RequestException, Exception) as e:
        LOGGER.warning(f"Failed to download and upload file {url}: {e}")
        raise
