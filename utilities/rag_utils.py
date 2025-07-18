from contextlib import contextmanager
from ocp_resources.resource import NamespacedResource
from kubernetes.dynamic import DynamicClient
from typing import Any, Dict, Generator, List, TypedDict
from llama_stack_client import Agent, AgentEventLogger
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


class LlamaStackDistribution(NamespacedResource):
    api_group: str = "llamastack.io"

    def __init__(self, replicas: int, server: Dict[str, Any], **kwargs: Any):
        """
        Args:
            kwargs: Keyword arguments to pass to the LlamaStackDistribution constructor
        """
        super().__init__(
            **kwargs,
        )
        self.replicas = replicas
        self.server = server

    def to_dict(self) -> None:
        super().to_dict()
        if not self.kind_dict and not self.yaml_file:
            self.res["spec"] = {}
            _spec = self.res["spec"]
            _spec["replicas"] = self.replicas
            _spec["server"] = self.server


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
        teardown=teardown,
    ) as llama_stack_distribution:
        yield llama_stack_distribution


class TurnExpectation(TypedDict):
    question: str
    expected_keywords: List[str]
    description: str


class TurnResult(TypedDict):
    question: str
    description: str
    expected_keywords: List[str]
    found_keywords: List[str]
    missing_keywords: List[str]
    response_content: str
    response_length: int
    event_count: int
    success: bool
    error: str | None


class ValidationSummary(TypedDict):
    total_turns: int
    successful_turns: int
    failed_turns: int
    success_rate: float
    total_events: int
    total_response_length: int


class ValidationResult(TypedDict):
    success: bool
    results: List[TurnResult]
    summary: ValidationSummary


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

        except Exception as e:
            LOGGER.error(f"Error processing turn '{question}': {str(e)}")
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
                "error": str(e),
            }
            all_results.append(turn_result)

    # Generate summary
    summary = {
        "total_turns": total_turns,
        "successful_turns": successful_turns,
        "failed_turns": total_turns - successful_turns,
        "success_rate": successful_turns / total_turns if total_turns > 0 else 0,
        "total_events": sum(result["event_count"] for result in all_results),
        "total_response_length": sum(result["response_length"] for result in all_results),
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

    return {"success": overall_success, "results": all_results, "summary": summary}
