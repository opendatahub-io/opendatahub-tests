import json
import os
import re
from contextlib import contextmanager
from typing import Generator, Any
from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from tests.model_serving.model_runtime.vllm.constant import CHAT_QUERY, COMPLETION_QUERY
from tenacity import retry, stop_after_attempt, wait_exponential

from utilities.constants import Ports
from utilities.exceptions import NotSupportedError
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient
from utilities.plugins.tgis_grpc_plugin import TGISGRPCPlugin
from tests.model_serving.model_runtime.vllm.constant import VLLM_SUPPORTED_QUANTIZATION
from tests.model_serving.model_runtime.vllm.constant import (
    OPENAI_ENDPOINT_NAME,
    TGIS_ENDPOINT_NAME,
)
import portforward
import pytest

LOGGER = get_logger(name=__name__)


class InferenceValidationError(Exception):
    """Raised when contract-based (fuzzy) inference validation fails."""


class EmptyResponseTextError(InferenceValidationError):
    """Raised when a response text is empty (tolerated at list level, min 1 valid required)."""


ALPHABETIC_WORD = re.compile(r"[a-zA-Z]{2,}")
# Conservative failure indicators: patterns a broken runtime emits into the response text
# itself. Generic words like "error" or exception names are intentionally excluded because
# code-generation answers legitimately mention them.
ERROR_INDICATORS = re.compile(
    r"traceback|cuda\s*error|out\s*of\s*memory|segmentation\s*fault|segfault|core\s*dumped|cuda_error",
    re.IGNORECASE,
)
VALID_FINISH_REASONS = ["stop", "length", "eos_token", "stop_sequence", "tool_calls"]
# TGIS (fmaas) stop reasons: 2 = end of sequence, 6 = max new tokens reached, 7 = error
TGIS_VALID_STOP_REASONS = (2, 6)
TGIS_ERROR_STOP_REASON = 7


def _validate_response_text(text: str, context: str) -> None:
    if not text or not text.strip():
        raise EmptyResponseTextError(f"{context}: text is empty or whitespace-only")
    # Recognizable content: a 2+ letter word, a digit (numeric answers e.g. math), or CJK text
    has_words = (
        bool(ALPHABETIC_WORD.search(text))
        or any(char.isdigit() for char in text)
        or any(ord(char) > 0x2E80 for char in text)
    )
    if not has_words:
        raise InferenceValidationError(f"{context}: text contains no recognizable words. Text: '{text[:100]}'")
    if error_match := ERROR_INDICATORS.search(text):
        raise InferenceValidationError(
            f"{context}: text contains error indicator '{error_match.group()}'. Text: '{text[:200]}'"
        )


def _validate_tool_calls(message: dict[str, Any], context: str) -> None:
    tool_calls = message.get("tool_calls")
    if not tool_calls or not isinstance(tool_calls, list):
        raise InferenceValidationError(f"{context}: message has no tool_calls")
    for tc_idx, tool_call in enumerate(tool_calls):
        tc_context = f"{context}.tool_calls[{tc_idx}]"
        if not isinstance(tool_call, dict):
            raise InferenceValidationError(f"{tc_context}: expected dict, got {type(tool_call).__name__}")
        function = tool_call.get("function") or {}
        name = function.get("name")
        if not name or not isinstance(name, str):
            raise InferenceValidationError(f"{tc_context}: missing function name")
        arguments = function.get("arguments")
        if arguments is None:
            raise InferenceValidationError(f"{tc_context}: missing function arguments")
        if isinstance(arguments, str):
            try:
                json.loads(arguments)
            except json.JSONDecodeError as err:
                raise InferenceValidationError(f"{tc_context}: function arguments are not valid JSON: {err}") from err
        elif not isinstance(arguments, dict):
            raise InferenceValidationError(
                f"{tc_context}: function arguments must be a JSON string or dict, got {type(arguments).__name__}"
            )
        LOGGER.info(f"{tc_context}: valid tool call '{name}'")


def _validate_openai_choice(choice: Any, context: str) -> None:
    if not isinstance(choice, dict):
        raise InferenceValidationError(f"{context}: expected dict choice, got {type(choice).__name__}")
    finish_reason = (choice.get("finish_reason") or "").lower() if isinstance(choice.get("finish_reason"), str) else ""
    if finish_reason and finish_reason not in VALID_FINISH_REASONS:
        LOGGER.warning(f"{context}: unexpected finish_reason '{finish_reason}' (expected one of {VALID_FINISH_REASONS})")
    message = choice.get("message")
    if isinstance(message, dict):
        if message.get("tool_calls"):
            _validate_tool_calls(message=message, context=f"{context}.message")
            return
        _validate_response_text(text=message.get("content") or "", context=f"{context}.message.content")
        return
    if "text" in choice:
        _validate_response_text(text=choice["text"] or "", context=f"{context}.text")
        return
    raise InferenceValidationError(f"{context}: choice has neither 'message' nor 'text' field")


def _validate_tgis_response(response: Any, context: str) -> None:
    if not isinstance(response, dict):
        raise InferenceValidationError(f"{context}: expected dict, got {type(response).__name__}")
    _validate_response_text(text=response.get("output_text") or "", context=f"{context}.output_text")
    output_tokens = response.get("output_tokens")
    if isinstance(output_tokens, int) and output_tokens <= 0:
        raise InferenceValidationError(f"{context}: output_tokens={output_tokens}")
    stop_reason = response.get("stop_reason")
    if stop_reason == TGIS_ERROR_STOP_REASON:
        raise InferenceValidationError(f"{context}: tgis stop_reason={stop_reason} (error)")
    if stop_reason is not None and stop_reason not in TGIS_VALID_STOP_REASONS:
        LOGGER.warning(f"{context}: unexpected tgis stop_reason {stop_reason}")


def _validate_response_item(item: Any, context: str) -> None:
    if item is None:
        raise InferenceValidationError(f"{context}: response item is None")
    if isinstance(item, str):
        _validate_response_text(text=item, context=context)
        return
    if isinstance(item, list):
        _validate_response_list(items=item, context=context)
        return
    if isinstance(item, dict):
        if "choices" in item:
            choices = item["choices"]
            if not isinstance(choices, list) or not choices:
                raise InferenceValidationError(f"{context}: 'choices' is missing or empty")
            for idx, choice in enumerate(choices):
                _validate_openai_choice(choice=choice, context=f"{context}.choices[{idx}]")
            return
        if "message" in item or "text" in item:
            _validate_openai_choice(choice=item, context=context)
            return
        if "output_text" in item:
            _validate_tgis_response(response=item, context=context)
            return
        if "embedding" in item:
            embedding = item["embedding"]
            if not isinstance(embedding, list) or not embedding:
                raise InferenceValidationError(f"{context}: embedding is empty")
            return
        if not item:
            raise InferenceValidationError(f"{context}: empty dict response")
        return
    # Non-standard objects (e.g. gRPC/protobuf ModelInfoResponse) - accept as-is
    LOGGER.debug(f"{context}: passing through non-standard response type {type(item).__name__}")


def _validate_response_list(items: list[Any], context: str) -> None:
    """Validate a list of responses, tolerating empty texts as long as at least one
    response is valid (models legitimately emit EOS as first token for some prompts,
    depending on hardware/kernel)."""
    valid_count = 0
    for idx, item in enumerate(items):
        try:
            _validate_response_item(item=item, context=f"{context}[{idx}]")
            valid_count += 1
        except EmptyResponseTextError as err:
            LOGGER.warning(f"{context}[{idx}]: empty response text, skipping ({err})")
    if items and valid_count == 0:
        raise InferenceValidationError(f"{context}: no valid responses in list of {len(items)}")


@contextmanager
def kserve_s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={
            "serving.kserve.io/s3-endpoint": aws_s3_endpoint.replace("https://", ""),
            "serving.kserve.io/s3-region": aws_s3_region,
            "serving.kserve.io/s3-useanoncredential": "false",
            "serving.kserve.io/s3-verifyssl": "0",
            "serving.kserve.io/s3-usehttps": "1",
        },
        string_data={
            "AWS_ACCESS_KEY_ID": aws_access_key,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret


def fetch_openai_response(  # type: ignore
    url: str,
    model_name: str,
    chat_query=CHAT_QUERY,
    completion_query=COMPLETION_QUERY,
    tool_calling: dict[Any, Any] | None = None,
) -> tuple[Any, list[Any], list[Any]]:
    completion_responses = []
    chat_responses = []
    inference_client = OpenAIClient(host=url, model_name=model_name, streaming=True)
    if chat_query:
        for query in chat_query:
            chat_response = inference_client.request_http(
                endpoint=OpenAIEnpoints.CHAT_COMPLETIONS, query=query, extra_param=tool_calling
            )
            chat_responses.append(chat_response)
    if completion_query:
        for query in completion_query:
            completion_response = inference_client.request_http(
                endpoint=OpenAIEnpoints.COMPLETIONS, query=query, extra_param={"max_tokens": 100}
            )
            completion_responses.append(completion_response)

    model_info = OpenAIClient.get_request_http(host=url, endpoint=OpenAIEnpoints.MODELS_INFO)
    return model_info, chat_responses, completion_responses


def fetch_tgis_response(  # type: ignore
    url: str,
    model_name: str,
    completion_query=COMPLETION_QUERY,
) -> tuple[Any, list[Any], list[Any]]:
    completion_responses = []
    stream_completion_responses = []
    inference_client = TGISGRPCPlugin(host=url, model_name=model_name, streaming=True)
    model_info = inference_client.get_model_info()
    if completion_query:
        for query in completion_query:
            completion_response = inference_client.make_grpc_request(query=query)
            completion_responses.append(completion_response)
            stream_response = inference_client.make_grpc_request_stream(query=query)
            stream_completion_responses.append(stream_response)
    return model_info, completion_responses, stream_completion_responses


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def run_raw_inference(
    pod_name: str,
    isvc: InferenceService,
    port: int,
    endpoint: str,
    chat_query: list[list[dict[str, str]]] = CHAT_QUERY,
    completion_query: list[dict[str, str]] = COMPLETION_QUERY,
    tool_calling: dict[Any, Any] | None = None,
) -> tuple[Any, list[Any], list[Any]]:
    LOGGER.info(pod_name)
    with portforward.forward(
        pod_or_service=pod_name,
        namespace=isvc.namespace,
        from_port=port,
        to_port=port,
    ):
        if endpoint == "tgis":
            model_detail, grpc_chat_response, grpc_chat_stream_responses = fetch_tgis_response(
                url=f"localhost:{port}",
                model_name=isvc.instance.metadata.name,
                completion_query=completion_query,
            )
            return model_detail, grpc_chat_response, grpc_chat_stream_responses

        elif endpoint == "openai":
            model_info, completion_responses, stream_completion_responses = fetch_openai_response(
                url=f"http://localhost:{port}",
                model_name=isvc.instance.metadata.name,
                chat_query=chat_query,
                completion_query=completion_query,
                tool_calling=tool_calling,
            )
            return model_info, completion_responses, stream_completion_responses
        else:
            raise NotSupportedError(f"{endpoint} endpoint")


def validate_supported_quantization_schema(q_type: str) -> None:
    if q_type not in VLLM_SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization type: {q_type}")


def validate_inference_output(*args: tuple[str, ...] | list[Any], response_snapshot: Any) -> None:
    """Validate inference responses.

    Default mode is contract-based (fuzzy) validation: response schema, non-empty output
    text, sane finish/stop reasons, valid tool calls and no runtime error indicators.
    This keeps the suite portable across GPU architectures and vLLM image z-streams,
    where greedy decoding (temperature=0, fixed seed) is not bitwise reproducible.

    Set CHECK_SNAPSHOT=true to enforce exact golden snapshot comparison instead
    (only reproducible on the hardware and image the snapshots were generated with).
    """
    if os.getenv("CHECK_SNAPSHOT", "false").lower() == "true":
        for data in args:
            assert data == response_snapshot, f"output mismatch for {data}"
        return

    for arg_idx, data in enumerate(args):
        if data is None:
            raise InferenceValidationError(f"Validation argument #{arg_idx} is None")
        items = data if isinstance(data, list) else [data]
        if not items:
            LOGGER.warning(f"Validation argument #{arg_idx}: no responses to validate")
            continue
        _validate_response_list(items=items, context=f"response[{arg_idx}]")


def validate_raw_openai_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    chat_query: list[list[dict[str, str]]],
    completion_query: list[dict[str, str]],
    tool_calling: dict[Any, Any] | None = None,
) -> None:
    model_info, chat_responses, completion_responses = run_raw_inference(
        pod_name=pod_name,
        isvc=isvc,
        port=Ports.REST_PORT,
        endpoint=OPENAI_ENDPOINT_NAME,
        chat_query=chat_query,
        completion_query=completion_query,
        tool_calling=tool_calling,
    )
    validate_inference_output(
        model_info,
        chat_responses,
        completion_responses,
        response_snapshot=response_snapshot,
    )


def validate_raw_tgis_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    completion_query: list[dict[str, str]],
) -> None:
    model_info, chat_responses, completion_responses = run_raw_inference(
        pod_name=pod_name,
        isvc=isvc,
        port=Ports.GRPC_PORT,
        endpoint=TGIS_ENDPOINT_NAME,
        completion_query=completion_query,
    )
    validate_inference_output(
        model_info,
        chat_responses,
        completion_responses,
        response_snapshot=response_snapshot,
    )


def validate_serverless_openai_inference_request(
    url: str,
    model_name: str,
    response_snapshot: Any,
    chat_query: list[list[dict[str, str]]],
    completion_query: list[dict[str, str]],
    tool_calling: dict[Any, Any] | None = None,
) -> None:
    model_info, chat_responses, completion_responses = fetch_openai_response(
        url=url,
        model_name=model_name,
        chat_query=chat_query,
        completion_query=completion_query,
        tool_calling=tool_calling,
    )
    validate_inference_output(
        model_info,
        chat_responses,
        completion_responses,
        response_snapshot=response_snapshot,
    )


def skip_if_not_deployment_mode(isvc: InferenceService, deployment_type: str) -> None:
    if isvc.instance.metadata.annotations["serving.kserve.io/deploymentMode"] != deployment_type:
        pytest.skip(f"Test is being skipped because model is not being deployed in {deployment_type} mode")
