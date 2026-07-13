import os
import re
import subprocess
from collections.abc import Iterable
from typing import Any

import portforward
import structlog
from ocp_resources.inference_service import InferenceService
from tenacity import retry, stop_after_attempt, wait_exponential

from tests.model_serving.model_runtime.vllm.modelcar.constant import (
    AUDIO_FILE_LOCAL_PATH,
    AUDIO_FILE_URL,
    AUDIO_TRANSCRIPTION_KEYWORDS,
    COMPLETION_QUERY,
    EMBEDDING_QUERY,
    OPENAI_ENDPOINT_NAME,
)
from utilities.constants import Ports
from utilities.exceptions import NotSupportedError
from utilities.inference_utils import get_exposed_isvc_url
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient

LOGGER = structlog.get_logger(name=__name__)

ALPHABETIC_WORD = re.compile(r"[a-zA-Z]{2,}")
ERROR_INDICATORS = re.compile(
    r"traceback|cuda\s*error|out\s*of\s*memory|oom|"
    r"segmentation\s*fault|segfault|core\s*dumped|exception|"
    r"failed\s*to|error:|cuda_error|runtime\s*error|memory\s*error|"
    r"assertion\s*error|valueerror|typeerror|keyerror|attributeerror",
    re.IGNORECASE,
)
VALID_FINISH_REASONS = ["stop", "length", "eos_token", "stop_sequence"]


class InferenceValidationError(Exception):
    """Raised when fuzzy inference validation fails."""


def normalize_text_completion_response(response: Any) -> dict[str, Any]:
    """Wrap OpenAI client completion payloads into a standard choices envelope."""
    if isinstance(response, str):
        return {"choices": [{"text": response, "index": 0, "finish_reason": "stop"}]}
    if isinstance(response, dict):
        if "choices" in response:
            return response
        if "text" in response or "message" in response:
            return {"choices": [response]}
    raise InferenceValidationError(
        f"Text response must be a choice dict or choices envelope, got {type(response).__name__}"
    )


def normalize_audio_transcription_response(response: Any) -> dict[str, Any]:
    """Wrap OpenAI client audio payloads into a standard transcription dict."""
    if isinstance(response, str):
        return {"text": response}
    if isinstance(response, dict) and "text" in response:
        return response
    raise InferenceValidationError(
        f"Audio response must be a transcription string or dict with 'text', got {type(response).__name__}"
    )


def normalize_embedding_response(response: Any) -> dict[str, Any]:
    """Wrap OpenAI client embedding payloads into a standard data envelope."""
    if isinstance(response, dict):
        if "data" in response:
            return response
        if "embedding" in response:
            return {"data": [response]}
    raise InferenceValidationError(
        f"Embedding response must be an embedding item or data envelope, got {type(response).__name__}"
    )


def validate_inference_output(*args: tuple[str, ...] | list[Any], response_snapshot: Any) -> None:
    for data in args:
        assert data == response_snapshot, f"output mismatch for {data}"


def validate_text_inference_fuzzy(
    completion_responses: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    model_info: Any,
    require_keywords: bool = True,
    allow_empty_responses: bool = False,
    min_valid_responses: int = 1,
) -> None:
    if not completion_responses:
        raise InferenceValidationError("No responses provided for validation")

    if len(completion_responses) != len(queries):
        raise InferenceValidationError(
            f"Response count ({len(completion_responses)}) doesn't match query count ({len(queries)})"
        )
    if model_info is None:
        raise InferenceValidationError("Model info is None")
    if not isinstance(model_info, list):
        raise InferenceValidationError(f"Model info should be a list, got {type(model_info).__name__}")
    if len(model_info) == 0:
        raise InferenceValidationError("Model info list is empty")
    for idx, model in enumerate(model_info):
        if not isinstance(model, dict):
            raise InferenceValidationError(f"Model info item #{idx} should be a dict, got {type(model).__name__}")
        if "id" not in model or "object" not in model:
            raise InferenceValidationError(f"Model info item #{idx} missing 'id' or 'object' field")

    LOGGER.info("Model info validation passed")
    valid_response_count = 0
    for idx, (response, query) in enumerate(zip(completion_responses, queries)):
        query_text = query.get("text", "")
        expected_keywords = query.get("keywords", [])

        LOGGER.info(f"Validating response #{idx} for query: '{query_text[:50]}...'")
        if not isinstance(response, dict):
            raise InferenceValidationError(f"Response #{idx}: Expected dict, got {type(response).__name__}")
        if "choices" not in response:
            raise InferenceValidationError(f"Response #{idx}: Missing 'choices' field")
        choices = response["choices"]
        if not isinstance(choices, list):
            raise InferenceValidationError(f"Response #{idx}: 'choices' must be a list")
        if len(choices) == 0:
            raise InferenceValidationError(f"Response #{idx}: 'choices' list is empty")

        response_passed = False
        for choice_idx, choice in enumerate(choices):
            if not isinstance(choice, dict):
                raise InferenceValidationError(f"Response #{idx}, choice #{choice_idx}: Expected dict")

            finish_reason = (choice.get("finish_reason") or "").lower()
            if finish_reason and finish_reason not in VALID_FINISH_REASONS:
                LOGGER.warning(
                    f"Response #{idx}, choice #{choice_idx}: Unexpected finish_reason '{finish_reason}' "
                    f"(expected one of {VALID_FINISH_REASONS})"
                )

            has_text = "text" in choice
            has_message = "message" in choice and isinstance(choice["message"], dict)
            if not has_text and not has_message:
                raise InferenceValidationError(
                    f"Response #{idx}, choice #{choice_idx}: Missing 'text' or 'message' field"
                )
            if "text" in choice:
                text = choice["text"]
            elif "message" in choice and isinstance(choice["message"], dict):
                text = choice["message"].get("content", "")
            else:
                text = ""
            if not text or not text.strip():
                if allow_empty_responses:
                    LOGGER.warning(
                        f"Response #{idx}: Text is empty, skipping validation for this prompt "
                        f"(query: '{query_text[:50]}...')"
                    )
                    break
                raise InferenceValidationError(f"Response #{idx}: Text is empty or whitespace-only")

            if not expected_keywords and not ALPHABETIC_WORD.search(text):
                raise InferenceValidationError(
                    f"Response #{idx}: Text contains no alphabetic words (2+ chars). Text: '{text[:100]}...'"
                )
            text_lower = text.lower()
            if error_match := ERROR_INDICATORS.search(text_lower):
                raise InferenceValidationError(
                    f"Response #{idx}: Text contains error indicator '{error_match.group()}'. Text: '{text[:200]}...'"
                )
            if expected_keywords and require_keywords:
                found_keywords = [kw for kw in expected_keywords if kw.lower() in text_lower]
                if not found_keywords:
                    raise InferenceValidationError(
                        f"Response #{idx}: None of the expected keywords {expected_keywords} "
                        f"found in response to query: '{query_text[:100]}...'. Response text: '{text[:200]}...'"
                    )

                LOGGER.info(f"Response #{idx}: Found keywords {found_keywords} from expected {expected_keywords}")
            elif expected_keywords:
                LOGGER.info(
                    f"Response #{idx}: Skipping keyword check (smoke validation); "
                    f"expected keywords were {expected_keywords}"
                )
            else:
                LOGGER.warning(f"Response #{idx}: No keywords provided for validation")

            response_passed = True
            LOGGER.info(f"Response #{idx} passed all validation checks")
            break

        if not response_passed:
            continue

        valid_response_count += 1

    if allow_empty_responses:
        if valid_response_count < min_valid_responses:
            raise InferenceValidationError(
                f"Smoke validation: only {valid_response_count}/{len(completion_responses)} responses had "
                f"non-empty output; need at least {min_valid_responses}"
            )
        LOGGER.info(
            f"Smoke validation: {valid_response_count}/{len(completion_responses)} responses passed "
            f"(empty responses skipped)"
        )
    else:
        LOGGER.info(f"All {len(completion_responses)} responses passed fuzzy validation")


def validate_audio_inference_output(
    model_info: Any,
    completion_responses: Iterable[Any],
    expected_keywords: list[str] | None = None,
) -> None:
    if model_info is None:
        raise InferenceValidationError("Model info is None")
    if not isinstance(completion_responses, (list, tuple)):
        raise InferenceValidationError(
            f"Completion responses should be a list or tuple, got {type(completion_responses).__name__}"
        )
    if len(completion_responses) == 0:
        raise InferenceValidationError("Completion responses should not be empty")

    keywords = expected_keywords or AUDIO_TRANSCRIPTION_KEYWORDS
    for idx, response in enumerate(completion_responses):
        normalized = normalize_audio_transcription_response(response=response)
        text = normalized.get("text", "")
        if not text or not text.strip():
            raise InferenceValidationError(f"Audio response #{idx}: Transcription is empty")

        text_lower = text.lower()
        if ERROR_INDICATORS.search(text_lower):
            raise InferenceValidationError(
                f"Audio response #{idx}: Transcription contains error indicators. Text: '{text[:200]}...'"
            )

        if keywords:
            found_keywords = [kw for kw in keywords if kw.lower() in text_lower]
            if not found_keywords:
                raise InferenceValidationError(
                    f"Audio response #{idx}: None of the expected keywords {keywords} "
                    f"found in transcription. Text: '{text[:200]}...'"
                )
            LOGGER.info(f"Audio response #{idx}: Found keywords {found_keywords} from expected {keywords}")

    LOGGER.info(f"All {len(completion_responses)} audio transcription responses passed validation")


def validate_embedding_inference_output(model_info: Any, embedding_responses: Iterable[Any]) -> None:
    if model_info is None:
        raise InferenceValidationError("Model info is None")
    if not isinstance(embedding_responses, (list, tuple)):
        raise InferenceValidationError(
            f"Embedding responses should be a list or tuple, got {type(embedding_responses).__name__}"
        )
    if len(embedding_responses) == 0:
        raise InferenceValidationError("Embedding responses should not be empty")

    for idx, response in enumerate(embedding_responses):
        normalized = normalize_embedding_response(response=response)
        data = normalized.get("data", [])
        if not isinstance(data, list) or len(data) == 0:
            raise InferenceValidationError(f"Embedding response #{idx}: 'data' list is empty")

        for embed_idx, embedding_obj in enumerate(data):
            if not isinstance(embedding_obj, dict):
                raise InferenceValidationError(
                    f"Embedding response #{idx}, item #{embed_idx}: Expected dict, got {type(embedding_obj).__name__}"
                )
            embedding = embedding_obj.get("embedding", [])
            if not isinstance(embedding, list) or len(embedding) == 0:
                raise InferenceValidationError(
                    f"Embedding response #{idx}, item #{embed_idx}: Embedding vector is empty"
                )
            if not all(isinstance(value, (int, float)) for value in embedding):
                raise InferenceValidationError(
                    f"Embedding response #{idx}, item #{embed_idx}: Embedding vector must contain only numbers"
                )

    LOGGER.info(f"All {len(embedding_responses)} embedding responses passed validation")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def run_raw_inference(
    isvc: InferenceService,
    endpoint: str,
    completion_query: list[dict[str, str]] = COMPLETION_QUERY,
    url: str | None = None,
    pod_name: str | None = None,
    port: int | None = Ports.REST_PORT,
) -> tuple[Any, list[Any]]:
    if url is not None:
        LOGGER.info("Using external route for inference: %s", url)
        if endpoint == "openai":
            return fetch_openai_response(
                url=url,
                model_name=isvc.instance.metadata.name,
                completion_query=completion_query,
            )
        raise NotSupportedError(f"{endpoint} endpoint")

    LOGGER.info("Using port forwarding for inference on pod: %s", pod_name)
    if pod_name is None or port is None:
        raise ValueError("pod_name and port are required when url is not provided")

    with portforward.forward(
        pod_or_service=pod_name,
        namespace=isvc.namespace,
        from_port=port,
        to_port=port,
    ):
        if endpoint == "openai":
            return fetch_openai_response(
                url=f"http://localhost:{port}",
                model_name=isvc.instance.metadata.name,
                completion_query=completion_query,
            )
        raise NotSupportedError(f"{endpoint} endpoint")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def run_embedding_inference(
    endpoint: str,
    model_name: str,
    url: str | None = None,
    pod_name: str | None = None,
    isvc: InferenceService | None = None,
    port: int | None = Ports.REST_PORT,
    embedding_query: list[dict[str, str]] = EMBEDDING_QUERY,
) -> tuple[Any, list[Any]]:
    LOGGER.info("Running embedding inference for model: %s on endpoint: %s", model_name, endpoint)
    if url is not None:
        LOGGER.info("Using provided URL for inference: %s", url)
        inference_client = OpenAIClient(host=url, model_name=model_name, streaming=True)
        embedding_responses = []
        for query in embedding_query:
            embedding_response = inference_client.request_http(
                endpoint=OpenAIEnpoints.EMBEDDINGS,
                query=query,
            )
            embedding_responses.append(embedding_response)
        model_info = OpenAIClient.get_request_http(host=url, endpoint=OpenAIEnpoints.MODELS_INFO)
        return model_info, embedding_responses
    else:
        LOGGER.info("Using port forwarding for inference on pod: %s", pod_name)
        if pod_name is None or isvc is None or port is None:
            raise ValueError("pod_name, isvc, and port are required when url is not provided")

        with portforward.forward(
            pod_or_service=pod_name,
            namespace=isvc.namespace,
            from_port=port,
            to_port=port,
        ):
            if endpoint == "openai":
                embedding_responses = []
                inference_client = OpenAIClient(host=f"http://localhost:{port}", model_name=model_name, streaming=True)
                for query in embedding_query:
                    embedding_response = inference_client.request_http(endpoint=OpenAIEnpoints.EMBEDDINGS, query=query)
                    embedding_responses.append(embedding_response)
                model_info = OpenAIClient.get_request_http(
                    host=f"http://localhost:{port}", endpoint=OpenAIEnpoints.MODELS_INFO
                )
                return model_info, embedding_responses
            else:
                raise NotSupportedError(f"{endpoint} endpoint for embedding inference")


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def run_audio_inference(
    endpoint: str,
    model_name: str,
    audio_file_path: str = AUDIO_FILE_LOCAL_PATH,
    audio_file_url: str = AUDIO_FILE_URL,
    url: str | None = None,
    pod_name: str | None = None,
    isvc: InferenceService | None = None,
    port: int | None = Ports.REST_PORT,
) -> tuple[Any, list[Any]]:
    LOGGER.info("Running audio inference for model: %s on endpoint: %s", model_name, endpoint)
    download_audio_file(audio_file_url=audio_file_url, destination_path=audio_file_path)

    if url is not None:
        LOGGER.info("Using provided URL for inference: %s", url)
        inference_client = OpenAIClient(host=url, model_name=model_name, streaming=True)
        completion_responses = []
        completion_response = inference_client.request_audio(
            endpoint=OpenAIEnpoints.AUDIO_TRANSCRIPTION,
            audio_file_path=audio_file_path,
            model_name=model_name,
        )
        completion_responses.append(completion_response)
        model_info = OpenAIClient.get_request_http(host=url, endpoint=OpenAIEnpoints.MODELS_INFO)
        return model_info, completion_responses
    else:
        LOGGER.info("Using port forwarding for inference on pod: %s", pod_name)
        if pod_name is None or isvc is None or port is None:
            raise ValueError("pod_name, isvc, and port are required when url is not provided")

        with portforward.forward(
            pod_or_service=pod_name,
            namespace=isvc.namespace,
            from_port=port,
            to_port=port,
        ):
            if endpoint == "openai":
                completion_responses = []
                inference_client = OpenAIClient(host=f"http://localhost:{port}", model_name=model_name, streaming=True)
                completion_response = inference_client.request_audio(
                    endpoint=OpenAIEnpoints.AUDIO_TRANSCRIPTION, audio_file_path=audio_file_path, model_name=model_name
                )
                completion_responses.append(completion_response)
                model_info = OpenAIClient.get_request_http(
                    host=f"http://localhost:{port}", endpoint=OpenAIEnpoints.MODELS_INFO
                )
                return model_info, completion_responses
            else:
                raise NotSupportedError(f"{endpoint} endpoint for audio inference")


def validate_raw_openai_inference_request(
    isvc: InferenceService,
    response_snapshot: Any | None = None,
    completion_query: list[dict[str, Any]] | None = None,
    model_output_type: str = "text",
    model_name: str | None = None,
    pod_name: str | None = None,
    port: int = Ports.REST_PORT,
) -> None:
    if model_output_type == "audio":
        LOGGER.info("Running audio inference test")
        model_info, completion_responses = run_audio_inference(
            url=get_exposed_isvc_url(isvc=isvc),
            endpoint=OPENAI_ENDPOINT_NAME,
            model_name=model_name,
        )
        validate_audio_inference_output(
            model_info=model_info,
            completion_responses=completion_responses,
            expected_keywords=AUDIO_TRANSCRIPTION_KEYWORDS,
        )
        if os.path.exists(AUDIO_FILE_LOCAL_PATH):
            os.remove(AUDIO_FILE_LOCAL_PATH)
        return
    elif model_output_type == "text":
        LOGGER.info("Running text inference test")
        effective_query = completion_query or COMPLETION_QUERY
        model_info, completion_responses = run_raw_inference(
            isvc=isvc,
            endpoint=OPENAI_ENDPOINT_NAME,
            completion_query=effective_query,
            url=get_exposed_isvc_url(isvc=isvc),
        )
        wrapped_completion_responses = [
            normalize_text_completion_response(response) for response in completion_responses
        ]
        validate_text_inference_fuzzy(
            completion_responses=wrapped_completion_responses,
            queries=effective_query,
            model_info=model_info,
            require_keywords=True,
            allow_empty_responses=True,
            min_valid_responses=1,
        )
        if os.getenv("CHECK_SNAPSHOT", "false").lower() == "true":
            validate_inference_output(
                completion_responses,
                response_snapshot=response_snapshot,
            )
    elif model_output_type == "embedding":
        model_info, embedding_responses = run_embedding_inference(
            url=get_exposed_isvc_url(isvc=isvc),
            endpoint=OPENAI_ENDPOINT_NAME,
            embedding_query=EMBEDDING_QUERY,
            model_name=model_name,
        )
        validate_embedding_inference_output(model_info=model_info, embedding_responses=embedding_responses)

    else:
        raise NotSupportedError(f"Model output type {model_output_type} is not supported for raw inference request.")


def download_audio_file(audio_file_url: str = AUDIO_FILE_URL, destination_path: str = AUDIO_FILE_LOCAL_PATH) -> None:
    """
    Download an audio file and save to destination_path if it's missing or empty.

    :param audio_file_url: The URL of the audio file to download.
    :param destination_path: The local path where the audio file should be saved.
    """
    dir_ = os.path.dirname(destination_path)
    os.makedirs(dir_, exist_ok=True)

    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
        LOGGER.info("Audio file already exists at %s, skipping download.", destination_path)
        return
    cmd = ["curl", "-fSL", "-o", destination_path, audio_file_url]
    try:
        subprocess.run(args=cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)  # noqa: UP022
        LOGGER.info("Audio file downloaded successfully to %s", destination_path)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else str(e)
        LOGGER.error("Failed to download audio file: %s", stderr)
        raise RuntimeError(f"Failed to download audio file: {stderr}") from e


def fetch_openai_response(
    url: str,
    model_name: str,
    completion_query: list[dict[str, str]] | None = None,
) -> tuple[Any, list[Any]]:
    model_info = OpenAIClient.get_request_http(host=url, endpoint=OpenAIEnpoints.MODELS_INFO)
    model_name = model_info[0]["id"] if model_info else model_name
    if completion_query is None:
        completion_query = COMPLETION_QUERY
    completion_responses = []
    inference_client = OpenAIClient(host=url, model_name=model_name, streaming=True)
    if completion_query:
        for query in completion_query:
            completion_response = inference_client.request_http(
                endpoint=OpenAIEnpoints.COMPLETIONS, query=query, extra_param={"max_tokens": 100}
            )
            completion_responses.append(completion_response)

    return model_info, completion_responses
