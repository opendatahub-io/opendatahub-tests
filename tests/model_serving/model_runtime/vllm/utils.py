import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import portforward
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.secret import Secret
from tenacity import retry, stop_after_attempt, wait_exponential

from tests.model_serving.model_runtime.utils import validate_inference_output, validate_text_inference_fuzzy
from tests.model_serving.model_runtime.vllm.constant import (
    CHAT_QUERY,
    COMPLETION_QUERY,
    VLLM_SUPPORTED_QUANTIZATION,
)
from utilities.constants import Ports
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient

LOGGER = structlog.get_logger(name=__name__)


def dedupe_vllm_cli_args(arguments: list[str]) -> list[str]:
    """Keep the first occurrence of each CLI flag (e.g. --model) to avoid vLLM duplicate-key warnings."""
    seen: set[str] = set()
    deduped: list[str] = []
    for arg in arguments:
        flag = arg.split("=", maxsplit=1)[0]
        if flag in seen:
            continue
        seen.add(flag)
        deduped.append(arg)
    return deduped


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
            messages = [msg for msg in query if "role" in msg]
            chat_extra_param: dict[str, Any] = {"max_tokens": 256}
            if tool_calling:
                chat_extra_param.update(tool_calling)
            chat_response = inference_client.request_http(
                endpoint=OpenAIEnpoints.CHAT_COMPLETIONS,
                query=messages,
                extra_param=chat_extra_param,
            )
            chat_responses.append(chat_response)
    if completion_query:
        for query in completion_query:
            completion_response = inference_client.request_http(
                endpoint=OpenAIEnpoints.COMPLETIONS, query=query, extra_param={"max_tokens": 256}
            )
            completion_responses.append(completion_response)

    model_info = OpenAIClient.get_request_http(host=url, endpoint=OpenAIEnpoints.MODELS_INFO)
    return model_info, chat_responses, completion_responses


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def run_raw_inference(
    pod_name: str,
    isvc: InferenceService,
    port: int,
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
        return fetch_openai_response(
            url=f"http://localhost:{port}",
            model_name=isvc.instance.metadata.name,
            chat_query=chat_query,
            completion_query=completion_query,
            tool_calling=tool_calling,
        )


def validate_supported_quantization_schema(q_type: str) -> None:
    if q_type not in VLLM_SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization type: {q_type}")


def validate_raw_openai_inference_request(
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    chat_query: list[list[dict[str, Any]]],
    completion_query: list[dict[str, Any]],
    tool_calling: dict[Any, Any] | None = None,
) -> None:
    model_info, chat_responses, completion_responses = run_raw_inference(
        pod_name=pod_name,
        isvc=isvc,
        port=Ports.REST_PORT,
        chat_query=chat_query,
        completion_query=completion_query,
        tool_calling=tool_calling,
    )

    if chat_responses:
        chat_validation_queries = []
        for idx, query_list in enumerate(chat_query):
            keywords_dict = next((msg for msg in query_list if "keywords" in msg), None)
            if keywords_dict is None:
                raise ValueError(f"chat_query[{idx}] is missing keywords metadata")
            user_text = " ".join(
                msg.get("content", "") for msg in query_list if isinstance(msg, dict) and "content" in msg
            )
            chat_validation_queries.append({"text": user_text, "keywords": keywords_dict["keywords"]})

        wrapped_chat_responses = [resp if "choices" in resp else {"choices": [resp]} for resp in chat_responses]
        validate_text_inference_fuzzy(
            completion_responses=wrapped_chat_responses,
            queries=chat_validation_queries,
            model_info=model_info,
            require_keywords=False,
            allow_empty_responses=True,
            min_valid_responses=1,
        )

    if completion_responses:
        wrapped_completion_responses = [
            resp if "choices" in resp else {"choices": [resp]} for resp in completion_responses
        ]
        validate_text_inference_fuzzy(
            completion_responses=wrapped_completion_responses,
            queries=completion_query,
            model_info=model_info,
            require_keywords=False,
            allow_empty_responses=True,
            min_valid_responses=1,
        )

    if os.getenv("CHECK_SNAPSHOT", "false").lower() == "true":
        validate_inference_output(
            model_info,
            chat_responses,
            completion_responses,
            response_snapshot=response_snapshot,
        )


def skip_if_not_deployment_mode(isvc: InferenceService, deployment_types: str | tuple[str, ...]) -> None:
    actual = isvc.instance.metadata.annotations["serving.kserve.io/deploymentMode"]
    expected = (deployment_types,) if isinstance(deployment_types, str) else deployment_types
    if actual not in expected:
        pytest.skip(f"Test is being skipped because deployment mode {actual!r} is not in {expected}")
