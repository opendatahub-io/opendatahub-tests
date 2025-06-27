from contextlib import contextmanager
from typing import Generator, Any
from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger

from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient
from tests.model_serving.model_runtime.vllm.constant import VLLM_SUPPORTED_QUANTIZATION
from tests.model_serving.model_runtime.model_validation.constant import PULL_SECRET_ACCESS_TYPE
import pytest
from tests.model_serving.model_runtime.model_validation.constant import CHAT_QUERY, COMPLETION_QUERY

LOGGER = get_logger(name=__name__)


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


@contextmanager
def kserve_registry_pull_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    registry_pull_secret: str,
    registry_host: str,
) -> Generator[Secret, Any, Any]:
    dockerconfigjson = f'{{"auths": {{"{registry_host}": {{"auth": "{registry_pull_secret}"}}}}}}'
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        string_data={
            ".dockerconfigjson": dockerconfigjson,
            "ACCESS_TYPE": PULL_SECRET_ACCESS_TYPE,
            "OCI_HOST": registry_host,
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
        for query in COMPLETION_QUERY:
            completion_response = inference_client.request_http(
                endpoint=OpenAIEnpoints.COMPLETIONS, query=query, extra_param={"max_tokens": 100}
            )
            completion_responses.append(completion_response)

    model_info = OpenAIClient.get_request_http(host=url, endpoint=OpenAIEnpoints.MODELS_INFO)
    return model_info, chat_responses, completion_responses


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


def validate_supported_quantization_schema(q_type: str) -> None:
    if q_type not in VLLM_SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization type: {q_type}")


def validate_inference_output(*args: tuple[str, ...] | list[Any], response_snapshot: Any) -> None:
    for data in args:
        assert data == response_snapshot, f"output mismatch for {data}"


def skip_if_deployment_mode(isvc: InferenceService, deployment_type: str, deployment_message: str) -> None:
    if isvc.instance.metadata.annotations["serving.kserve.io/deploymentMode"] == deployment_type:
        pytest.skip(deployment_message)
