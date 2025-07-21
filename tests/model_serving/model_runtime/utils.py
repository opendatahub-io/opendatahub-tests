from typing import Any

import portforward
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

from tests.model_serving.model_runtime.model_validation.constant import COMPLETION_QUERY
from tests.model_serving.model_runtime.model_validation.constant import OPENAI_ENDPOINT_NAME
from utilities.constants import Ports
from utilities.exceptions import NotSupportedError
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient
from utilities.plugins.tgis_grpc_plugin import TGISGRPCPlugin

LOGGER = get_logger(name=__name__)


def validate_inference_output(*args: tuple[str, ...] | list[Any], response_snapshot: Any) -> None:
    for data in args:
        assert data == response_snapshot, f"output mismatch for {data}"


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
    completion_query: list[dict[str, str]] = COMPLETION_QUERY,
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
            model_info, completion_responses = fetch_openai_response(
                url=f"http://localhost:{port}",
                model_name=isvc.instance.metadata.name,
                completion_query=completion_query,
            )
            return model_info, completion_responses  # type: ignore
        else:
            raise NotSupportedError(f"{endpoint} endpoint")


def validate_raw_openai_inference_request(
    pod_name: str, isvc: InferenceService, response_snapshot: Any, completion_query: list[dict[str, str]]
) -> None:
    model_info, completion_responses = run_raw_inference(
        pod_name=pod_name,
        isvc=isvc,
        port=Ports.REST_PORT,
        endpoint=OPENAI_ENDPOINT_NAME,
        completion_query=completion_query,
    )
    validate_inference_output(
        model_info,
        completion_responses,
        response_snapshot=response_snapshot,
    )


def fetch_openai_response(
    url: str,
    model_name: str,
    completion_query: list[dict[str, str]] | None = None,
) -> tuple[Any, list[Any]]:
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

    model_info = OpenAIClient.get_request_http(host=url, endpoint=OpenAIEnpoints.MODELS_INFO)
    return model_info, completion_responses


def validate_serverless_openai_inference_request(
    url: str, model_name: str, response_snapshot: Any, completion_query: list[dict[str, str]]
) -> None:
    print(f"this is the url: {url} for model: {model_name} serverless inference")
    model_info, completion_responses = fetch_openai_response(
        url=url, model_name=model_name, completion_query=completion_query
    )
    validate_inference_output(
        model_info,
        completion_responses,
        response_snapshot=response_snapshot,
    )
