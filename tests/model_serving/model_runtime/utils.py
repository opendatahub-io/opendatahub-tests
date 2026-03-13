import os
import subprocess
from collections.abc import Iterable
from typing import Any

import portforward
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from tenacity import retry, stop_after_attempt, wait_exponential

from tests.model_serving.model_runtime.model_validation.constant import (
    AUDIO_FILE_LOCAL_PATH,
    AUDIO_FILE_URL,
    COMPLETION_QUERY,
    EMBEDDING_QUERY,
    OPENAI_ENDPOINT_NAME,
    SPYRE_INFERENCE_SERVICE_PORT,
)
from utilities.constants import Ports
from utilities.exceptions import NotSupportedError
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient
from utilities.plugins.tgis_grpc_plugin import TGISGRPCPlugin

LOGGER = get_logger(name=__name__)


def validate_inference_output(*args: tuple[str, ...] | list[Any], response_snapshot: Any) -> None:
    for data in args:
        assert data == response_snapshot, f"output mismatch for {data}"


def validate_audio_inference_output(model_info: Any, completion_responses: Iterable[Any]) -> None:
    assert model_info is not None, "Model info should not be None"
    assert isinstance(model_info, (list, tuple)), "Model info should be a list or tuple"
    assert isinstance(completion_responses, (list, tuple)), "Completion responses should be a list or tuple"
    assert len(completion_responses) > 0, "Completion responses should not be empty"


def validate_embedding_inference_output(model_info: Any, embedding_responses: Iterable[Any]) -> None:
    assert model_info is not None, "Model info should not be None"
    assert isinstance(model_info, (list, tuple)), "Model info should be a list or tuple"
    assert isinstance(embedding_responses, (list, tuple)), "Embedding responses should be a list or tuple"
    assert len(embedding_responses) > 0, "Embedding responses should not be empty"


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
    LOGGER.info("audio_inference:start endpoint=%s pod=%s", endpoint, pod_name)
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
    pod_name: str,
    isvc: InferenceService,
    response_snapshot: Any,
    completion_query: list[dict[str, str]],
    model_output_type: str,
    model_name: str,
    port: int = Ports.REST_PORT,
) -> None:
    if model_output_type == "audio":
        LOGGER.info("Running audio inference test")
        model_info, completion_responses = run_audio_inference(
            pod_name=pod_name,
            isvc=isvc,
            port=port,
            endpoint=OPENAI_ENDPOINT_NAME,
            model_name=model_name,
        )
        validate_audio_inference_output(model_info=model_info, completion_responses=completion_responses)
        if os.path.exists(AUDIO_FILE_LOCAL_PATH):
            os.remove(AUDIO_FILE_LOCAL_PATH)
        return
    elif model_output_type == "text":
        LOGGER.info("Running text inference test")
        scheduler_name = getattr(isvc.instance.spec.predictor, "schedulerName", "") or ""
        if scheduler_name.lower() == "spyre-scheduler":
            port = SPYRE_INFERENCE_SERVICE_PORT
        model_info, completion_responses = run_raw_inference(
            pod_name=pod_name,
            isvc=isvc,
            port=port,
            endpoint=OPENAI_ENDPOINT_NAME,
            completion_query=completion_query,
        )
        validate_inference_output(
            completion_responses,
            response_snapshot=response_snapshot,
        )
    elif model_output_type == "embedding":
        model_info, embedding_responses = run_embedding_inference(
            pod_name=pod_name,
            isvc=isvc,
            port=port,
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
