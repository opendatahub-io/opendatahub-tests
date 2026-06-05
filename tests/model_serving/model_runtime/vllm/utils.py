from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
import csv
import os
import re
from datetime import datetime
import portforward
import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from tenacity import retry, stop_after_attempt, wait_exponential

from tests.model_serving.model_runtime.vllm.constant import (
    CHAT_QUERY,
    COMPLETION_QUERY,
    OPENAI_ENDPOINT_NAME,
    TGIS_ENDPOINT_NAME,
    VLLM_SUPPORTED_QUANTIZATION,
)
from utilities.constants import Ports
from utilities.exceptions import NotSupportedError
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient
from utilities.plugins.tgis_grpc_plugin import TGISGRPCPlugin

LOGGER = structlog.get_logger(name=__name__)


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
        for query in COMPLETION_QUERY:
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
        for query in COMPLETION_QUERY:
            completion_response = inference_client.make_grpc_request(query=query)
            completion_responses.append(completion_response)
            stream_response = inference_client.make_grpc_request_stream(query=query)
            completion_responses.append(completion_response)
            stream_completion_responses.append(stream_response)
    return model_info, completion_responses, stream_completion_responses


@retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=6))
def run_raw_inference(
    pod_name: str,
    isvc: InferenceService,
    port: int,
    endpoint: str,
    chat_query: list[list[dict[str, str]]] | None = None,
    #chat_query: list[list[dict[str, str]]] = CHAT_QUERY,
    completion_query: list[dict[str, str]] = COMPLETION_QUERY,
    tool_calling: dict[Any, Any] | None = None,
) -> tuple[Any, list[Any], list[Any]]:
    chat_query = chat_query or CHAT_QUERY
    LOGGER.info(pod_name)
    with portforward.forward(
        pod_or_service=pod_name,
        namespace=isvc.namespace,
        from_port=0,
        to_port=port,
    ) as pf:
        local_port = pf.from_port
        if endpoint == "tgis":
            model_detail, grpc_chat_response, grpc_chat_stream_responses = fetch_tgis_response(
                url=f"localhost:{local_port}",
                model_name=isvc.instance.metadata.name,
                completion_query=completion_query,
            )
            return model_detail, grpc_chat_response, grpc_chat_stream_responses

        elif endpoint == "openai":
            model_info, completion_responses, stream_completion_responses = fetch_openai_response(
                url=f"http://localhost:{local_port}",
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
    for data in args:
        assert data == response_snapshot, f"output mismatch for {data}"


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


def skip_if_not_deployment_mode(isvc: InferenceService, deployment_type: str) -> None:
    if isvc.instance.metadata.annotations["serving.kserve.io/deploymentMode"] != deployment_type:
        pytest.skip(f"Test is being skipped because model is not being deployed in {deployment_type} mode")


# Performance helper methods

def get_vllm_version(namespace: str, pod_name: str) -> str:
    pod = Pod(name=pod_name, namespace=namespace)
    result = pod.execute(command=["python", "-c", "import vllm; print(vllm.__version__)"])
    return result.strip()

def get_vllm_throughput_logs(namespace: str, pod_name: str, start_time: str | None = None) -> str:
    pod = Pod(name=pod_name, namespace=namespace)
    logs = pod.log(container="kserve-container", since_time=start_time) if start_time else pod.log(
        container="kserve-container"
    )
    return "\n".join(line for line in logs.splitlines() if "Avg prompt throughput" in line)

def parse_vllm_logs(logs: str, used_entries: set[str]) -> list[dict[str, float]]:
    parsed = []

    for line in logs.splitlines():
        if line in used_entries:
            continue

        match = re.search(
            r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s",
            line,
        )

        if match:
            parsed.append(
                {
                    "prompt_tokens_per_sec": float(match.group(1)),
                    "generation_tokens_per_sec": float(match.group(2)),
                }
            )
            used_entries.add(line)

    return parsed

def save_performance_report(
    model_name: str,
    version: str,
    logs: str,
    request_type: str,
    input_prompt: str,
    used_entries: set[str],
) -> None:
    parsed_logs = parse_vllm_logs(logs=logs, used_entries=used_entries)

    last = parsed_logs[-1] if parsed_logs else {}
    max_prompt = max((x["prompt_tokens_per_sec"] for x in parsed_logs), default=0)
    max_generation = max((x["generation_tokens_per_sec"] for x in parsed_logs), default=0)

    worker = os.getenv("PYTEST_XDIST_WORKER", "gw0")
    report_file = f"performance_report_{worker}.csv"
    file_exists = os.path.isfile(report_file)

    with open(report_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "model",
                    "vllm_version",
                    "request_type",
                    "input_prompt",
                    "last_prompt_tokens_per_sec",
                    "last_generation_tokens_per_sec",
                    "max_prompt_tokens_per_sec",
                    "max_generation_tokens_per_sec",
                ]
            )
        writer.writerow(
            [
                model_name,
                version,
                request_type,
                input_prompt,
                last.get("prompt_tokens_per_sec", 0),
                last.get("generation_tokens_per_sec", 0),
                max_prompt,
                max_generation,
            ]
        )

