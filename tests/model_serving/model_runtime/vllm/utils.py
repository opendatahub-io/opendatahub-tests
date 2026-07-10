import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.node import Node
from ocp_resources.secret import Secret
from tenacity import retry, stop_after_attempt, wait_exponential

from tests.model_serving.model_runtime.utils import validate_inference_output, validate_text_inference_fuzzy
from tests.model_serving.model_runtime.vllm.constant import (
    CHAT_QUERY,
    COMPLETION_QUERY,
    VLLM_SUPPORTED_QUANTIZATION,
)
from utilities.inference_utils import get_exposed_isvc_url
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient

LOGGER = structlog.get_logger(name=__name__)

ZONE_LABEL = "topology.kubernetes.io/zone"


def get_gpu_node_zone_selector(
    client: DynamicClient,
    gpu_resource: str,
    min_gpus: int = 1,
) -> dict[str, str] | None:
    """Return a nodeSelector for the zone with the most allocatable GPU capacity.

    Used to pin PVC download pods to the same availability zone as GPU worker nodes,
    so ReadWriteOnce volumes bind where the vLLM predictor can schedule.

    Args:
        client: Kubernetes client for cluster introspection.
        gpu_resource: Kubernetes extended resource name (e.g. nvidia.com/gpu).
        min_gpus: Minimum allocatable GPUs required on a node to count it.

    Returns:
        A nodeSelector dict keyed by topology.kubernetes.io/zone, or None when no
        worker node exposes the GPU resource with a zone label (on-prem clusters).

    Note:
        Only effective when the storage class uses WaitForFirstConsumer binding.
        Immediate-binding storage classes may provision volumes before the download
        pod runs, requiring cluster-level topology alignment instead.
    """
    zone_stats: dict[str, dict[str, int]] = {}
    for node in Node.get(client=client, label_selector="node-role.kubernetes.io/worker"):
        allocatable = node.instance.status.allocatable or {}
        try:
            gpu_count = int(allocatable.get(gpu_resource, 0))
        except TypeError, ValueError:
            continue
        if gpu_count < min_gpus:
            continue

        zone = node.instance.metadata.labels.get(ZONE_LABEL)
        if not zone:
            LOGGER.info(
                "GPU node %s has no %s label; skipping zone selection",
                node.name,
                ZONE_LABEL,
            )
            continue

        if zone not in zone_stats:
            zone_stats[zone] = {"total_gpus": 0, "node_count": 0}
        zone_stats[zone]["total_gpus"] += gpu_count
        zone_stats[zone]["node_count"] += 1

    if not zone_stats:
        LOGGER.info(
            "No GPU worker nodes with zone labels found for %s; download pod will not be zone-pinned",
            gpu_resource,
        )
        return None

    selected_zone = max(
        zone_stats,
        key=lambda zone: (zone_stats[zone]["total_gpus"], zone_stats[zone]["node_count"]),
    )
    LOGGER.info(
        "Selected zone %s for PVC download (%s GPU(s) on %s node(s))",
        selected_zone,
        zone_stats[selected_zone]["total_gpus"],
        zone_stats[selected_zone]["node_count"],
    )
    return {ZONE_LABEL: selected_zone}


def add_image_pull_secrets_if_configured(
    isvc_kwargs: dict[str, Any],
    kserve_registry_pull_secret: Secret | None,
) -> None:
    """Add imagePullSecrets to ISVC kwargs when a registry pull secret is configured."""
    if kserve_registry_pull_secret is not None:
        isvc_kwargs["image_pull_secrets"] = [kserve_registry_pull_secret.name]


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
    isvc: InferenceService,
    chat_query: list[list[dict[str, str]]] = CHAT_QUERY,
    completion_query: list[dict[str, str]] = COMPLETION_QUERY,
    tool_calling: dict[Any, Any] | None = None,
) -> tuple[Any, list[Any], list[Any]]:
    url = get_exposed_isvc_url(isvc=isvc)
    LOGGER.info("Using external route for inference: %s", url)
    return fetch_openai_response(
        url=url,
        model_name=isvc.instance.metadata.name,
        chat_query=chat_query,
        completion_query=completion_query,
        tool_calling=tool_calling,
    )


def validate_supported_quantization_schema(q_type: str) -> None:
    if q_type not in VLLM_SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization type: {q_type}")


def validate_raw_openai_inference_request(
    isvc: InferenceService,
    response_snapshot: Any,
    chat_query: list[list[dict[str, Any]]],
    completion_query: list[dict[str, Any]],
    tool_calling: dict[Any, Any] | None = None,
) -> None:
    model_info, chat_responses, completion_responses = run_raw_inference(
        isvc=isvc,
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
