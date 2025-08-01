import pytest
from simple_logger.logger import get_logger
from typing import Any, Generator
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.inference_service import InferenceService
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_server.utils import verify_keda_scaledobject, verify_final_pod_count
from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.vllm.basic_model_deployment.test_granite_7b_starter import (
    SERVING_ARGUMENT,
    MODEL_PATH,
)
from utilities.monitoring import validate_metrics_field

LOGGER = get_logger(name=__name__)


BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

INITIAL_POD_COUNT = 1
FINAL_POD_COUNT = 5

VLLM_MODEL_NAME = "granite-vllm-keda"
VLLM_METRICS_QUERY_REQUESTS = f'vllm:num_requests_running{{namespace="{VLLM_MODEL_NAME}",pod=~"{VLLM_MODEL_NAME}.*"}}'
VLLM_METRICS_THRESHOLD_REQUESTS = 4

pytestmark = [pytest.mark.keda, pytest.mark.usefixtures("skip_if_no_supported_gpu_type", "valid_aws_config")]


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_cuda_serving_runtime, stressed_keda_vllm_inference_service",
    [
        pytest.param(
            {"name": VLLM_MODEL_NAME},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": VLLM_MODEL_NAME,
                "initial_pod_count": INITIAL_POD_COUNT,
                "final_pod_count": FINAL_POD_COUNT,
                "metrics_query": VLLM_METRICS_QUERY_REQUESTS,
                "metrics_threshold": VLLM_METRICS_THRESHOLD_REQUESTS,
            },
            id=f"{VLLM_MODEL_NAME}-single-gpu",
        ),
    ],
    indirect=True,
)
class TestVllmKedaScaling:
    """
    Test Keda functionality for a gpu based inference service.
    This class verifies pod scaling, metrics availability, and the creation of a keda scaled object.
    """

    def test_vllm_keda_scaling_verify_scaledobject(
        self,
        unprivileged_model_namespace: Namespace,
        vllm_cuda_serving_runtime,
        unprivileged_client: DynamicClient,
        stressed_keda_vllm_inference_service: Generator[InferenceService, Any, Any],
    ):
        verify_keda_scaledobject(
            client=unprivileged_client,
            isvc=stressed_keda_vllm_inference_service,
            expected_trigger_type="prometheus",
            expected_query=VLLM_METRICS_QUERY_REQUESTS,
            expected_threshold=VLLM_METRICS_THRESHOLD_REQUESTS,
        )

    def test_vllm_keda_scaling_verify_metrics(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        vllm_cuda_serving_runtime,
        stressed_keda_vllm_inference_service: Generator[InferenceService, Any, Any],
        prometheus,
    ):
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=VLLM_METRICS_QUERY_REQUESTS,
            expected_value=str(VLLM_METRICS_THRESHOLD_REQUESTS),
            greater_than=True,
        )

    def test_vllm_keda_scaling_verify_final_pod_count(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        vllm_cuda_serving_runtime,
        stressed_keda_vllm_inference_service: Generator[InferenceService, Any, Any],
    ):
        verify_final_pod_count(
            unprivileged_client=unprivileged_client,
            isvc=stressed_keda_vllm_inference_service,
            final_pod_count=FINAL_POD_COUNT,
        )
