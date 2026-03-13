from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_server.utils import verify_final_pod_count, verify_keda_scaledobject
from utilities.constants import KServeDeploymentType
from utilities.monitoring import validate_metrics_field

LOGGER = get_logger(name=__name__)

SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
]

MODEL_PATH: str = "granite-7b-starter"

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
                "model-name": VLLM_MODEL_NAME,
                "initial_pod_count": INITIAL_POD_COUNT,
                "final_pod_count": FINAL_POD_COUNT,
                "metrics_query": VLLM_METRICS_QUERY_REQUESTS,
                "metrics_threshold": VLLM_METRICS_THRESHOLD_REQUESTS,
            },
            id="granite-vllm-keda-single-gpu",
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
        model_namespace: Namespace,
        vllm_cuda_serving_runtime,
        admin_client: DynamicClient,
        stressed_keda_vllm_inference_service: Generator[InferenceService, Any, Any],
    ):
        """Test KEDA ScaledObject configuration for GPU-based inference service."""
        verify_keda_scaledobject(
            client=admin_client,
            isvc=stressed_keda_vllm_inference_service,
            expected_trigger_type="prometheus",
            expected_query=VLLM_METRICS_QUERY_REQUESTS,
            expected_threshold=VLLM_METRICS_THRESHOLD_REQUESTS,
        )

    def test_vllm_keda_scaling_verify_metrics(
        self,
        model_namespace: Namespace,
        admin_client: DynamicClient,
        vllm_cuda_serving_runtime,
        stressed_keda_vllm_inference_service: Generator[InferenceService, Any, Any],
        prometheus,
    ):
        """Test that vLLM metrics are available and above the expected threshold."""
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=VLLM_METRICS_QUERY_REQUESTS,
            expected_value=str(VLLM_METRICS_THRESHOLD_REQUESTS),
            greater_than=True,
        )

    def test_vllm_keda_scaling_verify_final_pod_count(
        self,
        model_namespace: Namespace,
        admin_client: DynamicClient,
        vllm_cuda_serving_runtime,
        stressed_keda_vllm_inference_service: Generator[InferenceService, Any, Any],
    ):
        """Test that pods scale up to the expected count after load generation."""
        verify_final_pod_count(
            unprivileged_client=admin_client,
            isvc=stressed_keda_vllm_inference_service,
            final_pod_count=FINAL_POD_COUNT,
        )
