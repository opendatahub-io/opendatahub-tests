import pytest
from simple_logger.logger import get_logger
from typing import Any, Generator
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_server.utils import (
    verify_keda_scaledobject,
    run_vllm_concurrent_load,
    run_ovms_concurrent_load,
)
from utilities.infra import get_pods_by_isvc_label
from tests.model_serving.model_runtime.vllm.constant import (
    COMPLETION_QUERY,
    CHAT_QUERY,
    BASE_RAW_DEPLOYMENT_CONFIG,
)
from utilities.constants import ModelFormat, ModelVersion, RunTimeConfigs

LOGGER = get_logger(name=__name__)


SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
    "--chat-template=/app/data/template/template_chatglm.jinja",
]

MODEL_PATH: str = "granite-7b-starter"

BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

INITIAL_POD_COUNT = 1
FINAL_POD_COUNT = 5

pytestmark = pytest.mark.usefixtures("skip_if_no_supported_gpu_type", "valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_serving_runtime, keda_vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-vllm-keda"},
            {"model-dir": MODEL_PATH},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                "gpu_count": 1,
                "name": "granite-vllm-keda",
            },
            id="granite-vllm-keda-single-gpu",
        ),
    ],
    indirect=True,
)
class TestVllmKedaScaling:
    def test_vllm_keda_scaling(
        self,
        unprivileged_client: DynamicClient,
        keda_vllm_inference_service: Generator[InferenceService, Any, Any],
        response_snapshot: Any,
        prometheus,
    ):
        initial_pod_count = len(get_pods_by_isvc_label(client=unprivileged_client, isvc=keda_vllm_inference_service))
        assert initial_pod_count == INITIAL_POD_COUNT, (
            f"Initial pod count {initial_pod_count} does not match expected {INITIAL_POD_COUNT}"
        )

        run_vllm_concurrent_load(
            isvc=keda_vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )

        final_pod_count = len(get_pods_by_isvc_label(client=unprivileged_client, isvc=keda_vllm_inference_service))
        assert final_pod_count == FINAL_POD_COUNT, (
            f"Final pod count {final_pod_count} does not match expected {FINAL_POD_COUNT}"
        )

        # validate_metrics_field(
        #     prometheus=prometheus,
        #     metrics_query="vllm:num_requests_running",
        #     expected_value=str(2),
        # )
        verify_keda_scaledobject(
            client=unprivileged_client,
            isvc=keda_vllm_inference_service,
            expected_trigger_type="prometheus",
            expected_query="vllm:num_requests_running",
            expected_threshold=2,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_keda_inference_service",
    [
        pytest.param(
            {"name": "ovms-keda"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {"name": ModelFormat.ONNX, "model-version": ModelVersion.OPSET13, "model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestOVMSKedaScaling:
    def test_ovms_keda_scaling(
        self,
        unprivileged_client: DynamicClient,
        ovms_keda_inference_service: Generator[InferenceService, Any, Any],
        prometheus,
    ):
        initial_pod_count = len(get_pods_by_isvc_label(client=unprivileged_client, isvc=ovms_keda_inference_service))
        assert initial_pod_count == INITIAL_POD_COUNT, (
            f"Initial pod count {initial_pod_count} does not match expected {INITIAL_POD_COUNT}"
        )

        run_ovms_concurrent_load(isvc=ovms_keda_inference_service)

        final_pod_count = len(get_pods_by_isvc_label(client=unprivileged_client, isvc=ovms_keda_inference_service))
        assert final_pod_count == FINAL_POD_COUNT, (
            f"Final pod count {final_pod_count} does not match expected {FINAL_POD_COUNT}"
        )

        # validate_metrics_field(
        #     prometheus=prometheus,
        #     metrics_query="ovms_requests_success",
        #     expected_value=str(50),
        #     greater_than=True,
        # )

        verify_keda_scaledobject(
            client=unprivileged_client,
            isvc=ovms_keda_inference_service,
            expected_trigger_type="prometheus",
            expected_query="ovms_requests_success",
            expected_threshold=50,
        )
