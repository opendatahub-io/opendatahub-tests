import pytest
from simple_logger.logger import get_logger
from typing import Any, Generator
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from utilities.constants import KServeDeploymentType, Timeout
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
from tests.model_serving.model_runtime.vllm.basic_model_deployment.test_granite_7b_starter import (
    SERVING_ARGUMENT,
    MODEL_PATH,
)
from tests.model_serving.model_server.serverless.utils import inference_service_pods_sampler
from utilities.constants import ModelFormat, ModelVersion, RunTimeConfigs
from utilities.monitoring import validate_metrics_field

LOGGER = get_logger(name=__name__)


BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

INITIAL_POD_COUNT = 1
FINAL_POD_COUNT = 5

VLLM_MODEL_NAME = "granite-vllm-keda"
VLLM_METRICS_QUERY_GPU = f'sum_over_time(vllm:gpu_cache_usage_perc \
    {{namespace="{VLLM_MODEL_NAME}",pod=~"{VLLM_MODEL_NAME}.*"}}[5m])'
VLLM_METRICS_THRESHOLD_GPU = 0.02
VLLM_METRICS_QUERY_REQUESTS = f'vllm:num_requests_running{{namespace="{VLLM_MODEL_NAME}",pod=~"{VLLM_MODEL_NAME}.*"}}'
VLLM_METRICS_THRESHOLD_REQUESTS = 4

OVMS_MODEL_NAMESPACE = "ovms-keda"
OVMS_MODEL_NAME = "onnx-raw"
OVMS_METRICS_QUERY = (
    f"sum by (name) (rate(ovms_inference_time_us_sum{{"
    f"namespace='{OVMS_MODEL_NAMESPACE}', name='{OVMS_MODEL_NAME}'"
    f"}}[5m])) / "
    f"sum by (name) (rate(ovms_inference_time_us_count{{"
    f"namespace='{OVMS_MODEL_NAMESPACE}', name='{OVMS_MODEL_NAME}'"
    f"}}[5m]))"
)
OVMS_METRICS_THRESHOLD = 200

pytestmark = [pytest.mark.keda, pytest.mark.usefixtures("skip_if_no_supported_gpu_type", "valid_aws_config")]


@pytest.fixture
def verify_initial_pod_count(unprivileged_client: DynamicClient):
    """Verify initial pod count before running load tests."""

    def _verify_initial_pod_count(isvc: InferenceService, expected_count: int):
        initial_pod_count = len(get_pods_by_isvc_label(client=unprivileged_client, isvc=isvc))
        assert initial_pod_count == expected_count, (
            f"Initial pod count {initial_pod_count} does not match expected {expected_count}"
        )

    return _verify_initial_pod_count


@pytest.fixture
def verify_final_pod_count(unprivileged_client: DynamicClient):
    """Verify final pod count after running load tests."""

    def _verify_final_pod_count(isvc: InferenceService):
        for pods in inference_service_pods_sampler(
            client=unprivileged_client,
            isvc=isvc,
            timeout=Timeout.TIMEOUT_5MIN,
            sleep=10,
        ):
            if pods:
                assert len(pods) == FINAL_POD_COUNT, (
                    f"Final pod count {len(pods)} does not match expected {FINAL_POD_COUNT}"
                )

    return _verify_final_pod_count


@pytest.fixture
def pod_resource(unprivileged_client: DynamicClient, keda_vllm_inference_service: InferenceService) -> Pod:
    return get_pods_by_isvc_label(client=unprivileged_client, isvc=keda_vllm_inference_service)[0]


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, vllm_serving_runtime, keda_vllm_inference_service",
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
    def test_vllm_keda_scaling(
        self,
        unprivileged_client: DynamicClient,
        keda_vllm_inference_service: Generator[InferenceService, Any, Any],
        response_snapshot: Any,
        prometheus,
        verify_initial_pod_count,
        verify_final_pod_count,
        pod_resource: Pod,
    ):
        verify_initial_pod_count(isvc=keda_vllm_inference_service, expected_count=INITIAL_POD_COUNT)

        run_vllm_concurrent_load(
            pod_name=pod_resource.name,
            isvc=keda_vllm_inference_service,
            response_snapshot=response_snapshot,
            chat_query=CHAT_QUERY,
            completion_query=COMPLETION_QUERY,
        )

        verify_final_pod_count(isvc=keda_vllm_inference_service)

        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=VLLM_METRICS_QUERY_REQUESTS,
            expected_value=str(VLLM_METRICS_THRESHOLD_REQUESTS),
            greater_than=True,
        )

        verify_keda_scaledobject(
            client=unprivileged_client,
            isvc=keda_vllm_inference_service,
            expected_trigger_type="prometheus",
            expected_query=VLLM_METRICS_QUERY_REQUESTS,
            expected_threshold=VLLM_METRICS_THRESHOLD_REQUESTS,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_keda_inference_service",
    [
        pytest.param(
            {"name": "ovms-keda"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "initial_pod_count": INITIAL_POD_COUNT,
                "final_pod_count": FINAL_POD_COUNT,
                "metrics_query": OVMS_METRICS_QUERY,
                "metrics_threshold": OVMS_METRICS_THRESHOLD,
            },
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
        verify_initial_pod_count,
        verify_final_pod_count,
    ):
        verify_initial_pod_count(isvc=ovms_keda_inference_service, expected_count=INITIAL_POD_COUNT)

        run_ovms_concurrent_load(isvc=ovms_keda_inference_service)

        verify_final_pod_count(isvc=ovms_keda_inference_service)

        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=OVMS_METRICS_QUERY,
            expected_value=str(OVMS_METRICS_THRESHOLD),
            greater_than=True,
        )

        verify_keda_scaledobject(
            client=unprivileged_client,
            isvc=ovms_keda_inference_service,
            expected_trigger_type="prometheus",
            expected_query=OVMS_METRICS_QUERY,
            expected_threshold=OVMS_METRICS_THRESHOLD,
        )
