import pytest
from simple_logger.logger import get_logger
from typing import Any, Generator
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.inference_service import InferenceService
from tests.model_serving.model_server.utils import verify_keda_scaledobject, verify_final_pod_count
from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_runtime.vllm.basic_model_deployment.test_granite_7b_starter import SERVING_ARGUMENT
from utilities.constants import ModelFormat, ModelVersion, RunTimeConfigs
from utilities.monitoring import validate_metrics_field

LOGGER = get_logger(name=__name__)


BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

INITIAL_POD_COUNT = 1
FINAL_POD_COUNT = 5

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

pytestmark = [pytest.mark.keda, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, stressed_ovms_keda_inference_service",
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
    """
    Test Keda functionality for a cpu based inference service.
    This class verifies pod scaling, metrics availability, and the creation of a keda scaled object.
    """

    def test_ovms_keda_scaling_verify_scaledobject(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        ovms_kserve_serving_runtime,
        stressed_ovms_keda_inference_service: Generator[InferenceService, Any, Any],
    ):
        verify_keda_scaledobject(
            client=unprivileged_client,
            isvc=stressed_ovms_keda_inference_service,
            expected_trigger_type="prometheus",
            expected_query=OVMS_METRICS_QUERY,
            expected_threshold=OVMS_METRICS_THRESHOLD,
        )

    def test_ovms_keda_scaling_verify_metrics(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        ovms_kserve_serving_runtime,
        stressed_ovms_keda_inference_service: Generator[InferenceService, Any, Any],
        prometheus,
    ):
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=OVMS_METRICS_QUERY,
            expected_value=str(OVMS_METRICS_THRESHOLD),
            greater_than=True,
        )

    def test_ovms_keda_scaling_verify_final_pod_count(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        ovms_kserve_serving_runtime,
        stressed_ovms_keda_inference_service: Generator[InferenceService, Any, Any],
    ):
        verify_final_pod_count(
            unprivileged_client=unprivileged_client,
            isvc=stressed_ovms_keda_inference_service,
            final_pod_count=FINAL_POD_COUNT,
        )
