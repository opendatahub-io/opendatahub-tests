from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace

from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG, GRANITE_SERVING_ARGUMENT
from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
    verify_final_pod_count,
    verify_keda_scaledobject,
)
from utilities.constants import ModelFormat, ModelVersion, Protocols, RunTimeConfigs
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.monitoring import validate_metrics_field

LOGGER = structlog.get_logger(name=__name__)


BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = GRANITE_SERVING_ARGUMENT

INITIAL_POD_COUNT = 1
FINAL_POD_COUNT = 5

MLSERVER_MODEL_NAMESPACE = "test-mlserver-keda"
MLSERVER_MODEL_NAME = "onnx-raw"
# MLServer exposes the request counter on its metrics port; the `model` label is
# the InferenceService name (KServe sets MLSERVER_MODEL_NAME to the ISVC name).
MLSERVER_METRICS_QUERY = (
    f'sum(sum_over_time(model_infer_request_success_total{{namespace="{MLSERVER_MODEL_NAMESPACE}", '
    f'model="{MLSERVER_MODEL_NAME}"}}[5m]))'
)
MLSERVER_METRICS_THRESHOLD = 2.0

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.keda,
    pytest.mark.usefixtures("valid_aws_config"),
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, mlserver_kserve_serving_runtime, stressed_mlserver_keda_inference_service",
    [
        pytest.param(
            {"name": MLSERVER_MODEL_NAMESPACE},
            RunTimeConfigs.MLSERVER_RUNTIME_CONFIG,
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.V1,
                "model-dir": "test-dir",
                "model-name": MLSERVER_MODEL_NAME,
                "initial_pod_count": INITIAL_POD_COUNT,
                "final_pod_count": FINAL_POD_COUNT,
                "metrics_query": MLSERVER_METRICS_QUERY,
                "metrics_threshold": MLSERVER_METRICS_THRESHOLD,
            },
        )
    ],
    indirect=True,
)
class TestMLServerKedaScaling:
    """
    Test Keda functionality for a cpu based inference service.
    This class verifies pod scaling, metrics availability, and the creation of a keda scaled object.
    """

    def test_mlserver_keda_scaling_verify_scaledobject(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        mlserver_kserve_serving_runtime,
        stressed_mlserver_keda_inference_service: Generator[InferenceService, Any, Any],
        admin_client: DynamicClient,
    ):
        """Test KEDA ScaledObject configuration and run inference multiple times to trigger scaling."""
        verify_keda_scaledobject(
            client=unprivileged_client,
            isvc=stressed_mlserver_keda_inference_service,
            expected_trigger_type="prometheus",
            expected_query=MLSERVER_METRICS_QUERY,
            expected_threshold=MLSERVER_METRICS_THRESHOLD,
        )
        # Run inference multiple times to test KEDA scaling
        run_inference_multiple_times(
            isvc=stressed_mlserver_keda_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            model_name=MLSERVER_MODEL_NAME,
            iterations=10,
            run_in_parallel=True,
        )

    def test_mlserver_keda_scaling_verify_metrics(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        mlserver_kserve_serving_runtime,
        stressed_mlserver_keda_inference_service: Generator[InferenceService, Any, Any],
        prometheus,
    ):
        """Test that MLServer metrics are available and above the expected threshold."""
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=MLSERVER_METRICS_QUERY,
            expected_value=MLSERVER_METRICS_THRESHOLD,
            greater_than=True,
            timeout=300,
        )

    def test_mlserver_keda_scaling_verify_final_pod_count(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        mlserver_kserve_serving_runtime,
        stressed_mlserver_keda_inference_service: Generator[InferenceService, Any, Any],
    ):
        """Test that pods scale up to the expected count after load generation."""
        verify_final_pod_count(
            unprivileged_client=unprivileged_client,
            isvc=stressed_mlserver_keda_inference_service,
            final_pod_count=FINAL_POD_COUNT,
        )
