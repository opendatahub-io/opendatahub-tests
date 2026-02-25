from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.vllm.basic_model_deployment.test_granite_7b_starter import SERVING_ARGUMENT
from tests.model_serving.model_runtime.vllm.constant import BASE_RAW_DEPLOYMENT_CONFIG
from tests.model_serving.model_server.utils import (
    run_inference_multiple_times,
    verify_final_pod_count,
    verify_keda_scaledobject,
)
from utilities.constants import ModelFormat, ModelVersion, Protocols, RunTimeConfigs, Timeout
from utilities.inference_utils import Inference
from utilities.jira import is_jira_open
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from utilities.monitoring import validate_metrics_field

LOGGER = get_logger(name=__name__)


BASE_RAW_DEPLOYMENT_CONFIG["runtime_argument"] = SERVING_ARGUMENT

INITIAL_POD_COUNT = 1
FINAL_POD_COUNT = 5

OVMS_MODEL_NAMESPACE = "test-ovms-keda"
OVMS_MODEL_NAME = "onnx-raw"
OVMS_METRICS_QUERY = (
    f'sum(sum_over_time(ovms_requests_success{{namespace="{OVMS_MODEL_NAMESPACE}", name="{OVMS_MODEL_NAME}"}}[5m]))'
)
OVMS_METRICS_THRESHOLD = 2.0

pytestmark = [pytest.mark.keda, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, stressed_ovms_keda_inference_service",
    [
        pytest.param(
            {"name": OVMS_MODEL_NAMESPACE},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "model-name": OVMS_MODEL_NAME,
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
        admin_client: DynamicClient,
    ):
        """Test KEDA ScaledObject configuration and run inference multiple times to trigger scaling."""

        if is_jira_open(jira_id="RHOAIENG-31386", admin_client=admin_client):
            isvc_dict = stressed_ovms_keda_inference_service.instance.to_dict()
            metrics = isvc_dict.get("spec", {}).get("predictor", {}).get("autoScaling", {}).get("metrics", [])

            if metrics and isinstance(metrics[0], dict) and metrics[0].get("external") is not None:
                metrics[0].setdefault("external", {})["authenticationRef"] = {
                    "authModes": "bearer",
                    "authenticationRef": {"name": "inference-prometheus-auth"},
                }

                ResourceEditor(
                    patches={
                        stressed_ovms_keda_inference_service: {
                            "spec": {
                                "predictor": {
                                    "autoScaling": {
                                        "metrics": metrics,
                                    }
                                }
                            }
                        }
                    }
                ).update()

        verify_keda_scaledobject(
            client=unprivileged_client,
            isvc=stressed_ovms_keda_inference_service,
            expected_trigger_type="prometheus",
            expected_query=OVMS_METRICS_QUERY,
            expected_threshold=OVMS_METRICS_THRESHOLD,
        )
        # Run inference multiple times to test KEDA scaling
        run_inference_multiple_times(
            isvc=stressed_ovms_keda_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            model_name=OVMS_MODEL_NAME,
            iterations=10,
            run_in_parallel=True,
        )

    def test_ovms_keda_scaling_verify_metrics(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        ovms_kserve_serving_runtime,
        stressed_ovms_keda_inference_service: Generator[InferenceService, Any, Any],
        prometheus,
    ):
        """Test that OVMS metrics are available and above the expected threshold."""
        validate_metrics_field(
            prometheus=prometheus,
            metrics_query=OVMS_METRICS_QUERY,
            expected_value=OVMS_METRICS_THRESHOLD,
            greater_than=True,
            timeout=Timeout.TIMEOUT_5MIN,
        )

    def test_ovms_keda_scaling_verify_final_pod_count(
        self,
        unprivileged_model_namespace: Namespace,
        unprivileged_client: DynamicClient,
        ovms_kserve_serving_runtime,
        stressed_ovms_keda_inference_service: Generator[InferenceService, Any, Any],
    ):
        """Test that pods scale up to the expected count after load generation."""
        verify_final_pod_count(
            unprivileged_client=unprivileged_client,
            isvc=stressed_ovms_keda_inference_service,
            final_pod_count=FINAL_POD_COUNT,
        )
