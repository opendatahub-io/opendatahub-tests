import pytest

from tests.model_serving.model_server.kserve.stop_resume.utils import consistently_verify_no_pods_exist
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelFormat,
    ModelVersion,
    Protocols,
    RunTimeConfigs,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.rawdeployment
@pytest.mark.smoke
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "kserve-raw-stop-resume"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "stop": False,
            },
        )
    ],
    indirect=True,
)
class TestStopRaw:
    def test_raw_onnx_rest_inference(
        self, unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service
    ):
        """Verify that kserve Raw ONNX model can be queried using REST"""
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_raw_inference_service_stop_annotation",
        [pytest.param({"stop": "true"})],
        indirect=True,
    )
    def test_stop_and_update_to_true_delete_pod_rollout(
        self,
        unprivileged_client,
        unprivileged_model_namespace,
        ovms_kserve_serving_runtime,
        ovms_raw_inference_service,
        patched_raw_inference_service_stop_annotation,
    ):
        """Verify pod rollout is deleted when the stop annotation updated to true"""
        """Verify pods do not exist"""
        result = consistently_verify_no_pods_exist(
            client=unprivileged_client,
            isvc=patched_raw_inference_service_stop_annotation,
        )
        assert result, "Verification failed: pods were found when none should exist"


@pytest.mark.rawdeployment
@pytest.mark.smoke
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "kserve-raw-stop-resume"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "stop": True,
            },
        )
    ],
    indirect=True,
)
class TestStoppedResumeRaw:
    def test_stop_and_true_no_pod_rollout(
        self,
        unprivileged_client,
        unprivileged_model_namespace,
        ovms_kserve_serving_runtime,
        ovms_raw_inference_service,
    ):
        """Verify no pod rollout when the stop annotation is true"""
        """Verify pods do not exist"""
        result = consistently_verify_no_pods_exist(
            client=unprivileged_client,
            isvc=ovms_raw_inference_service,
        )
        assert result, "Verification failed: pods were found when none should exist"

    @pytest.mark.parametrize(
        "patched_raw_inference_service_stop_annotation",
        [pytest.param({"stop": "false"})],
        indirect=True,
    )
    def test_stop_and_update_to_false_pod_rollout(
        self,
        unprivileged_client,
        unprivileged_model_namespace,
        ovms_kserve_serving_runtime,
        ovms_raw_inference_service,
        patched_raw_inference_service_stop_annotation,
    ):
        """Verify pod rollout when the stop annotation is updated to false"""
        """Verify that kserve Raw ONNX model can be queried using REST"""
        verify_inference_response(
            inference_service=patched_raw_inference_service_stop_annotation,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
