import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelVersion,
    Protocols,
    RunTimeConfigs,
    Annotations,
)
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG
from tests.model_serving.model_server.serverless.utils import verify_no_inference_pods

pytestmark = [pytest.mark.serverless, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.rawdeployment
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
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "stop": "False",
            },
        )
    ],
    indirect=True,
)
class TestStopRaw:
    @pytest.mark.smoke
    def test_raw_onnx_rest_inference(self, unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service):
        """Verify that kserve Raw ONNX model can be queried using REST"""
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_raw_inference_service_stop_annotation_true",
        [
            pytest.param(
                {"annotations": {Annotations.KserveIo.FORCE_STOP_RUNTIME: "true"}}
            )
        ],
        indirect=True,
    )
    def test_stop_ann_update_to_true_delete_pod_rollout(
        self,
        unprivileged_client,
        unprivileged_model_namespace,
        ovms_kserve_serving_runtime,
        ovms_raw_inference_service,
        patched_raw_inference_service_stop_annotation_true,
    ):
        """Verify pod rollout is deleted when the stop annotation updated to true"""
        """Verify pods do not exist"""
        verify_no_inference_pods(
            client=unprivileged_client,
            isvc=patched_raw_inference_service_stop_annotation_true,
        )

@pytest.mark.rawdeployment
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
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "stop": "True",
            },
        )
    ],
    indirect=True,
)
class TestStoppedResumeRaw:
    @pytest.mark.smoke
    def test_stop_ann_true_no_pod_rollout(
        self,
        unprivileged_client,
        unprivileged_model_namespace,
        ovms_kserve_serving_runtime,
        ovms_raw_inference_service,
    ):
        """Verify no pod rollout when the stop annotation is true"""
        """Verify pods do not exist"""
        verify_no_inference_pods(
            client=unprivileged_client,
            isvc=ovms_raw_inference_service,
        )

    @pytest.mark.parametrize(
        "patched_raw_inference_service_stop_annotation_false",
        [
            pytest.param(
                {"annotations": {Annotations.KserveIo.FORCE_STOP_RUNTIME: "false"}}
            )
        ],
        indirect=True,
    )
    def test_stop_ann_update_to_false_pod_rollout(self, unprivileged_client, unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service, patched_raw_inference_service_stop_annotation_false):
        """Verify pod rollout when the stop annotation is updated to false"""
        """Verify that kserve Raw ONNX model can be queried using REST"""
        verify_inference_response(
            inference_service=patched_raw_inference_service_stop_annotation_false,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
