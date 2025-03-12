import pytest
from ocp_resources.deployment import Deployment

from tests.model_serving.model_server.serverless.utils import verify_no_inference_pods
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelFormat,
    ModelInferenceRuntime,
    ModelVersion,
    Protocols,
)
from utilities.exceptions import DeploymentValidationError
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.serverless,
    pytest.mark.sanity,
    pytest.mark.usefixtures("valid_aws_config"),
]


@pytest.mark.serverless
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_serverless_inference_service",
    [
        pytest.param(
            {"name": "serverless-scale-zero"},
            {
                "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
                "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
            },
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
            },
        )
    ],
    indirect=True,
)
class TestServerlessScaleToZero:
    def test_serverless_before_scale_to_zero(self, ovms_serverless_inference_service):
        """Verify model can be queried before scaling to zero"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "inference_service_patched_replicas",
        [pytest.param({"min-replicas": 0})],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_no_serverless_pods_after_scale_to_zero")
    def test_no_serverless_pods_after_scale_to_zero(self, admin_client, inference_service_patched_replicas):
        """Verify pods are scaled to zero"""
        verify_no_inference_pods(client=admin_client, isvc=inference_service_patched_replicas)

    @pytest.mark.dependency(depends=["test_no_serverless_pods_after_scale_to_zero"])
    def test_serverless_inference_after_scale_to_zero(self, ovms_serverless_inference_service):
        """Verify model can be queried after scaling to zero"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_no_serverless_pods_after_scale_to_zero"])
    def test_no_serverless_pods_when_no_traffic(self, admin_client, ovms_serverless_inference_service):
        """Verify pods are scaled to zero when no traffic is sent"""
        verify_no_inference_pods(client=admin_client, isvc=ovms_serverless_inference_service)

    @pytest.mark.parametrize(
        "inference_service_patched_replicas",
        [pytest.param({"min-replicas": 1})],
        indirect=True,
    )
    def test_serverless_pods_after_scale_to_one_replica(self, admin_client, inference_service_patched_replicas):
        """Verify pod is running after scaling to 1 replica"""
        for deployment in Deployment.get(
            client=admin_client,
            namespace=inference_service_patched_replicas.namespace,
        ):
            if deployment.labels["serving.knative.dev/configurationGeneration"] == "3":
                deployment.wait_for_replicas()
                return

        raise DeploymentValidationError(
            f"Inference Service {inference_service_patched_replicas.name} new deployment not found"
        )
