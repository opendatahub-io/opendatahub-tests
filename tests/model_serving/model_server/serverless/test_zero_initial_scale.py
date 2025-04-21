import pytest
from ocp_resources.deployment import Deployment

from tests.model_serving.model_server.serverless.utils import verify_no_inference_pods
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelVersion,
    Protocols,
    RunTimeConfigs,
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
    "model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "serverless-initial-scale-zero"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": ModelFormat.ONNX,
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "deployment-mode": KServeDeploymentType.SERVERLESS,
                "min-replicas": 0,
            },
        )
    ],
    indirect=True,
)
class TestServerlessInitialScaleZero:
    @pytest.mark.dependency(name="test_no_serverless_pods_created_for_zero_initial_scale")
    def test_no_serverless_pods_created_for_zero_initial_scale(self, admin_client, ovms_kserve_inference_service):
        """Verify no pods are created when inference service initial scale is zero, i.e. zero min-replicas requested."""
        verify_no_inference_pods(client=admin_client, isvc=ovms_kserve_inference_service)

    @pytest.mark.dependency(name="test_no_serverless_replicas_created_for_zero_initial_scale")
    def test_no_serverless_replicas_created_for_zero_initial_scale(self, admin_client, ovms_kserve_inference_service):
        """Verify replica count is zero when inference service initial scale is zero"""
        for deployment in Deployment.get(
            client=admin_client,
            namespace=ovms_kserve_inference_service.namespace,
        ):
            if deployment.labels["serving.knative.dev/configurationGeneration"] == "1":
                if deployment.replicas == 0:
                    deployment.wait_for_replicas()
                    return
                raise DeploymentValidationError(
                    f"Inference Service {ovms_kserve_inference_service.name} deployment should have 0 replicas"
                )

        raise DeploymentValidationError(
            f"Inference Service {ovms_kserve_inference_service.name} new deployment not found"
        )

    @pytest.mark.dependency(
        depends=[
            "test_no_serverless_pods_created_for_zero_initial_scale",
            "test_no_serverless_replicas_created_for_zero_initial_scale",
        ]
    )
    def test_serverless_inference_after_zero_initial_scale(self, ovms_kserve_inference_service):
        """Verify model can be queried after being created with an initial scale of zero."""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
