import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    Annotations,
    KServeDeploymentType,
    Labels,
    ModelFormat,
    ModelVersion,
    OpenshiftRouteTimeout,
    Protocols,
    RunTimeConfigs,
)
from utilities.exceptions import InferenceResponseError
from utilities.inference_utils import Inference
from utilities.infra import wait_for_route_timeout
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.tier1, pytest.mark.usefixtures("valid_aws_config"), pytest.mark.rawdeployment]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-onnx-mnist-rest"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {
                "name": "onnx-route-visibility",
                "model-version": ModelVersion.OPSET13,
                "model-dir": "test-dir",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
            },
        )
    ],
    indirect=True,
)
class TestRestRawDeploymentRoutes:
    """Validate REST route visibility transitions for KServe raw deployment with MNIST ONNX.

    Steps:
        1. Deploy an ONNX MNIST model as a raw deployment with no external route.
        2. Verify the default route visibility label is not set.
        3. Query the model via the internal route and confirm a successful response.
        4. Patch the ISVC to expose the route externally and verify inference over HTTPS.
        5. Revert the route to local-cluster visibility and verify internal access still works.
    """

    def test_default_visibility_value(self, ovms_kserve_inference_service):
        """Test default route visibility label is absent on a raw deployment."""
        if labels := ovms_kserve_inference_service.labels:
            assert labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) is None

    def test_rest_raw_deployment_internal_route(self, ovms_kserve_inference_service):
        """Test inference succeeds over the internal HTTP route."""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_kserve_isvc_visibility_label",
        [pytest.param({"visibility": Labels.Kserve.EXPOSED})],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_rest_raw_deployment_routes__exposed_route")
    def test_rest_raw_deployment_exposed_route(self, patched_kserve_isvc_visibility_label):
        """Test inference succeeds over the exposed external HTTPS route."""
        verify_inference_response(
            inference_service=patched_kserve_isvc_visibility_label,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_rest_raw_deployment_routes__exposed_route"])
    @pytest.mark.parametrize(
        "patched_kserve_isvc_visibility_label",
        [pytest.param({"visibility": "local-cluster"})],
        indirect=True,
    )
    def test_disabled_rest_raw_deployment_exposed_route(self, patched_kserve_isvc_visibility_label):
        """Test inference succeeds over the internal route after external route is disabled."""
        verify_inference_response(
            inference_service=patched_kserve_isvc_visibility_label,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-onnx-mnist-rest-timeout"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {"name": ModelFormat.ONNX, "model-version": ModelVersion.OPSET13, "model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestRestRawDeploymentRoutesTimeout:
    """Validate REST route timeout behavior for KServe raw deployment with MNIST ONNX.

    Steps:
        1. Deploy an ONNX MNIST model as a raw deployment with an external route.
        2. Verify inference succeeds over the exposed HTTPS route.
        3. Patch the route with an extremely low timeout annotation.
        4. Verify inference fails with a 504 Gateway Time-out error.
    """

    @pytest.mark.dependency(name="test_rest_raw_deployment_routes_timeout__exposed_route")
    def test_rest_raw_deployment_exposed_route(self, ovms_raw_inference_service):
        """Test inference succeeds over the exposed external HTTPS route."""
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "ovms_raw_isvc_patched_annotations",
        [
            pytest.param({
                "annotations": {Annotations.HaproxyRouterOpenshiftIo.TIMEOUT: OpenshiftRouteTimeout.TIMEOUT_1MICROSEC}
            })
        ],
        indirect=True,
    )
    @pytest.mark.dependency(depends=["test_rest_raw_deployment_routes_timeout__exposed_route"])
    def test_rest_raw_deployment_exposed_route_with_timeout(self, ovms_raw_isvc_patched_annotations):
        """Test inference fails with 504 when the route timeout is set to 1 microsecond."""
        wait_for_route_timeout(
            name=ovms_raw_isvc_patched_annotations.name,
            namespace=ovms_raw_isvc_patched_annotations.namespace,
            route_timeout=OpenshiftRouteTimeout.TIMEOUT_1MICROSEC,
        )

        with pytest.raises(InferenceResponseError, match="504"):
            verify_inference_response(
                inference_service=ovms_raw_isvc_patched_annotations,
                inference_config=ONNX_INFERENCE_CONFIG,
                inference_type=Inference.INFER,
                protocol=Protocols.HTTPS,
                use_default_query=True,
            )
