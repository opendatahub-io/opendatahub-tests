import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelVersion, Protocols, ModelInferenceRuntime
from utilities.inference_utils import Inference
from utilities.infra import get_model_mesh_route
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-onnx"},
            {
                "runtime-name": ModelInferenceRuntime.ONNX_RUNTIME,
                "model-format": {ModelFormat.ONNX: ModelVersion.OPSET13},
            },
            {"name": ModelFormat.ONNX, "model-version": ModelVersion.OPSET13, "model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestONNXRaw:
    """Test suite for  Validating reconciliation """

    @pytest.mark.smoke
    @pytest.mark.jira("RHOAIENG-9045")
    def test_raw_onnx_rout_reconciliation(self, admin_client, ovms_raw_inference_service):
        """
        Verify that the KServe Raw ONNX model can be queried using REST
        and ensure that the model rout reconciliation works correctly .
        """
        # Initial inference validation
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

        # Validate ingress status before and after route deletion
        self.assert_ingress_status_changed(admin_client, ovms_raw_inference_service)

        # Final inference validation after route update
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @staticmethod
    def assert_ingress_status_changed(admin_client, inference_service):
        """
        Validates that the ingress status changes correctly after route deletion.
        """
        route = get_model_mesh_route(admin_client, inference_service)

        initial_status = route.instance.status["ingress"][0]["conditions"][0]
        initial_host = route.host
        initial_transition_time = initial_status["lastTransitionTime"]
        initial_status_value = initial_status["status"]

        route.delete()

        updated_status = route.instance.status["ingress"][0]["conditions"][0]
        updated_host = route.host
        updated_transition_time = updated_status["lastTransitionTime"]
        updated_status_value = updated_status["status"]

        assert updated_host == initial_host, "Host should remain the same after route update"
        assert updated_transition_time != initial_transition_time, "Transition time should change"
        assert updated_status_value == "True", "Updated ingress status should be True"
        assert initial_status_value == "True", "Initial ingress status should be True"
