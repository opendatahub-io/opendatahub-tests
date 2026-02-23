import pytest

from tests.model_serving.model_server.kserve.raw_deployment.utils import assert_ingress_status_changed
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelVersion, Protocols, RunTimeConfigs
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "kserve-raw-route-reconciliation"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {"name": ModelFormat.ONNX, "model-version": ModelVersion.OPSET13, "model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestONNXRawRouteReconciliation:
    """Test suite for  Validating reconciliation"""

    @pytest.mark.smoke
    def test_raw_onnx_rout_reconciliation(self, ovms_raw_inference_service):
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

    def test_route_value_before_and_after_deletion(self, unprivileged_client, ovms_raw_inference_service):
        # Validate ingress status before and after route deletion
        assert_ingress_status_changed(client=unprivileged_client, inference_service=ovms_raw_inference_service)

    def test_model_works_after_route_is_recreated(self, ovms_raw_inference_service):
        # Final inference validation after route update
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
