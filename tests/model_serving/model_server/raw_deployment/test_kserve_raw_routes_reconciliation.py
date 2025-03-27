import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelVersion, Protocols, RunTimeConfigs
from utilities.exceptions import ResourceNotFound
from utilities.inference_utils import Inference
from utilities.infra import get_model_route
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_raw_inference_service",
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
    def assert_ingress_status_changed(admin_client: DynamicClient, inference_service: InferenceService):
        """
        Validates that the ingress status changes correctly after route deletion.

        Args:
            admin_client (DynamicClient): The administrative client used to manage the model route.
            inference_service (InferenceService): The inference service whose route status is being checked.

        Raises:
            ResourceNotFound: If the route does not exist before or after deletion.
            AssertionError: If any of the validation checks fail.

        Returns:
            None
        """
        route = get_model_route(admin_client, inference_service)
        if not route.exists:
            raise ResourceNotFound("Route before deletion not found: No active route is currently available.")

        initial_status = route.instance.status["ingress"][0]["conditions"][0]
        initial_host = route.host
        initial_transition_time = initial_status["lastTransitionTime"]
        initial_status_value = initial_status["status"]

        route.delete()

        if not route.exists:
            raise ResourceNotFound("Route after deletion not found: No active route is currently available.")

        updated_status = route.instance.status["ingress"][0]["conditions"][0]
        updated_host = route.host
        updated_transition_time = updated_status["lastTransitionTime"]
        updated_status_value = updated_status["status"]

        # Collect failures instead of stopping at the first failed assertion
        failures = []

        if updated_host != initial_host:
            failures.append(f"Host mismatch: before={initial_host}, after={updated_host}")

        if updated_transition_time == initial_transition_time:
            failures.append(
                f"Transition time did not change: before={initial_transition_time}, after={updated_transition_time}"
            )

        if updated_status_value != "True":
            failures.append(f"Updated ingress status incorrect: expected=True, actual={updated_status_value}")

        if initial_status_value != "True":
            failures.append(f"Initial ingress status incorrect: expected=True, actual={initial_status_value}")

        # Assert all failures at once
        assert not failures, "Ingress status validation failed:\n" + "\n".join(failures)
