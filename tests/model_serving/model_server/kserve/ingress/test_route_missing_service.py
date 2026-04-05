"""
Tests for route reconciler handling of missing predictor service.

Bug 8 (Medium): The raw route reconciler returns (nil, nil) when the predictor
Service doesn't exist yet, causing no Route to be created and no requeue to
happen — a silent race condition.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.route import Route

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelVersion, Protocols, RunTimeConfigs
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


def _get_routes_for_isvc(admin_client: DynamicClient, namespace: str, isvc_name: str) -> list[Route]:
    """Get Routes associated with an InferenceService."""
    routes = []
    for route in Route.get(client=admin_client, namespace=namespace):
        owner_refs = route.instance.metadata.get("ownerReferences", [])
        if any(ref.get("name") == isvc_name for ref in owner_refs) or isvc_name in route.name:
            routes.append(route)
    return routes


@pytest.mark.tier2
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "kserve-route-missing-svc"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {"name": ModelFormat.ONNX, "model-version": ModelVersion.OPSET13, "model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestRouteMissingService:
    """Verify that Routes are eventually created even with timing races.

    Preconditions:
        - OVMS ONNX model deployed as raw deployment

    Test Steps:
        1. Deploy ISVC and wait for Ready
        2. Verify Route exists for the ISVC
        3. Verify inference works through the Route

    Expected Results:
        - Route should be created and functional
        - Inference should work end-to-end
    """

    def test_route_exists_after_deployment(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace,
        ovms_raw_inference_service: InferenceService,
    ):
        """Verify a Route is created for the ISVC after deployment."""
        routes = _get_routes_for_isvc(
            admin_client=admin_client,
            namespace=unprivileged_model_namespace.name,
            isvc_name=ovms_raw_inference_service.name,
        )
        assert len(routes) > 0, (
            f"No Route found for ISVC {ovms_raw_inference_service.name}. "
            "The route reconciler may have returned (nil, nil) for a missing Service (Bug 8)."
        )

    def test_inference_via_route(self, ovms_raw_inference_service: InferenceService):
        """Verify inference works through the reconciled Route."""
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
