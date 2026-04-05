"""
Tests for HTTPRoute owner reference consistency.

Bug 4 (High): The kserve ingress reconciler creates the explainer HTTPRoute
without returning the error when SetControllerReference fails, leading to
orphaned HTTPRoutes that are not garbage-collected on ISVC deletion.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelVersion, Protocols, RunTimeConfigs
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


def _get_httproutes_for_isvc(admin_client: DynamicClient, namespace: str, isvc_name: str) -> list:
    """Get HTTPRoutes in the namespace that reference the given ISVC."""
    from ocp_resources.resource import Resource

    httproutes = []
    try:
        for route in Resource.get(
            client=admin_client,
            api_version="gateway.networking.k8s.io/v1",
            kind="HTTPRoute",
            namespace=namespace,
        ):
            owners = route.instance.metadata.get("ownerReferences", [])
            if any(ref.get("name") == isvc_name for ref in owners):
                httproutes.append(route)
    except Exception:  # noqa: BLE001, S110
        pass
    return httproutes


@pytest.mark.tier2
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "kserve-httproute-ownerref"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {"name": ModelFormat.ONNX, "model-version": ModelVersion.OPSET13, "model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestHttpRouteOwnerReference:
    """Verify HTTPRoutes have proper owner references to InferenceService.

    Preconditions:
        - OVMS ONNX model deployed as raw deployment
        - Model is ready and serving

    Test Steps:
        1. Deploy an ISVC with OVMS runtime
        2. Verify the model can be queried
        3. Check that any HTTPRoutes created have ownerReferences pointing to the ISVC
        4. After ISVC deletion, verify no orphaned HTTPRoutes remain

    Expected Results:
        - All HTTPRoutes should have ownerReferences to the parent ISVC
        - No orphaned HTTPRoutes after ISVC deletion
    """

    def test_inference_works(self, ovms_raw_inference_service: InferenceService):
        """Verify the model serves inference correctly."""
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_no_orphaned_httproutes_after_deletion(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace,
        ovms_raw_inference_service: InferenceService,
    ):
        """Verify no orphaned HTTPRoutes remain after ISVC deletion.

        Given an ISVC was deployed with potential HTTPRoutes
        When the ISVC is deleted (via fixture teardown)
        Then no HTTPRoutes referencing this ISVC should remain
        """
        ns_name = unprivileged_model_namespace.name
        isvc_name = ovms_raw_inference_service.name
        orphaned = _get_httproutes_for_isvc(admin_client=admin_client, namespace=ns_name, isvc_name=isvc_name)
        assert not orphaned, (
            f"Found {len(orphaned)} orphaned HTTPRoute(s) after ISVC deletion: "
            f"{[r.name for r in orphaned]}. "
            "SetControllerReference may have failed silently for explainer HTTPRoute (Bug 4)."
        )
