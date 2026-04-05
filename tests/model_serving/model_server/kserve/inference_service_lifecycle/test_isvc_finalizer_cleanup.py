"""
Tests for ISVC finalizer cleanup behavior.

Bug 5 (High): odh-model-controller removes the finalizer even when resource
cleanup fails, potentially leaving orphaned ClusterRoleBindings after ISVC deletion.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelFormat, ModelVersion, Protocols, RunTimeConfigs
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.usefixtures("valid_aws_config")]


def _get_isvc_cluster_role_bindings(admin_client: DynamicClient, isvc_name: str) -> list[ClusterRoleBinding]:
    """Get ClusterRoleBindings associated with an InferenceService."""
    crbs = []
    for crb in ClusterRoleBinding.get(client=admin_client):
        labels = crb.instance.metadata.get("labels", {})
        if labels.get("serving.kserve.io/inferenceservice") == isvc_name:
            crbs.append(crb)
    return crbs


@pytest.mark.tier2
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_raw_inference_service",
    [
        pytest.param(
            {"name": "kserve-finalizer-cleanup"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            {"name": ModelFormat.ONNX, "model-version": ModelVersion.OPSET13, "model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestIsvcFinalizerCleanup:
    """Verify that deleting an ISVC properly cleans up all associated cluster-scoped resources.

    Preconditions:
        - OVMS ONNX model deployed as raw deployment
        - Model is ready and serving

    Test Steps:
        1. Deploy an ISVC with OVMS runtime
        2. Verify the model is queryable
        3. Record any ClusterRoleBindings created for the ISVC
        4. Delete the ISVC
        5. Verify no orphaned ClusterRoleBindings remain

    Expected Results:
        - All ClusterRoleBindings associated with the ISVC are removed after deletion
    """

    def test_inference_before_deletion(self, ovms_raw_inference_service: InferenceService):
        """Verify the model serves inference before we test deletion."""
        verify_inference_response(
            inference_service=ovms_raw_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_no_orphaned_crbs_after_deletion(
        self,
        admin_client: DynamicClient,
        ovms_raw_inference_service: InferenceService,
    ):
        """Verify ClusterRoleBindings are cleaned up when ISVC is deleted.

        Given an ISVC with associated ClusterRoleBindings
        When the ISVC is deleted
        Then all related ClusterRoleBindings should be removed
        """
        isvc_name = ovms_raw_inference_service.name
        orphaned_crbs = _get_isvc_cluster_role_bindings(admin_client=admin_client, isvc_name=isvc_name)
        assert not orphaned_crbs, (
            f"Found {len(orphaned_crbs)} orphaned ClusterRoleBinding(s) after ISVC deletion: "
            f"{[crb.name for crb in orphaned_crbs]}. "
            "The finalizer may have been removed before cleanup completed (Bug 5)."
        )
