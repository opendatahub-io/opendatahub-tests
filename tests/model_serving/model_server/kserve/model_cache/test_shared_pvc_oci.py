"""Downstream qualification for OCI import into shared NFS storage."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LocalModelNamespaceCache,
    assert_modelcar_absent_from_node_image_cache,
    assert_predictors_share_read_only_pvc,
    cache_status_dict,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import ModelCarImage, ModelFormat, ModelName, Protocols, RuntimeTemplates
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.rawdeployment,
    pytest.mark.slow,
    pytest.mark.usefixtures("skip_if_no_nfs_storage_class"),
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template",
    [
        pytest.param(
            {"name": f"{ModelFormat.OPENVINO}-model-cache-shared-pvc"},
            {
                "name": f"{ModelName.MNIST}-runtime",
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
            },
            id="test_public_oci_shared_pvc",
        )
    ],
    indirect=True,
)
class TestPublicOCISharedPVC:
    """Validate a public OCI import shared by two NFS-backed predictor replicas.

    Steps:
        1. Import the standard MNIST ModelCar into a pre-provisioned NFS RWX PVC.
        2. Serve two replicas on different nodes from the same read-only claim.
        3. Verify inference and the absence of transfer containers, local model copies, and CRI-O image pulls.
    """

    def test_import_and_multi_replica_serving(
        self,
        unprivileged_client: DynamicClient,
        serving_runtime_from_template: ServingRuntime,
        shared_model_cache_pvc: PersistentVolumeClaim,
        modelcar_shared_pvc_cache: LocalModelNamespaceCache,
        modelcar_shared_pvc_isvc: InferenceService,
    ) -> None:
        """Verify the standard OCI model is imported once and served by two PVC-backed replicas."""
        status = cache_status_dict(cache=modelcar_shared_pvc_cache)
        ready = next(condition for condition in status["conditions"] if condition["type"] == "Ready")
        assert ready["status"] == "True" and ready["reason"] == "ImportSucceeded"
        assert status.get("copies") == {"available": 1, "failed": 0, "total": 1}
        assert not status.get("nodeStatus")

        pods = assert_predictors_share_read_only_pvc(
            client=unprivileged_client,
            isvc=modelcar_shared_pvc_isvc,
            runtime_name=serving_runtime_from_template.name,
            pvc_name=shared_model_cache_pvc.name,
        )
        verify_inference_response(
            inference_service=modelcar_shared_pvc_isvc,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
        assert_modelcar_absent_from_node_image_cache(pods=pods, source_uri=ModelCarImage.MNIST_8_1)
