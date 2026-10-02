"""Downstream qualification for authenticated OCI import into shared NFS storage."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.serving_runtime import ServingRuntime
from pytest import FixtureRequest

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LocalModelNamespaceCache,
    assert_modelcar_absent_from_node_image_cache,
    assert_predictors_share_read_only_pvc,
    cache_status_dict,
    wait_for_shared_pvc_cache_condition,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols, RunTimeConfigs
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.rawdeployment,
    pytest.mark.slow,
    pytest.mark.usefixtures("skip_if_no_nfs_storage_class"),
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime",
    [
        pytest.param(
            {"name": "kserve-private-oci-shared-pvc"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            id="test_private_oci_shared_pvc",
        )
    ],
    indirect=True,
)
class TestPrivateOCISharedPVC:
    """Validate one authenticated OCI import shared by two NFS-backed predictor replicas.

    Steps:
        1. Import a private ONNX ModelCar into a pre-provisioned NFS RWX PVC.
        2. Serve two replicas on different nodes from the same read-only claim.
        3. Verify inference and the absence of transfer containers, local model copies, and CRI-O image pulls.
    """

    def test_authenticated_import_and_multi_replica_serving(
        self,
        unprivileged_client: DynamicClient,
        ovms_kserve_serving_runtime: ServingRuntime,
        private_modelcar_source: tuple[str, str, str, str],
        shared_model_cache_pvc: PersistentVolumeClaim,
        private_modelcar_shared_pvc_cache: LocalModelNamespaceCache,
        private_modelcar_shared_pvc_isvc: InferenceService,
    ) -> None:
        """Given valid registry credentials, the imported model is served from one shared read-only PVC."""
        status = cache_status_dict(cache=private_modelcar_shared_pvc_cache)
        ready = next(condition for condition in status["conditions"] if condition["type"] == "Ready")
        assert ready["status"] == "True" and ready["reason"] == "ImportSucceeded"
        assert status.get("copies") == {"available": 1, "failed": 0, "total": 1}
        assert not status.get("nodeStatus")

        pods = assert_predictors_share_read_only_pvc(
            client=unprivileged_client,
            isvc=private_modelcar_shared_pvc_isvc,
            runtime_name=ovms_kserve_serving_runtime.name,
            pvc_name=shared_model_cache_pvc.name,
        )
        verify_inference_response(
            inference_service=private_modelcar_shared_pvc_isvc,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
        assert_modelcar_absent_from_node_image_cache(pods=pods, source_uri=private_modelcar_source[0])


@pytest.mark.parametrize(
    "credential_mode, unprivileged_model_namespace",
    [
        pytest.param(
            "missing",
            {"name": "kserve-private-oci-missing-credentials"},
            id="test_missing_registry_credentials",
        ),
        pytest.param(
            "invalid",
            {"name": "kserve-private-oci-invalid-credentials"},
            id="test_invalid_registry_credentials",
        ),
    ],
    indirect=["unprivileged_model_namespace"],
)
@pytest.mark.usefixtures("model_cache_infra_ready")
def test_registry_credentials_failure_is_actionable(
    credential_mode: str,
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    shared_model_cache_pvc: PersistentVolumeClaim,
    private_modelcar_source: tuple[str, str, str, str],
    request: FixtureRequest,
) -> None:
    """Given missing or invalid credentials, the cache reports an actionable non-Ready result."""
    uri, model_size, _, _ = private_modelcar_source
    secret_names = []
    if credential_mode == "invalid":
        secret_names = [request.getfixturevalue("invalid_model_cache_registry_secret").name]
    with LocalModelNamespaceCache(
        client=admin_client,
        name=f"private-oci-{credential_mode}",
        namespace=unprivileged_model_namespace.name,
        source_model_uri=uri,
        model_size=model_size,
        pvc_ref=shared_model_cache_pvc.name,
        image_pull_secrets=secret_names,
    ) as cache:
        status = wait_for_shared_pvc_cache_condition(
            cache=cache,
            expected_status="False",
            expected_reasons={"ImportCredentialError", "ImportFailed"},
            timeout=600,
        )

    ready = next(condition for condition in status["conditions"] if condition["type"] == "Ready")
    assert ready["message"], f"Expected actionable failure message, got {ready!r}"
