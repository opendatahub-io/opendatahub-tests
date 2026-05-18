from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LocalModelCache,
    assert_predictor_storage_initializer_uses_pvc,
    cache_status_dict,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Protocols, RunTimeConfigs
from utilities.inference_utils import Inference
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.rawdeployment,
    pytest.mark.usefixtures("valid_aws_config", "skip_if_disconnected"),
]


class TestModelCacheSmoke:
    """Smoke coverage for KServe local model cache (TC-04, TC-05)."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [pytest.param({"name": "kserve-model-cache-smoke"}, RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG)],
        indirect=True,
    )
    def test_local_model_cache_reaches_node_downloaded(
        self,
        unprivileged_model_namespace: Any,
        ovms_kserve_serving_runtime: Any,
        mnist_local_model_cache: LocalModelCache,
    ) -> None:
        """TC-04: `LocalModelCache` reports `NodeDownloaded` and healthy `copies` for all nodes.

        After the shared fixture waits for downloads, re-read status and assert fields explicitly.
        """
        status = cache_status_dict(cache=mnist_local_model_cache)
        node_status = status.get("nodeStatus") or {}
        assert node_status, "status.nodeStatus must list at least one node"

        for node_name, state in node_status.items():
            assert state == "NodeDownloaded", f"node {node_name} expected NodeDownloaded, got {state!r}"

        copies = status.get("copies") or {}
        assert copies.get("failed") == 0
        assert copies.get("available") == copies.get("total")
        assert (copies.get("available") or 0) >= 1

    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [pytest.param({"name": "kserve-model-cache-smoke"}, RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG)],
        indirect=True,
    )
    def test_inference_service_with_local_model_label_succeeds(
        self,
        unprivileged_client: DynamicClient,
        mnist_local_model_cache: LocalModelCache,
        mnist_onnx_local_model_cache_inference_service: InferenceService,
        ovms_kserve_serving_runtime: ServingRuntime,
    ) -> None:
        """TC-05: predictor uses PVC-backed storage rewrite and MNIST ONNX inference returns HTTP 200."""
        isvc = mnist_onnx_local_model_cache_inference_service
        assert_predictor_storage_initializer_uses_pvc(
            client=unprivileged_client,
            isvc=isvc,
            runtime_name=ovms_kserve_serving_runtime.name,
        )

        verify_inference_response(
            inference_service=isvc,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

        mnist_local_model_cache.get()
        status = cache_status_dict(cache=mnist_local_model_cache)
        bound = [
            ref
            for ref in (status.get("inferenceServices") or [])
            if ref.get("namespace") == isvc.namespace and ref.get("name") == isvc.name
        ]
        assert bound, (
            f"Expected InferenceService {isvc.namespace}/{isvc.name} listed under "
            f"LocalModelCache {mnist_local_model_cache.name} status.inferenceServices; "
            f"got {status.get('inferenceServices')!r}"
        )
