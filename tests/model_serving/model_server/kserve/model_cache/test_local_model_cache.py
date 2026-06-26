from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LocalModelNamespaceCache,
    assert_predictor_uses_cached_pvc,
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
    """Smoke coverage for KServe local model namespace cache (TC-04, TC-05)."""

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [
            pytest.param(
                {"name": "kserve-model-cache-smoke"},
                RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
                id="test_namespace_cache_node_downloaded",
            )
        ],
        indirect=True,
    )
    def test_local_model_cache_reaches_node_downloaded(
        self,
        unprivileged_model_namespace: Any,
        ovms_kserve_serving_runtime: Any,
        mnist_local_model_cache: LocalModelNamespaceCache,
    ) -> None:
        """Given a provisioned LocalModelNamespaceCache, when status is refreshed,
        then all nodes in the node group are NodeDownloaded and copies are healthy.
        """
        status = cache_status_dict(cache=mnist_local_model_cache)
        node_status = status.get("nodeStatus") or {}
        assert node_status, "status.nodeStatus must list at least one node"

        for node_name, state in node_status.items():
            assert state == "NodeDownloaded", f"node {node_name} expected NodeDownloaded, got {state!r}"

        copies = status.get("copies") or {}
        assert copies.get("failed", 0) == 0
        assert copies.get("available") == copies.get("total")
        assert (copies.get("available") or 0) >= 1

    @pytest.mark.parametrize(
        "unprivileged_model_namespace, ovms_kserve_serving_runtime",
        [
            pytest.param(
                {"name": "kserve-model-cache-smoke"},
                RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
                id="test_cached_model_inference",
            )
        ],
        indirect=True,
    )
    def test_cached_model_inference_succeeds(
        self,
        unprivileged_client: DynamicClient,
        mnist_local_model_cache: LocalModelNamespaceCache,
        mnist_onnx_local_model_cache_inference_service: InferenceService,
        ovms_kserve_serving_runtime: ServingRuntime,
    ) -> None:
        """Given an ISVC whose storageUri matches a cached model, when inference runs
        over HTTPS, then PVC rewrite is present and response succeeds.
        """
        isvc = mnist_onnx_local_model_cache_inference_service
        assert_predictor_uses_cached_pvc(
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
            f"LocalModelNamespaceCache {mnist_local_model_cache.name} status.inferenceServices; "
            f"got {status.get('inferenceServices')!r}"
        )
