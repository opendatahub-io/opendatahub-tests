from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.kserve.model_cache.utils import (
    LocalModelNamespaceCache,
    assert_llmisvc_uses_cached_pvc,
    cache_status_dict,
)
from tests.model_serving.model_server.llmd.utils import (
    ns_from_file,
    parse_completion_text,
    send_chat_completions,
    workaround_503_no_healthy_upstream,
)

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.llmd_cpu,
    pytest.mark.usefixtures("valid_aws_config", "skip_if_disconnected"),
]

NAMESPACE = ns_from_file(file=__file__)


@pytest.mark.parametrize("unprivileged_model_namespace", [{"name": NAMESPACE}], indirect=True)
class TestLLMDModelCacheSmoke:
    """Smoke coverage for KServe local model namespace cache with ``LLMInferenceService`` workloads.

    Mirrors ``TestModelCacheSmoke`` in ``kserve/model_cache/test_local_model_cache.py`` (TC-04/TC-05),
    proving that local model caching — already covered for ``InferenceService`` — also works for the
    newer ``LLMInferenceService`` CRD, since both are watched and reconciled by the same
    ``LocalModelNamespaceCache`` controller.
    """

    @pytest.mark.slow
    def test_llmd_local_model_cache_reaches_node_downloaded(
        self,
        unprivileged_model_namespace: Any,
        tinyllama_local_model_cache: LocalModelNamespaceCache,
    ) -> None:
        """Test steps:

        1. Retrieve the LocalModelNamespaceCache status for the TinyLlama model.
        2. Assert every node in ``status.nodeStatus`` reports ``NodeDownloaded``.
        3. Assert ``copies.failed`` is 0 and ``copies.available`` equals ``copies.total``.
        """
        status = cache_status_dict(cache=tinyllama_local_model_cache)
        node_status = status.get("nodeStatus") or {}
        assert node_status, "status.nodeStatus must list at least one node"

        for node_name, state in node_status.items():
            assert state == "NodeDownloaded", f"node {node_name} expected NodeDownloaded, got {state!r}"

        copies = status.get("copies") or {}
        assert copies.get("failed", 0) == 0
        assert copies.get("available") == copies.get("total")
        assert (copies.get("available") or 0) >= 1

    @pytest.mark.slow
    def test_cached_llmisvc_inference_succeeds(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Any,
        tinyllama_local_model_cache: LocalModelNamespaceCache,
        tinyllama_llmisvc_local_model_cache: LLMInferenceService,
    ) -> None:
        """Test steps:

        1. Assert the LLMISVC spec was rewritten to use a PVC-backed volume (cache hit).
        2. Warm up the inference endpoint (workaround for RHOAIENG-55154).
        3. Send a chat completion request and assert the response status is 200.
        4. Assert the completion text is non-empty.
        5. Refresh the LocalModelNamespaceCache and assert the LLMISVC is listed
           under ``status.llmInferenceServices``.
        """
        llmisvc = tinyllama_llmisvc_local_model_cache
        prompt = "What is the capital of Italy?"

        assert_llmisvc_uses_cached_pvc(client=unprivileged_client, llmisvc=llmisvc)

        workaround_503_no_healthy_upstream(llmisvc=llmisvc, prompt=prompt)

        status, body = send_chat_completions(llmisvc=llmisvc, prompt=prompt)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert completion, f"Expected non-empty completion text, got: {body}"

        tinyllama_local_model_cache.get()
        status_dict = cache_status_dict(cache=tinyllama_local_model_cache)
        bound = [
            ref
            for ref in (status_dict.get("llmInferenceServices") or [])
            if ref.get("namespace") == llmisvc.namespace and ref.get("name") == llmisvc.name
        ]
        assert bound, (
            f"Expected LLMInferenceService {llmisvc.namespace}/{llmisvc.name} listed under "
            f"LocalModelNamespaceCache {tinyllama_local_model_cache.name} status.llmInferenceServices; "
            f"got {status_dict.get('llmInferenceServices')!r}"
        )
