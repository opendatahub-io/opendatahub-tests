import pytest

from tests.model_serving.model_server.llmd.utils import (
    verify_gateway_status,
    verify_llm_service_status,
    verify_llmd_no_failed_pods,
)
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.deepseek_coder_v2_lite import DEEPSEEK_CODER_V2_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.llmd_gpu,
    pytest.mark.gpu,
    pytest.mark.model_server_gpu,
]


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param({"name": "llmd-multinode-test"})],
    indirect=True,
)
class TestSingleNodePrefillDecode:
    """Multi Node LLMISVC test cases."""

    def test_prefill_decode(self, unprivileged_client, llmd_gateway, llmisvc_singlenode_prefill_decode):
        """Test multi node llmisvc with DP + EP."""

        llmisvc = llmisvc_singlenode_prefill_decode

        assert verify_gateway_status(llmd_gateway), "Gateway should be ready"
        assert verify_llm_service_status(llmisvc), "LLMInferenceService should be ready"

        verify_inference_response_llmd(
            llm_service=llmisvc,
            inference_config=DEEPSEEK_CODER_V2_INFERENCE_CONFIG,
            inference_type="completions",
            protocol=Protocols.HTTPS,
            use_default_query=True,
            insecure=True,
            model_name=llmisvc.name,
        )

        verify_llmd_no_failed_pods(client=unprivileged_client, llm_service=llmisvc)
