import pytest
from ocp_resources.llm_inference_service import LLMInferenceService

from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG

pytestmark = [pytest.mark.llmd_cpu]


class TestLlmdPreUpgrade:
    """Pre-upgrade: deploy LLMD InferenceService and validate inference."""

    @pytest.mark.pre_upgrade
    def test_llmd_llmisvc_deployed(self, llmd_inference_service_fixture: LLMInferenceService):
        """Test steps:

        1. Verify LLMInferenceService resource exists on the cluster.
        """
        assert llmd_inference_service_fixture.exists, (
            f"LLMInferenceService {llmd_inference_service_fixture.name} does not exist"
        )

    @pytest.mark.pre_upgrade
    def test_llmd_inference_pre_upgrade(self, llmd_inference_service_fixture: LLMInferenceService):
        """Test steps:

        1. Send a chat completion request via verify_inference_response_llmd.
        2. Assert the response matches expected output.
        """
        verify_inference_response_llmd(
            llm_service=llmd_inference_service_fixture,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmd_inference_service_fixture.name,
        )
