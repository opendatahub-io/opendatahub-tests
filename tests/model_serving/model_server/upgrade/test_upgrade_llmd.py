import pytest
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.utils import (
    parse_completion_text,
    send_chat_completions,
)
from tests.model_serving.model_server.upgrade.utils import (
    verify_gateway_accepted,
    verify_llmd_pods_not_restarted,
    verify_llmd_router_not_restarted,
)

pytestmark = [pytest.mark.llmd_cpu]

PROMPT = "What is the capital of Italy?"
EXPECTED_ANSWER = "rome"


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

        1. Send a chat completion request to /v1/chat/completions.
        2. Assert the response status is 200.
        3. Assert the completion text contains the expected answer.
        """
        status, body = send_chat_completions(llmisvc=llmd_inference_service_fixture, prompt=PROMPT)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert EXPECTED_ANSWER in completion.lower(), f"Expected '{EXPECTED_ANSWER}' in response, got: {completion}"


class TestLlmdPostUpgrade:
    """Post-upgrade: verify LLMD deployment survived the platform upgrade."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="llmd_llmisvc_exists")
    def test_llmd_llmisvc_exists(self, llmd_inference_service_fixture: LLMInferenceService):
        """Test steps:

        1. Verify LLMInferenceService resource still exists after upgrade.
        """
        assert llmd_inference_service_fixture.exists, (
            f"LLMInferenceService {llmd_inference_service_fixture.name} does not exist after upgrade"
        )

    @pytest.mark.post_upgrade
    def test_llmd_gateway_exists(self, llmd_gateway_fixture: Gateway):
        """Test steps:

        1. Verify the LLMD Gateway resource exists.
        2. Verify the Gateway has an Accepted condition set to True.
        """
        verify_gateway_accepted(gateway=llmd_gateway_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_workload_pods_not_restarted(
        self,
        admin_client,
        llmd_inference_service_fixture: LLMInferenceService,
    ):
        """Test steps:

        1. Get all workload pods for the LLMInferenceService.
        2. Verify no container has restarted during the upgrade.
        """
        verify_llmd_pods_not_restarted(
            client=admin_client,
            llmisvc=llmd_inference_service_fixture,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_router_scheduler_not_restarted(
        self,
        admin_client,
        llmd_inference_service_fixture: LLMInferenceService,
    ):
        """Test steps:

        1. Get the router-scheduler pod for the LLMInferenceService.
        2. Verify no container has restarted during the upgrade.
        """
        verify_llmd_router_not_restarted(
            client=admin_client,
            llmisvc=llmd_inference_service_fixture,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_inference_post_upgrade(self, llmd_inference_service_fixture: LLMInferenceService):
        """Test steps:

        1. Send a chat completion request to /v1/chat/completions.
        2. Assert the response status is 200.
        3. Assert the completion text contains the expected answer.
        """
        status, body = send_chat_completions(llmisvc=llmd_inference_service_fixture, prompt=PROMPT)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert EXPECTED_ANSWER in completion.lower(), f"Expected '{EXPECTED_ANSWER}' in response, got: {completion}"
