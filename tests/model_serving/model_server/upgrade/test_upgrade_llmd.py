import pytest
import structlog
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.llmd.utils import (
    parse_completion_text,
    send_chat_completions,
)
from tests.model_serving.model_server.upgrade.utils import (
    load_baseline_from_configmap,
    verify_llmisvc_config_refs_exist,
    verify_llmisvc_config_refs_unchanged,
    verify_llmisvc_container_images_unchanged,
    verify_llmisvc_controller_healthy,
    verify_llmisvc_exists,
    verify_llmisvc_gateway,
    verify_llmisvc_generation_unchanged,
    verify_llmisvc_httproute_exists,
    verify_llmisvc_inference_pool_exists,
    verify_llmisvc_model_uri_unchanged,
    verify_llmisvc_replicas_unchanged,
    verify_llmisvc_restart_counts_unchanged,
    verify_llmisvc_url_unchanged,
)

pytestmark = [pytest.mark.llmd_cpu]

LOGGER = structlog.get_logger(name=__name__)

PROMPT = "What is the capital of Italy?"
EXPECTED_ANSWER = "rome"


class TestLlmdPreUpgrade:
    """Pre-upgrade:

    1. Deploy an LLMInferenceService with auth disabled
    2. Validate inference
    3. Capture baseline in ConfigMap to be used in post-upgrade
    """

    @pytest.mark.pre_upgrade
    def test_llmisvc_no_auth_exists(self, llmisvc_upgrade_no_auth: LLMInferenceService):
        """Test steps:

        1. Verify LLMInferenceService resource exists on the cluster.
        """
        LOGGER.info(
            event=f"[PRE-UPGRADE] Checking LLMInferenceService '{llmisvc_upgrade_no_auth.name}' "
            f"exists in namespace '{llmisvc_upgrade_no_auth.namespace}'"
        )
        assert llmisvc_upgrade_no_auth.exists, f"LLMInferenceService {llmisvc_upgrade_no_auth.name} does not exist"
        LOGGER.info(event=f"[PRE-UPGRADE] PASS: LLMInferenceService '{llmisvc_upgrade_no_auth.name}' is deployed")

    @pytest.mark.pre_upgrade
    def test_no_auth_inference(self, llmisvc_upgrade_no_auth: LLMInferenceService):
        """Test steps:

        1. Send a chat completion request to /v1/chat/completions.
        2. Assert the response status is 200.
        3. Assert the completion text contains the expected answer.
        """
        LOGGER.info(event=f"[PRE-UPGRADE] Sending inference to '{llmisvc_upgrade_no_auth.name}'")
        status, body = send_chat_completions(llmisvc=llmisvc_upgrade_no_auth, prompt=PROMPT)
        assert status == 200, f"Expected 200, got {status}: {body}"

        completion = parse_completion_text(response_body=body)
        assert EXPECTED_ANSWER in completion.lower(), f"Expected '{EXPECTED_ANSWER}', got: {completion}"
        LOGGER.info(event=f"[PRE-UPGRADE] PASS: Inference to '{llmisvc_upgrade_no_auth.name}' succeeded")


class TestLlmdPostUpgrade:
    """Post-upgrade: verify non-auth LLMISVC survived the platform upgrade."""

    @pytest.fixture(scope="class")
    def baseline(self, admin_client, llmisvc_upgrade_no_auth: LLMInferenceService) -> dict:
        """Load pre-upgrade baseline for the no-auth LLMISVC from the cluster ConfigMap."""
        baselines = load_baseline_from_configmap(client=admin_client, namespace=llmisvc_upgrade_no_auth.namespace)
        assert llmisvc_upgrade_no_auth.name in baselines, (
            f"LLMISVC '{llmisvc_upgrade_no_auth.name}' not in baseline. Available: {list(baselines.keys())}"
        )
        return baselines[llmisvc_upgrade_no_auth.name]

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="llmisvc_exists")
    def test_llmisvc_exists_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService, baseline):
        verify_llmisvc_exists(llmisvc=llmisvc_upgrade_no_auth, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_ready_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService):
        """Test steps:

        1. Get the Ready condition from the LLMInferenceService status.
        2. Assert Ready condition exists.
        3. Assert Ready condition is True.
        """
        conditions = llmisvc_upgrade_no_auth.instance.status.conditions
        ready = next((condition for condition in conditions if condition.type == "Ready"), None)
        assert ready, f"No Ready condition on '{llmisvc_upgrade_no_auth.name}'"
        assert ready.status == "True", (
            f"LLMInferenceService '{llmisvc_upgrade_no_auth.name}' is not Ready after upgrade: "
            f"reason={ready.reason}, message={ready.message}"
        )
        LOGGER.info(event=f"[POST-UPGRADE] PASS: '{llmisvc_upgrade_no_auth.name}' Ready=True")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_no_auth_inference_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService):
        """Test steps:

        1. Send a chat completion request to /v1/chat/completions.
        2. Assert the response status is 200.
        3. Assert the completion text contains the expected answer.
        """
        LOGGER.info(event=f"[POST-UPGRADE] Sending inference to '{llmisvc_upgrade_no_auth.name}'")
        status, body = send_chat_completions(llmisvc=llmisvc_upgrade_no_auth, prompt=PROMPT)
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert EXPECTED_ANSWER in completion.lower(), f"Expected '{EXPECTED_ANSWER}', got: {completion}"
        LOGGER.info(event=f"[POST-UPGRADE] PASS: Inference to '{llmisvc_upgrade_no_auth.name}' succeeded")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_no_auth_repeated_inference_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService):
        """Test steps:

        1. Send 10 sequential chat completion requests to /v1/chat/completions.
        2. Assert each response status is 200.
        3. Assert each completion text contains the expected answer.
        """
        total = 10
        LOGGER.info(event=f"[POST-UPGRADE] Sending {total} requests to '{llmisvc_upgrade_no_auth.name}'")
        for index in range(1, total + 1):
            status, body = send_chat_completions(llmisvc=llmisvc_upgrade_no_auth, prompt=PROMPT)
            assert status == 200, f"Request {index}/{total}: expected 200, got {status}: {body}"
            completion = parse_completion_text(response_body=body)
            assert EXPECTED_ANSWER in completion.lower(), (
                f"Request {index}/{total}: expected '{EXPECTED_ANSWER}', got: {completion}"
            )
            LOGGER.info(event=f"[POST-UPGRADE] Request {index}/{total}: OK")
        LOGGER.info(event=f"[POST-UPGRADE] PASS: All {total} inference requests succeeded")

    @pytest.mark.post_upgrade
    def test_gateway_post_upgrade(self, llmisvc_upgrade_gateway: Gateway):
        verify_llmisvc_gateway(gateway=llmisvc_upgrade_gateway)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_generation_unchanged_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService, baseline):
        verify_llmisvc_generation_unchanged(llmisvc=llmisvc_upgrade_no_auth, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_url_unchanged_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService, baseline):
        verify_llmisvc_url_unchanged(llmisvc=llmisvc_upgrade_no_auth, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_replicas_unchanged_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService, baseline):
        verify_llmisvc_replicas_unchanged(llmisvc=llmisvc_upgrade_no_auth, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_model_uri_unchanged_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService, baseline):
        verify_llmisvc_model_uri_unchanged(llmisvc=llmisvc_upgrade_no_auth, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_container_images_unchanged_post_upgrade(
        self, admin_client, llmisvc_upgrade_no_auth: LLMInferenceService, baseline
    ):
        verify_llmisvc_container_images_unchanged(
            client=admin_client, llmisvc=llmisvc_upgrade_no_auth, baseline=baseline
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_restart_counts_unchanged_post_upgrade(
        self, admin_client, llmisvc_upgrade_no_auth: LLMInferenceService, baseline
    ):
        verify_llmisvc_restart_counts_unchanged(client=admin_client, llmisvc=llmisvc_upgrade_no_auth, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_config_refs_exist_post_upgrade(self, admin_client, llmisvc_upgrade_no_auth: LLMInferenceService, baseline):
        verify_llmisvc_config_refs_exist(client=admin_client, llmisvc=llmisvc_upgrade_no_auth, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_config_refs_unchanged_post_upgrade(self, llmisvc_upgrade_no_auth: LLMInferenceService, baseline):
        verify_llmisvc_config_refs_unchanged(llmisvc=llmisvc_upgrade_no_auth, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_inference_pool_exists_post_upgrade(self, admin_client, llmisvc_upgrade_no_auth: LLMInferenceService):
        verify_llmisvc_inference_pool_exists(client=admin_client, llmisvc=llmisvc_upgrade_no_auth)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_httproute_exists_post_upgrade(self, admin_client, llmisvc_upgrade_no_auth: LLMInferenceService):
        verify_llmisvc_httproute_exists(client=admin_client, llmisvc=llmisvc_upgrade_no_auth)

    @pytest.mark.post_upgrade
    def test_controller_healthy_post_upgrade(self, admin_client):
        verify_llmisvc_controller_healthy(client=admin_client)
