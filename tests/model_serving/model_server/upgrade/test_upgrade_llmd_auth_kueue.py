import pytest
import structlog
from ocp_resources.llm_inference_service import LLMInferenceService
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.llmd.utils import (
    parse_completion_text,
    send_chat_completions,
)
from tests.model_serving.model_server.upgrade.utils import (
    get_llmisvc_kueue_integration_stats,
    load_baseline_from_configmap,
    verify_llmisvc_config_refs_exist,
    verify_llmisvc_config_refs_unchanged,
    verify_llmisvc_container_images_unchanged,
    verify_llmisvc_exists,
    verify_llmisvc_generation_unchanged,
    verify_llmisvc_httproute_exists,
    verify_llmisvc_inference_pool_exists,
    verify_llmisvc_model_uri_unchanged,
    verify_llmisvc_replicas_unchanged,
    verify_llmisvc_restart_counts_unchanged,
    verify_llmisvc_url_unchanged,
)
from utilities.kueue_utils import check_gated_pods_and_running_pods

pytestmark = [pytest.mark.llmd_cpu]

LOGGER = structlog.get_logger(name=__name__)

PROMPT = "What is the capital of Italy?"
EXPECTED_ANSWER = "rome"


class TestLlmdAuthKueuePreUpgrade:
    """Pre-upgrade: deploy auth+Kueue LLMISVC, validate inference, scale and gate."""

    @pytest.mark.pre_upgrade
    def test_llmisvc_auth_and_kueue_exists(self, llmisvc_upgrade_auth_and_kueue: LLMInferenceService):
        """Test steps:

        1. Verify auth-enabled LLMInferenceService resource exists on the cluster.
        """
        LOGGER.info(
            event=f"[PRE-UPGRADE] Checking auth LLMISVC '{llmisvc_upgrade_auth_and_kueue.name}' "
            f"exists in namespace '{llmisvc_upgrade_auth_and_kueue.namespace}'"
        )
        assert llmisvc_upgrade_auth_and_kueue.exists, (
            f"Auth LLMISVC {llmisvc_upgrade_auth_and_kueue.name} does not exist"
        )
        LOGGER.info(event=f"[PRE-UPGRADE] PASS: Auth LLMISVC '{llmisvc_upgrade_auth_and_kueue.name}' is deployed")

    @pytest.mark.pre_upgrade
    def test_auth_inference(
        self,
        llmisvc_upgrade_auth_and_kueue: LLMInferenceService,
        llmisvc_upgrade_token: str,
    ):
        """Test steps:

        1. Send an authenticated chat completion request to /v1/chat/completions.
        2. Assert the response status is 200.
        3. Assert the completion text contains the expected answer.
        """
        LOGGER.info(event=f"[PRE-UPGRADE] Sending auth inference to '{llmisvc_upgrade_auth_and_kueue.name}'")
        status, body = send_chat_completions(
            llmisvc=llmisvc_upgrade_auth_and_kueue,
            prompt=PROMPT,
            token=str(llmisvc_upgrade_token),
        )
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert EXPECTED_ANSWER in completion.lower(), f"Expected '{EXPECTED_ANSWER}', got: {completion}"
        LOGGER.info(event=f"[PRE-UPGRADE] PASS: Auth inference to '{llmisvc_upgrade_auth_and_kueue.name}' succeeded")

    @pytest.mark.pre_upgrade
    def test_kueue_scale_and_gate(
        self,
        admin_client,
        llmisvc_upgrade_auth_and_kueue: LLMInferenceService,
    ):
        """Test steps:

        1. Scale auth LLMISVC to 2 replicas (exceeds Kueue quota).
        2. Wait for 1 running + 1 gated pod.
        """
        llmisvc = llmisvc_upgrade_auth_and_kueue
        LOGGER.info(event=f"[PRE-UPGRADE] Scaling '{llmisvc.name}' to 2 replicas to trigger Kueue gating")

        spec_dict = llmisvc.instance.to_dict()
        spec_dict["spec"]["replicas"] = 2
        llmisvc.update(resource_dict=spec_dict)

        selector_labels = [
            f"app.kubernetes.io/name={llmisvc.name}",
            "kserve.io/component=workload",
        ]
        try:
            for running_pods, gated_pods in TimeoutSampler(
                wait_timeout=120,
                sleep=5,
                func=lambda: check_gated_pods_and_running_pods(
                    labels=selector_labels,
                    namespace=llmisvc.namespace,
                    admin_client=admin_client,
                ),
            ):
                if running_pods == 1 and gated_pods == 1:
                    break
        except TimeoutExpiredError:
            assert False, (
                f"Timeout waiting for Kueue gating: expected 1 running + 1 gated, "
                f"got {running_pods} running + {gated_pods} gated"
            )

        LOGGER.info(event=f"[PRE-UPGRADE] PASS: Kueue gating active — 1 running, 1 gated for '{llmisvc.name}'")


class TestLlmdAuthKueuePostUpgrade:
    """Post-upgrade: verify auth+Kueue LLMISVC survived the upgrade."""

    @pytest.fixture(scope="class")
    def baseline(self, admin_client, llmisvc_upgrade_auth_and_kueue: LLMInferenceService) -> dict:
        """Load pre-upgrade baseline for the auth+kueue LLMISVC from the cluster ConfigMap."""
        baselines = load_baseline_from_configmap(
            client=admin_client, namespace=llmisvc_upgrade_auth_and_kueue.namespace
        )
        assert llmisvc_upgrade_auth_and_kueue.name in baselines, (
            f"LLMISVC '{llmisvc_upgrade_auth_and_kueue.name}' not in baseline. Available: {list(baselines.keys())}"
        )
        return baselines[llmisvc_upgrade_auth_and_kueue.name]

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="llmisvc_exists")
    def test_llmisvc_exists_post_upgrade(self, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline):
        verify_llmisvc_exists(llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_kueue_local_queue_exists_post_upgrade(self, llmisvc_upgrade_kueue_resources):
        """Test steps:

        1. Verify Kueue LocalQueue still exists after upgrade.
        """
        assert llmisvc_upgrade_kueue_resources.exists, (
            "Kueue LocalQueue not found after upgrade — Kueue resources did not survive"
        )
        LOGGER.info(
            event=f"[POST-UPGRADE] PASS: Kueue LocalQueue '{llmisvc_upgrade_kueue_resources.name}' survived upgrade"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_kueue_conditions_post_upgrade(
        self,
        admin_client,
        llmisvc_upgrade_auth_and_kueue: LLMInferenceService,
        baseline,
    ):
        """Test steps:

        1. Verify RouterReady condition is True.
        2. Verify PresetsCombined condition is True.
        3. If Kueue was gating pods pre-upgrade, verify MainWorkloadReady=False
           with reason MinimumReplicasUnavailable.
        """
        conditions = llmisvc_upgrade_auth_and_kueue.instance.status.conditions
        condition_map = {c.type: c for c in conditions}
        LOGGER.info(event=f"[POST-UPGRADE] Auth LLMISVC conditions: {[(c.type, c.status) for c in conditions]}")

        router_ready = condition_map.get("RouterReady")
        assert router_ready, "RouterReady condition missing from LLMISVC status"
        assert router_ready.status == "True", (
            f"RouterReady={router_ready.status} reason={getattr(router_ready, 'reason', 'N/A')} — "
            "router/scheduler control plane did not survive upgrade"
        )
        LOGGER.info(event="[POST-UPGRADE] PASS: RouterReady=True — control plane survived upgrade")

        presets = condition_map.get("PresetsCombined")
        assert presets, "PresetsCombined condition missing from LLMISVC status"
        assert presets.status == "True", (
            f"PresetsCombined={presets.status} reason={getattr(presets, 'reason', 'N/A')} — "
            "LLMInferenceServiceConfig refs were not combined correctly after upgrade"
        )
        LOGGER.info(event="[POST-UPGRADE] PASS: PresetsCombined=True — config refs resolved correctly")

        expected_stats = baseline.get("kueue_integration_stats", {})
        if expected_stats.get("gated") and expected_stats["gated"] > 0:
            main_workload = condition_map.get("MainWorkloadReady")
            assert main_workload, "MainWorkloadReady condition missing from LLMISVC status"
            main_reason = getattr(main_workload, "reason", "N/A")
            main_message = getattr(main_workload, "message", "N/A")
            assert main_workload.status == "False", (
                f"MainWorkloadReady={main_workload.status} — expected False because Kueue is gating "
                f"the second replica (replicas={baseline.get('replicas')}, quota allows only 1). "
                f"reason={main_reason}, message={main_message}"
            )
            assert main_reason == "MinimumReplicasUnavailable", (
                f"MainWorkloadReady=False but reason='{main_reason}' instead of "
                f"'MinimumReplicasUnavailable'. message='{main_message}'. "
                "The second replica should be gated by Kueue quota, not failing for another reason."
            )
            LOGGER.info(
                event=f"[POST-UPGRADE] PASS: MainWorkloadReady=False reason={main_reason} — "
                "Kueue is gating the second replica as expected"
            )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_kueue_integration_stats_unchanged_post_upgrade(
        self, admin_client, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline
    ):
        """Test steps:

        1. Get running and gated pod counts for the LLMInferenceService.
        2. Compare against the pre-upgrade baseline.
        3. Assert counts have not changed.
        """
        expected = baseline["kueue_integration_stats"]
        current = get_llmisvc_kueue_integration_stats(client=admin_client, llmisvc=llmisvc_upgrade_auth_and_kueue)
        LOGGER.info(event=f"[POST-UPGRADE] kueue_integration_stats: expected={expected}, current={current}")
        assert current == expected, f"kueue_integration_stats changed: {expected} -> {current}"

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_auth_inference_post_upgrade(
        self,
        llmisvc_upgrade_auth_and_kueue: LLMInferenceService,
        llmisvc_upgrade_token: str,
    ):
        """Test steps:

        1. Send an authenticated chat completion request using the pre-upgrade SA token.
        2. Assert the response status is 200.
        3. Assert the completion text contains the expected answer.
        """
        LOGGER.info(
            event=f"[POST-UPGRADE] Sending auth inference to '{llmisvc_upgrade_auth_and_kueue.name}' "
            "using pre-upgrade SA token"
        )
        status, body = send_chat_completions(
            llmisvc=llmisvc_upgrade_auth_and_kueue,
            prompt=PROMPT,
            token=str(llmisvc_upgrade_token),
        )
        assert status == 200, f"Expected 200, got {status}: {body}"
        completion = parse_completion_text(response_body=body)
        assert EXPECTED_ANSWER in completion.lower(), f"Expected '{EXPECTED_ANSWER}', got: {completion}"
        LOGGER.info(
            event=f"[POST-UPGRADE] PASS: Auth inference to '{llmisvc_upgrade_auth_and_kueue.name}' "
            "succeeded with pre-upgrade RBAC"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_auth_repeated_inference_post_upgrade(
        self,
        llmisvc_upgrade_auth_and_kueue: LLMInferenceService,
        llmisvc_upgrade_token: str,
    ):
        """Test steps:

        1. Send 10 sequential authenticated chat completion requests.
        2. Assert each response status is 200.
        3. Assert each completion text contains the expected answer.
        """
        total = 10
        LOGGER.info(event=f"[POST-UPGRADE] Sending {total} auth requests to '{llmisvc_upgrade_auth_and_kueue.name}'")
        for index in range(1, total + 1):
            status, body = send_chat_completions(
                llmisvc=llmisvc_upgrade_auth_and_kueue,
                prompt=PROMPT,
                token=str(llmisvc_upgrade_token),
            )
            assert status == 200, f"Request {index}/{total}: expected 200, got {status}: {body}"
            completion = parse_completion_text(response_body=body)
            assert EXPECTED_ANSWER in completion.lower(), (
                f"Request {index}/{total}: expected '{EXPECTED_ANSWER}', got: {completion}"
            )
            LOGGER.info(event=f"[POST-UPGRADE] Auth request {index}/{total}: OK")
        LOGGER.info(event=f"[POST-UPGRADE] PASS: All {total} auth inference requests succeeded")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_auth_unauthorized_post_upgrade(
        self,
        llmisvc_upgrade_auth_and_kueue: LLMInferenceService,
    ):
        """Test steps:

        1. Send inference without token to auth-enabled LLMISVC.
        2. Assert 401 or 403 — auth was not silently disabled during upgrade.
        """
        LOGGER.info(
            event=f"[POST-UPGRADE] Sending unauthenticated request to "
            f"'{llmisvc_upgrade_auth_and_kueue.name}' — expecting rejection"
        )
        status, body = send_chat_completions(llmisvc=llmisvc_upgrade_auth_and_kueue, prompt=PROMPT)
        LOGGER.info(event=f"[POST-UPGRADE] Unauthenticated response: status={status}")
        assert status in (401, 403), (
            f"Auth LLMISVC '{llmisvc_upgrade_auth_and_kueue.name}' should reject "
            f"unauthenticated requests, got {status}: {body}"
        )
        LOGGER.info(
            event=f"[POST-UPGRADE] PASS: Auth LLMISVC '{llmisvc_upgrade_auth_and_kueue.name}' "
            "correctly rejected unauthenticated request"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_generation_unchanged_post_upgrade(self, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline):
        verify_llmisvc_generation_unchanged(llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_url_unchanged_post_upgrade(self, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline):
        verify_llmisvc_url_unchanged(llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_replicas_unchanged_post_upgrade(self, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline):
        verify_llmisvc_replicas_unchanged(llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_model_uri_unchanged_post_upgrade(self, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline):
        verify_llmisvc_model_uri_unchanged(llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_container_images_unchanged_post_upgrade(
        self, admin_client, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline
    ):
        verify_llmisvc_container_images_unchanged(
            client=admin_client, llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_restart_counts_unchanged_post_upgrade(
        self, admin_client, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline
    ):
        verify_llmisvc_restart_counts_unchanged(
            client=admin_client, llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_config_refs_exist_post_upgrade(
        self, admin_client, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline
    ):
        verify_llmisvc_config_refs_exist(client=admin_client, llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_config_refs_unchanged_post_upgrade(self, llmisvc_upgrade_auth_and_kueue: LLMInferenceService, baseline):
        verify_llmisvc_config_refs_unchanged(llmisvc=llmisvc_upgrade_auth_and_kueue, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_inference_pool_exists_post_upgrade(
        self, admin_client, llmisvc_upgrade_auth_and_kueue: LLMInferenceService
    ):
        verify_llmisvc_inference_pool_exists(client=admin_client, llmisvc=llmisvc_upgrade_auth_and_kueue)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmisvc_exists"])
    def test_httproute_exists_post_upgrade(self, admin_client, llmisvc_upgrade_auth_and_kueue: LLMInferenceService):
        verify_llmisvc_httproute_exists(client=admin_client, llmisvc=llmisvc_upgrade_auth_and_kueue)
