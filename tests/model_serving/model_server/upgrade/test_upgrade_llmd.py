import logging

import pytest
from ocp_resources.gateway import Gateway
from ocp_resources.llm_inference_service import LLMInferenceService

from tests.model_serving.model_server.upgrade.utils import (
    LLMISVCBaseline,
    verify_gateway_accepted,
    verify_llmisvc_config_refs_exist,
    verify_llmisvc_container_images_unchanged,
    verify_llmisvc_generation_unchanged,
    verify_llmisvc_pods_not_restarted_against_baseline,
    verify_llmisvc_status_fields,
)
from utilities.constants import Protocols
from utilities.llmd_utils import verify_inference_response_llmd
from utilities.manifests.tinyllama import TINYLLAMA_INFERENCE_CONFIG

pytestmark = [pytest.mark.llmd_cpu]

logger = logging.getLogger(__name__)


class TestLlmdPreUpgrade:
    """Pre-upgrade: deploy LLMD InferenceService, capture baseline, and validate inference."""

    @pytest.mark.pre_upgrade
    def test_llmd_llmisvc_deployed(self, llmd_inference_service_fixture: LLMInferenceService):
        """Test steps:

        1. Verify LLMInferenceService resource exists on the cluster.
        """
        logger.info(
            f"[PRE-UPGRADE] Checking LLMInferenceService '{llmd_inference_service_fixture.name}' "
            f"exists in namespace '{llmd_inference_service_fixture.namespace}'"
        )
        assert llmd_inference_service_fixture.exists, (
            f"LLMInferenceService {llmd_inference_service_fixture.name} does not exist"
        )
        logger.info(f"[PRE-UPGRADE] PASS: LLMInferenceService '{llmd_inference_service_fixture.name}' is deployed")

    @pytest.mark.pre_upgrade
    def test_llmd_baseline_captured(self, capture_llmd_upgrade_baseline):
        """Test steps:

        1. Capture LLMISVC baseline (generation, status, images, config refs, restart counts).
        2. Persist baseline to a ConfigMap for post-upgrade comparison.
        """
        logger.info(msg="[PRE-UPGRADE] Baseline capture triggered via capture_llmd_upgrade_baseline fixture")
        logger.info(msg="[PRE-UPGRADE] Baseline was persisted to ConfigMap — check capture_llmisvc_baseline logs above")

    @pytest.mark.pre_upgrade
    def test_llmd_inference_pre_upgrade(self, llmd_inference_service_fixture: LLMInferenceService):
        """Test steps:

        1. Send a chat completion request via verify_inference_response_llmd.
        2. Assert the response matches expected output.
        """
        logger.info(
            f"[PRE-UPGRADE] Sending chat_completions inference to '{llmd_inference_service_fixture.name}' "
            f"with model_name='{llmd_inference_service_fixture.name}'"
        )
        verify_inference_response_llmd(
            llm_service=llmd_inference_service_fixture,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmd_inference_service_fixture.name,
        )
        logger.info(f"[PRE-UPGRADE] PASS: Inference to '{llmd_inference_service_fixture.name}' succeeded")


class TestLlmdPostUpgrade:
    """Post-upgrade: verify LLMD deployment survived the platform upgrade."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="llmd_llmisvc_exists")
    def test_llmd_llmisvc_exists(self, llmd_inference_service_fixture: LLMInferenceService):
        """Test steps:

        1. Verify LLMInferenceService resource still exists after upgrade.
        """
        logger.info(
            f"[POST-UPGRADE] Checking LLMInferenceService '{llmd_inference_service_fixture.name}' "
            f"still exists in namespace '{llmd_inference_service_fixture.namespace}'"
        )
        assert llmd_inference_service_fixture.exists, (
            f"LLMInferenceService {llmd_inference_service_fixture.name} does not exist after upgrade"
        )
        logger.info(
            f"[POST-UPGRADE] PASS: LLMInferenceService '{llmd_inference_service_fixture.name}' survived upgrade"
        )

    @pytest.mark.post_upgrade
    def test_llmd_gateway_exists(self, llmd_gateway_fixture: Gateway):
        """Test steps:

        1. Verify the LLMD Gateway resource exists.
        2. Verify the Gateway has an Accepted condition set to True.
        """
        logger.info(
            f"[POST-UPGRADE] Checking Gateway '{llmd_gateway_fixture.name}' "
            f"in namespace '{llmd_gateway_fixture.namespace}'"
        )
        verify_gateway_accepted(gateway=llmd_gateway_fixture)
        logger.info(f"[POST-UPGRADE] PASS: Gateway '{llmd_gateway_fixture.name}' is Accepted")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_generation_unchanged(
        self,
        llmd_inference_service_fixture: LLMInferenceService,
        llmd_upgrade_baseline_fixture: dict[str, LLMISVCBaseline],
    ):
        """Test steps:

        1. Compare metadata.generation against pre-upgrade baseline.
        2. Fail if spec was mutated by the upgrade process.
        """
        baseline = llmd_upgrade_baseline_fixture[llmd_inference_service_fixture.name]
        logger.info(
            f"[POST-UPGRADE] Comparing generation for '{llmd_inference_service_fixture.name}': "
            f"baseline={baseline['spec_generation']}, "
            f"current={llmd_inference_service_fixture.instance.metadata.generation}"
        )
        verify_llmisvc_generation_unchanged(llmisvc=llmd_inference_service_fixture, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_status_fields_intact(
        self,
        llmd_inference_service_fixture: LLMInferenceService,
        llmd_upgrade_baseline_fixture: dict[str, LLMISVCBaseline],
    ):
        """Test steps:

        1. Verify Ready condition is True after upgrade.
        2. Verify URL has not changed.
        3. Verify replica count matches pre-upgrade baseline.
        """
        baseline = llmd_upgrade_baseline_fixture[llmd_inference_service_fixture.name]
        logger.info(
            f"[POST-UPGRADE] Checking status fields for '{llmd_inference_service_fixture.name}': "
            f"baseline_url='{baseline['url']}', baseline_replicas={baseline['replicas']}"
        )
        verify_llmisvc_status_fields(llmisvc=llmd_inference_service_fixture, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_container_images_unchanged(
        self,
        admin_client,
        llmd_inference_service_fixture: LLMInferenceService,
        llmd_upgrade_baseline_fixture: dict[str, LLMISVCBaseline],
    ):
        """Test steps:

        1. Compare container images on all workload and router pods against baseline.
        2. Fail if any image was changed by the upgrade.
        """
        baseline = llmd_upgrade_baseline_fixture[llmd_inference_service_fixture.name]
        baseline_images = baseline["container_images"]
        logger.info(
            f"[POST-UPGRADE] Comparing container images for '{llmd_inference_service_fixture.name}': "
            f"baseline has {len(baseline_images)} pod(s): {list(baseline_images.keys())}"
        )
        for pod_name, containers in baseline_images.items():
            for cname, cimage in containers.items():
                logger.info(f"[POST-UPGRADE]   baseline pod={pod_name} container={cname} image={cimage}")
        verify_llmisvc_container_images_unchanged(
            client=admin_client, llmisvc=llmd_inference_service_fixture, baseline=baseline
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_config_refs_survive_upgrade(
        self,
        admin_client,
        llmd_inference_service_fixture: LLMInferenceService,
        llmd_upgrade_baseline_fixture: dict[str, LLMISVCBaseline],
    ):
        """Test steps:

        1. Verify LLMInferenceServiceConfig CRs referenced pre-upgrade still exist.
        2. Catches regressions like RHOAIENG-65791.
        """
        baseline = llmd_upgrade_baseline_fixture[llmd_inference_service_fixture.name]
        config_refs = baseline["config_ref_names"]
        logger.info(
            f"[POST-UPGRADE] Checking config refs for '{llmd_inference_service_fixture.name}': "
            f"baseline captured {len(config_refs)} config ref(s): {config_refs}"
        )
        if not config_refs:
            pytest.skip(
                reason="No config refs in baseline — nothing to verify. "
                "No status annotations with prefix 'serving.kserve.io/config-llm-' were found pre-upgrade."
            )
        verify_llmisvc_config_refs_exist(client=admin_client, llmisvc=llmd_inference_service_fixture, baseline=baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_workload_pods_not_restarted(
        self,
        admin_client,
        llmd_inference_service_fixture: LLMInferenceService,
        llmd_upgrade_baseline_fixture: dict[str, LLMISVCBaseline],
    ):
        """Test steps:

        1. Compare restart counts against pre-upgrade baseline.
        2. Fail if any container restarted during the upgrade.
        """
        baseline = llmd_upgrade_baseline_fixture[llmd_inference_service_fixture.name]
        restart_counts = baseline["restart_counts"]
        logger.info(
            f"[POST-UPGRADE] Checking pod restart counts for '{llmd_inference_service_fixture.name}': "
            f"baseline has {len(restart_counts)} pod(s): {list(restart_counts.keys())}"
        )
        for pod_name, containers in restart_counts.items():
            for cname, count in containers.items():
                logger.info(f"[POST-UPGRADE]   baseline pod={pod_name} container={cname} restartCount={count}")
        verify_llmisvc_pods_not_restarted_against_baseline(
            client=admin_client, llmisvc=llmd_inference_service_fixture, baseline=baseline
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["llmd_llmisvc_exists"])
    def test_llmd_inference_post_upgrade(self, llmd_inference_service_fixture: LLMInferenceService):
        """Test steps:

        1. Send a chat completion request via verify_inference_response_llmd.
        2. Assert the response matches expected output.
        """
        logger.info(
            f"[POST-UPGRADE] Sending chat_completions inference to '{llmd_inference_service_fixture.name}' "
            f"with model_name='{llmd_inference_service_fixture.name}'"
        )
        verify_inference_response_llmd(
            llm_service=llmd_inference_service_fixture,
            inference_config=TINYLLAMA_INFERENCE_CONFIG,
            inference_type="chat_completions",
            protocol=Protocols.HTTP,
            use_default_query=True,
            insecure=True,
            model_name=llmd_inference_service_fixture.name,
        )
        logger.info(f"[POST-UPGRADE] PASS: Inference to '{llmd_inference_service_fixture.name}' succeeded")
