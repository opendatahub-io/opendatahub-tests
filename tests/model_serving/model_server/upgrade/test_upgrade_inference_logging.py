import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.upgrade.utils import (
    ISVCBaseline,
    get_isvc_baseline,
    verify_inference_generation,
    verify_isvc_pods_not_restarted_against_baseline,
)
from tests.model_serving.model_server.utils import (
    verify_inference_response,
    wait_for_isvc_agent_sidecar,
)
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.usefixtures("capture_logging_upgrade_baseline")
class TestPreUpgradeInferenceLogging:
    """Validate a logging-enabled raw deployment ISVC before an operator upgrade.

    Steps:
        1. Deploy a raw-deployment ISVC with inference logging enabled
           (spec.predictor.logger, the same mechanism TrustyAI uses).
        2. Verify the kserve-agent sidecar is injected into the predictor pod.
        3. Send an inference request via the internal HTTP route and verify a
           successful response.
        4. Capture baseline values (generation, restart counts) to ConfigMap.
    """

    @pytest.mark.pre_upgrade
    def test_pre_upgrade_logging_isvc_agent_sidecar(
        self,
        admin_client: DynamicClient,
        logging_inference_service_fixture: InferenceService,
    ) -> None:
        """Verify the kserve-agent sidecar runs in the predictor pod before upgrade"""
        wait_for_isvc_agent_sidecar(client=admin_client, isvc=logging_inference_service_fixture)

    @pytest.mark.pre_upgrade
    def test_pre_upgrade_logging_isvc_inference(
        self,
        logging_inference_service_fixture: InferenceService,
    ) -> None:
        """Test inference on the logging-enabled ISVC before upgrade"""
        verify_inference_response(
            inference_service=logging_inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )


class TestPostUpgradeInferenceLogging:
    """Validate a logging-enabled raw deployment ISVC survives an operator upgrade without a rollout.

    The upgrade ships a new kserve-agent configuration (image/resources in the
    inferenceservice-config ConfigMap). The agent sidecar is injected into pods
    at creation time by the pod mutator webhook and is not part of the Deployment
    pod template, so existing model pods must not be restarted (RHOAIENG-60588).

    Steps:
        1. Verify the ISVC still exists after the upgrade.
        2. Verify the ISVC was not modified during the upgrade.
        3. Verify pods have not restarted beyond the pre-upgrade baseline.
        4. Verify the kserve-agent sidecar is still present in the predictor pod.
        5. Send an inference request via the internal HTTP route and verify a
           successful response.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="logging_isvc_exists")
    def test_post_upgrade_logging_isvc_exists(
        self,
        logging_inference_service_fixture: InferenceService,
    ) -> None:
        """Test that the logging-enabled ISVC exists after upgrade"""
        assert logging_inference_service_fixture.exists

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["logging_isvc_exists"])
    def test_post_upgrade_logging_isvc_not_modified(
        self,
        logging_inference_service_fixture: InferenceService,
        upgrade_baseline_fixture: dict[str, dict[str, ISVCBaseline]],
    ) -> None:
        """Test that the logging-enabled ISVC is not modified in upgrade"""
        baseline = get_isvc_baseline(
            baselines=upgrade_baseline_fixture,
            isvc_name=logging_inference_service_fixture.name,
        )
        verify_inference_generation(
            isvc=logging_inference_service_fixture,
            expected_generation=baseline["isvc_observed_generation"],
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["logging_isvc_exists"])
    def test_post_upgrade_logging_pods_not_restarted(
        self,
        admin_client: DynamicClient,
        logging_inference_service_fixture: InferenceService,
        upgrade_baseline_fixture: dict[str, dict[str, ISVCBaseline]],
    ) -> None:
        """Verify pods of the logging-enabled ISVC have not restarted beyond pre-upgrade baseline"""
        baseline = get_isvc_baseline(
            baselines=upgrade_baseline_fixture,
            isvc_name=logging_inference_service_fixture.name,
        )
        verify_isvc_pods_not_restarted_against_baseline(
            client=admin_client,
            isvc=logging_inference_service_fixture,
            baseline_restart_counts=baseline["pod_restart_counts"],
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["logging_isvc_exists"])
    def test_post_upgrade_logging_agent_sidecar_present(
        self,
        admin_client: DynamicClient,
        logging_inference_service_fixture: InferenceService,
    ) -> None:
        """Verify the kserve-agent sidecar is still present in the predictor pod after upgrade"""
        wait_for_isvc_agent_sidecar(client=admin_client, isvc=logging_inference_service_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["logging_isvc_exists"])
    def test_post_upgrade_logging_isvc_inference(
        self,
        logging_inference_service_fixture: InferenceService,
    ) -> None:
        """Test inference on the logging-enabled ISVC after upgrade"""
        verify_inference_response(
            inference_service=logging_inference_service_fixture,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
