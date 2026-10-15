"""Pre/post upgrade tests for raw-deployment InferenceService with Kueue admission control."""

import pytest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.serving_runtime import ServingRuntime
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.upgrade.kserve_kueue_upgrade_config import (
    KSERVE_KUEUE_EXPECTED_RUNNING_PODS,
    KSERVE_KUEUE_INFERENCE_TIMEOUT,
    KSERVE_KUEUE_MIN_REPLICAS,
)
from tests.model_serving.model_server.upgrade.utils import (
    ISVCKueueBaseline,
    get_isvc_kueue_integration_stats,
    load_baseline_from_configmap,
    read_isvc_total_copies,
    verify_inference_generation,
    verify_isvc_pods_not_restarted_against_baseline,
    verify_kserve_kueue_upgrade_inference,
)
from utilities.constants import Timeout
from utilities.general import create_isvc_label_selector_str
from utilities.inference_utils import Inference
from utilities.kueue_utils import check_gated_pods_and_running_pods
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

pytestmark = [
    pytest.mark.rawdeployment,
    pytest.mark.kueue,
    pytest.mark.usefixtures("valid_aws_config", "skip_if_not_raw_deployment"),
]

LOGGER = get_logger(name=__name__)

EXPECTED_DEPLOYMENTS = 1


def _get_isvc_deployments(
    admin_client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str,
) -> list[Deployment]:
    """Return Deployments for an InferenceService and ServingRuntime."""
    deployment_labels = [
        create_isvc_label_selector_str(
            isvc=isvc,
            resource_type="deployment",
            runtime_name=runtime_name,
        )
    ]
    return list(
        Deployment.get(
            label_selector=",".join(deployment_labels),
            namespace=isvc.namespace,
            client=admin_client,
        )
    )


class TestKserveKueueRawPreUpgrade:
    """Pre-upgrade: deploy raw ISVC with Kueue and verify initial state."""

    @pytest.mark.pre_upgrade
    def test_isvc_exists(
        self,
        kserve_kueue_upgrade_inference_service: InferenceService,
    ) -> None:
        """Test steps:

        1. Verify the Kueue-integrated raw InferenceService exists on the cluster.
        """
        isvc = kserve_kueue_upgrade_inference_service
        LOGGER.info(f"[PRE-UPGRADE] Checking ISVC '{isvc.name}' exists in namespace '{isvc.namespace}'")
        assert isvc.exists, f"InferenceService {isvc.name} does not exist"
        LOGGER.info(f"[PRE-UPGRADE] PASS: ISVC '{isvc.name}' is deployed")

    @pytest.mark.pre_upgrade
    def test_initial_deployment_ready(
        self,
        admin_client: DynamicClient,
        kserve_kueue_upgrade_inference_service: InferenceService,
        kserve_kueue_upgrade_serving_runtime: ServingRuntime,
    ) -> None:
        """Test steps:

        1. Locate the ISVC Deployment.
        2. Wait for the initial replica to be deployed.
        """
        isvc = kserve_kueue_upgrade_inference_service
        deployments = _get_isvc_deployments(
            admin_client=admin_client,
            isvc=isvc,
            runtime_name=kserve_kueue_upgrade_serving_runtime.name,
        )
        assert len(deployments) == EXPECTED_DEPLOYMENTS, (
            f"Expected {EXPECTED_DEPLOYMENTS} deployment, got {len(deployments)}"
        )

        deployment = deployments[0]
        deployment.wait_for_replicas(deployed=True)
        replicas = deployment.instance.spec.replicas
        assert replicas == KSERVE_KUEUE_MIN_REPLICAS, (
            f"Deployment should have {KSERVE_KUEUE_MIN_REPLICAS} replica, got {replicas}"
        )
        LOGGER.info(f"[PRE-UPGRADE] PASS: Deployment '{deployment.name}' has {replicas} replica")


class TestKserveKueueRawPostUpgrade:
    """Post-upgrade: verify raw ISVC and Kueue resources survived the upgrade."""

    @pytest.fixture(scope="class")
    def baseline(
        self,
        admin_client: DynamicClient,
        kserve_kueue_upgrade_inference_service: InferenceService,
    ) -> ISVCKueueBaseline:
        """Load pre-upgrade baseline for the Kueue ISVC from the cluster ConfigMap."""
        baselines = load_baseline_from_configmap(
            client=admin_client,
            namespace=kserve_kueue_upgrade_inference_service.namespace,
        )
        isvc_name = kserve_kueue_upgrade_inference_service.name
        assert isvc_name in baselines, f"ISVC '{isvc_name}' not in baseline. Available: {list(baselines.keys())}"
        return baselines[isvc_name]

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="kserve_kueue_isvc_exists")
    def test_isvc_exists_post_upgrade(
        self,
        kserve_kueue_upgrade_inference_service: InferenceService,
    ) -> None:
        """Test steps:

        1. Verify the InferenceService still exists after upgrade.
        """
        isvc = kserve_kueue_upgrade_inference_service
        assert isvc.exists, f"InferenceService '{isvc.name}' not found after upgrade"
        LOGGER.info(f"[POST-UPGRADE] PASS: ISVC '{isvc.name}' exists")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kserve_kueue_isvc_exists"])
    def test_kueue_local_queue_exists_post_upgrade(
        self,
        kserve_upgrade_kueue_resources,
    ) -> None:
        """Test steps:

        1. Verify Kueue LocalQueue still exists after upgrade.
        """
        assert kserve_upgrade_kueue_resources.exists, (
            "Kueue LocalQueue not found after upgrade — Kueue resources did not survive"
        )
        LOGGER.info(f"[POST-UPGRADE] PASS: LocalQueue '{kserve_upgrade_kueue_resources.name}' survived upgrade")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kserve_kueue_isvc_exists"])
    def test_kueue_integration_stats_unchanged_post_upgrade(
        self,
        admin_client: DynamicClient,
        kserve_kueue_upgrade_inference_service: InferenceService,
        kserve_kueue_upgrade_serving_runtime: ServingRuntime,
        baseline: dict,
    ) -> None:
        """Test steps:

        1. Compare running/gated pod counts against the pre-upgrade baseline.
        2. Assert Kueue gating state is unchanged.
        """
        expected = baseline["kueue_integration_stats"]
        current = get_isvc_kueue_integration_stats(
            client=admin_client,
            isvc=kserve_kueue_upgrade_inference_service,
            runtime_name=kserve_kueue_upgrade_serving_runtime.name,
        )
        LOGGER.info(f"[POST-UPGRADE] kueue_integration_stats: expected={expected}, current={current}")
        assert current == expected, f"kueue_integration_stats changed: {expected} -> {current}"

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kserve_kueue_isvc_exists"])
    def test_total_copies_unchanged_post_upgrade(
        self,
        kserve_kueue_upgrade_inference_service: InferenceService,
        baseline: dict,
    ) -> None:
        """Test steps:

        1. Verify ISVC status.totalCopies matches the pre-upgrade baseline.
        """
        isvc = kserve_kueue_upgrade_inference_service
        total_copies = read_isvc_total_copies(isvc=isvc)
        expected = baseline["total_copies"]
        assert total_copies == expected, f"totalCopies changed after upgrade: expected {expected}, got {total_copies}"
        LOGGER.info(f"[POST-UPGRADE] PASS: totalCopies={total_copies} unchanged")

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kserve_kueue_isvc_exists"])
    def test_generation_unchanged_post_upgrade(
        self,
        kserve_kueue_upgrade_inference_service: InferenceService,
        baseline: dict,
    ) -> None:
        """Test steps:

        1. Verify ISVC observedGeneration matches the pre-upgrade baseline.
        """
        verify_inference_generation(
            isvc=kserve_kueue_upgrade_inference_service,
            expected_generation=baseline["isvc_observed_generation"],
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kserve_kueue_isvc_exists"])
    def test_pods_not_restarted_post_upgrade(
        self,
        admin_client: DynamicClient,
        kserve_kueue_upgrade_inference_service: InferenceService,
        baseline: dict,
    ) -> None:
        """Test steps:

        1. Verify baseline running pods still exist and container restart counts did not increase.
        2. Allow additional pods when Kueue admits previously gated workloads after upgrade.
        """
        verify_isvc_pods_not_restarted_against_baseline(
            client=admin_client,
            isvc=kserve_kueue_upgrade_inference_service,
            baseline_restart_counts=baseline["pod_restart_counts"],
            allow_new_pods=True,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["kserve_kueue_isvc_exists"])
    def test_inference_post_upgrade(
        self,
        admin_client: DynamicClient,
        kserve_kueue_upgrade_inference_service: InferenceService,
        kserve_kueue_upgrade_serving_runtime: ServingRuntime,
    ) -> None:
        """Test steps:

        1. Wait for at least one Running predictor pod (Kueue may gate additional replicas).
        2. Send an inference request to the ISVC after upgrade.
        3. Verify the model still serves predictions.
        """
        isvc = kserve_kueue_upgrade_inference_service
        runtime_name = kserve_kueue_upgrade_serving_runtime.name
        pod_labels = [
            create_isvc_label_selector_str(
                isvc=isvc,
                resource_type="pod",
                runtime_name=runtime_name,
            )
        ]
        running_pods = 0
        gated_pods = 0
        try:
            for running_pods, gated_pods in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_2MIN,
                sleep=5,
                func=lambda: check_gated_pods_and_running_pods(pod_labels, isvc.namespace, admin_client),
            ):
                if running_pods >= KSERVE_KUEUE_EXPECTED_RUNNING_PODS:
                    break
        except TimeoutExpiredError:
            pytest.fail(
                f"Timeout waiting for a Running predictor pod before inference: "
                f"expected >={KSERVE_KUEUE_EXPECTED_RUNNING_PODS} running, "
                f"got {running_pods} running + {gated_pods} gated"
            )

        verify_kserve_kueue_upgrade_inference(
            inference_service=kserve_kueue_upgrade_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            inference_timeout=KSERVE_KUEUE_INFERENCE_TIMEOUT,
        )
