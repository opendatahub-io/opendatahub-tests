import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.inference_service_lifecycle.constants import BASE_ISVC_CONFIG
from tests.model_serving.model_server.kserve.inference_service_lifecycle.utils import (
    trigger_isvc_reconcile,
    wait_for_elapsed,
)
from tests.model_serving.model_server.upgrade.utils import (
    capture_isvc_baseline,
    verify_isvc_pods_not_restarted_against_baseline,
)
from tests.model_serving.model_server.utils import isvc_running_pods_have_agent_sidecar
from utilities.constants import KServeDeploymentType, RunTimeConfigs, Timeout

# Bounded observation window after the kserve-agent config change and the
# triggered reconciliation. A rollout would surface as new/missing pods or an
# increased restart count within seconds of the reconcile, so a minute of
# sampling at 5s intervals is enough to catch it.
NO_ROLLOUT_WAIT_TIMEOUT: int = 60
NO_ROLLOUT_CHECK_INTERVAL: int = 5

RAW_DEPLOYMENT_AGENT_CONFIG_ISVC_CONFIG = {
    **BASE_ISVC_CONFIG,
    "name": "isvc-agent-config",
    "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
}

pytestmark = [pytest.mark.tier1, pytest.mark.rawdeployment, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.usefixtures("patched_kserve_agent_config")
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "test-agent-config-no-rollout"},
            RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG,
            RAW_DEPLOYMENT_AGENT_CONFIG_ISVC_CONFIG,
            id="test_raw_agent_config_change_no_rollout",
        ),
    ],
    indirect=True,
)
class TestRawISVCAgentConfigChangeNoRollout:
    """Validate that kserve-agent configuration changes do not restart logging-enabled ISVC pods.

    TrustyAI relies on inference logging: KServe injects the kserve-agent sidecar
    into model pods when `spec.predictor.logger` is set. Operator upgrades ship a
    new kserve-agent configuration (image/resources in the inferenceservice-config
    ConfigMap), which must not roll out running model pods
    (RHOAIENG-60588).

    Steps:
        1. Deploy a raw-deployment ISVC and enable inference logging.
        2. Verify the kserve-agent sidecar runs in the predictor pod.
        3. Change the agent configuration in the inferenceservice-config ConfigMap
           and trigger an ISVC reconciliation with a spec-neutral update.
        4. Verify no rollout: the same pods keep running, container restart counts
           do not increase and the agent sidecar stays present.
    """

    def test_agent_config_change_does_not_restart_pods(
        self,
        unprivileged_client: DynamicClient,
        logger_enabled_isvc: InferenceService,
    ) -> None:
        """Verify pods of a logging-enabled ISVC are not restarted on a kserve-agent config change"""
        baseline_restart_counts = capture_isvc_baseline(
            client=unprivileged_client,
            isvc=logger_enabled_isvc,
        )["pod_restart_counts"]

        with trigger_isvc_reconcile(client=unprivileged_client, isvc=logger_enabled_isvc):
            # The whole window is sampled (not exited on first success) so a
            # delayed or repeated rollout is still caught.
            for _ in wait_for_elapsed(
                sleep_interval=NO_ROLLOUT_CHECK_INTERVAL,
                wait_timeout=NO_ROLLOUT_WAIT_TIMEOUT,
            ):
                verify_isvc_pods_not_restarted_against_baseline(
                    client=unprivileged_client,
                    isvc=logger_enabled_isvc,
                    baseline_restart_counts=baseline_restart_counts,
                )
                assert isvc_running_pods_have_agent_sidecar(
                    client=unprivileged_client,
                    isvc=logger_enabled_isvc,
                ), f"kserve-agent sidecar disappeared from pods of {logger_enabled_isvc.name}"

        logger_enabled_isvc.wait_for_condition(
            condition=InferenceService.Condition.READY,
            status=InferenceService.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_1MIN,
        )
