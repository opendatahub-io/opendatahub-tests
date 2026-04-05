"""
Tests for kserve storage initializer nil pointer handling.

Bug 1 (Critical): When GetStorageContainerSpec returns an error, the storage
initializer injector accesses supported.Container.Name on a nil pointer,
causing a panic in the kserve controller webhook.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

from utilities.constants import KServeDeploymentType, ModelAndFormat, RuntimeTemplates
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns
from utilities.serving_runtime import ServingRuntimeFromTemplate

pytestmark = [pytest.mark.usefixtures("valid_aws_config")]

CONTROLLER_NAMESPACE = "opendatahub"
CONTROLLER_LABEL = "control-plane=kserve-controller-manager"


def _get_controller_pod(admin_client: DynamicClient) -> Pod:
    """Get the kserve controller manager pod."""
    pods = list(Pod.get(client=admin_client, namespace=CONTROLLER_NAMESPACE, label_selector=CONTROLLER_LABEL))
    assert len(pods) > 0, "No kserve controller manager pod found"
    return pods[0]


def _get_controller_restart_count(admin_client: DynamicClient) -> int:
    """Get total restart count for the kserve controller manager pod."""
    pod = _get_controller_pod(admin_client=admin_client)
    total = 0
    for cs in pod.instance.status.containerStatuses or []:
        total += cs.restartCount
    return total


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestStorageInitializerNilPanic:
    """Verify kserve controller handles storage initializer errors without crashing.

    When GetStorageContainerSpec fails, the webhook should return a meaningful error
    instead of panicking on a nil pointer dereference.

    Preconditions:
        - kserve controller manager is running in opendatahub namespace
        - Cluster has valid DSC/DSCI configuration

    Test Steps:
        1. Record kserve controller pod restart count
        2. Create an ISVC with an invalid/unreachable storage path
        3. Verify the controller pod did not crash (no panic, no restart)
        4. Check controller logs for nil pointer or panic indicators
    """

    def test_controller_no_panic_on_storage_error(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        ci_s3_bucket_name: str,
        ci_endpoint_s3_secret,
    ):
        """Verify the kserve controller does not panic when storage initialization fails.

        Given a running kserve controller
        When deploying an ISVC with a deliberately invalid storage configuration
        Then the controller should not panic or restart
        """
        initial_restarts = _get_controller_restart_count(admin_client=admin_client)

        with (
            create_ns(
                admin_client=admin_client,
                unprivileged_client=unprivileged_client,
                name="storage-init-nil-test",
            ) as ns,
            ServingRuntimeFromTemplate(
                client=unprivileged_client,
                namespace=ns.name,
                name="storage-nil-runtime",
                template_name=RuntimeTemplates.OVMS_KSERVE,
                multi_model=False,
            ) as runtime,
        ):
            try:
                with create_isvc(
                    client=unprivileged_client,
                    name="storage-nil-test-isvc",
                    namespace=ns.name,
                    runtime=runtime.name,
                    storage_key=ci_endpoint_s3_secret.name,
                    storage_path="nonexistent/invalid/path/model",
                    model_format=ModelAndFormat.OPENVINO_IR,
                    deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
                    wait_for_predictor_pods=False,
                ):
                    pass
            except Exception:  # noqa: BLE001, S110
                pass

        current_restarts = _get_controller_restart_count(admin_client=admin_client)
        assert current_restarts == initial_restarts, (
            f"Controller restarted during test. Initial: {initial_restarts}, Current: {current_restarts}. "
            "This may indicate a nil pointer panic in storage initializer."
        )

    def test_controller_logs_no_nil_pointer(self, admin_client: DynamicClient):
        """Verify no nil pointer panic appears in the kserve controller logs.

        Given a kserve controller manager pod
        Then its recent logs should not contain nil pointer dereference indicators
        """
        pod = _get_controller_pod(admin_client=admin_client)
        logs = pod.log(tail_lines=500)
        panic_indicators = ["nil pointer dereference", "runtime error: invalid memory", "panic:"]
        for indicator in panic_indicators:
            assert indicator not in logs, (
                f"Found '{indicator}' in kserve controller logs. Storage initializer may be crashing on nil pointer."
            )
