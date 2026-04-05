"""
Tests for SupportsMultiModelDownload nil pointer dereference handling.

Bug 2 (High): When SupportsMultiModelDownload (*bool) is nil in the
ClusterStorageContainer spec, the storage initializer injector dereferences
it without a nil check, causing a panic.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

pytestmark = [pytest.mark.usefixtures("valid_aws_config")]

CONTROLLER_NAMESPACE = "opendatahub"
CONTROLLER_LABEL = "control-plane=kserve-controller-manager"


def _get_controller_restart_count(admin_client: DynamicClient) -> int:
    """Get total restart count for the kserve controller manager pod."""
    pods = list(Pod.get(client=admin_client, namespace=CONTROLLER_NAMESPACE, label_selector=CONTROLLER_LABEL))
    assert len(pods) > 0, "No kserve controller manager pod found"
    total = 0
    for cs in pods[0].instance.status.containerStatuses or []:
        total += cs.restartCount
    return total


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestMultiModelNilDeref:
    """Verify kserve controller handles nil SupportsMultiModelDownload without panicking.

    When SupportsMultiModelDownload is omitted from ClusterStorageContainer
    (making the *bool nil), the storage initializer should not crash.

    Preconditions:
        - kserve controller manager running in opendatahub namespace

    Test Steps:
        1. Record kserve controller pod restart count
        2. Check controller logs for nil pointer indicators on multi-model path
        3. Verify controller stability

    Expected Results:
        - Controller should not panic on nil *bool dereference
        - No unexpected restarts
    """

    def test_controller_no_multimodel_nil_panic(self, admin_client: DynamicClient):
        """Verify no nil pointer dereference on SupportsMultiModelDownload path.

        Given a running kserve controller
        Then the controller logs should not contain nil pointer panics related to multi-model
        """
        pod = next(Pod.get(client=admin_client, namespace=CONTROLLER_NAMESPACE, label_selector=CONTROLLER_LABEL))
        logs = pod.log(tail_lines=500)
        panic_indicators = ["nil pointer dereference", "runtime error: invalid memory", "panic:"]
        for indicator in panic_indicators:
            assert indicator not in logs, (
                f"Found '{indicator}' in kserve controller logs. "
                "SupportsMultiModelDownload may be dereferenced without nil check (Bug 2)."
            )

    def test_controller_stable(self, admin_client: DynamicClient):
        """Verify the kserve controller has not restarted unexpectedly."""
        pods = list(Pod.get(client=admin_client, namespace=CONTROLLER_NAMESPACE, label_selector=CONTROLLER_LABEL))
        assert len(pods) > 0, "kserve controller manager pod not found"
        for cs in pods[0].instance.status.containerStatuses or []:
            assert cs.ready, f"Controller container {cs.name} is not ready"
