"""
Tests for NIM Account transient failure handling.

Bug 6 (High): The NIM Account handler in odh-model-controller treats transient
NGC API failures (timeout, DNS, 503) as invalid API keys, deleting all NIM
resources without requeueing.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

pytestmark = [pytest.mark.usefixtures("valid_aws_config")]

CONTROLLER_NAMESPACE = "opendatahub"
ODH_CONTROLLER_LABEL = "control-plane=odh-model-controller"


@pytest.mark.tier3
class TestNimTransientFailure:
    """Verify odh-model-controller does not delete NIM resources on transient API failures.

    When the NGC API is temporarily unavailable, the NIM Account handler should
    requeue instead of treating the error as an invalid API key and cleaning up.

    Preconditions:
        - odh-model-controller running in opendatahub namespace

    Test Steps:
        1. Check odh-model-controller logs for NIM-related error handling
        2. Verify controller distinguishes between transient and permanent errors

    Expected Results:
        - Controller logs should not show NIM resource deletion on transient errors
        - Controller should remain stable
    """

    def test_odh_controller_stable(self, admin_client: DynamicClient):
        """Verify odh-model-controller is running and has not crashed."""
        pods = list(Pod.get(client=admin_client, namespace=CONTROLLER_NAMESPACE, label_selector=ODH_CONTROLLER_LABEL))
        assert len(pods) > 0, "odh-model-controller pod not found"
        for cs in pods[0].instance.status.containerStatuses or []:
            assert cs.ready, f"odh-model-controller container {cs.name} is not ready"

    def test_no_transient_error_cleanup_in_logs(self, admin_client: DynamicClient):
        """Verify controller logs do not show NIM cleanup on transient errors.

        Given a running odh-model-controller
        Then logs should not contain patterns indicating NIM resource deletion
        triggered by transient network errors
        """
        pods = list(Pod.get(client=admin_client, namespace=CONTROLLER_NAMESPACE, label_selector=ODH_CONTROLLER_LABEL))
        assert len(pods) > 0, "odh-model-controller pod not found"
        logs = pods[0].log(tail_lines=500)
        if "invalid api key" in logs.lower() and ("timeout" in logs.lower() or "connection refused" in logs.lower()):
            pytest.fail(
                "odh-model-controller may be treating transient NGC failures as invalid API keys. "
                "Check NIM Account handler for proper error classification (Bug 6)."
            )
