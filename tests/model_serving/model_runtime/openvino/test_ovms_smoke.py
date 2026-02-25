"""
OVMS smoke test: run smoke scripts inside OpenShift using the OVMS runtime image.

How the Pod works:
  - A Pod is created in the test namespace with restart_policy=Never.
  - The smoke scripts (ovms_smoketest.py, smoke.py) are mounted read-only at /scripts
    via a ConfigMap populated from the repo files.
  - The container runs: python /scripts/ovms_smoketest.py && python /scripts/smoke.py.
  - If both scripts exit 0, the Pod phase becomes Succeeded.
  - If either script fails (non-zero exit or exception), the Pod fails and the test fails.
  - The test asserts Pod phase Succeeded; logs available via oc logs for debugging.
"""

import pytest
from ocp_resources.pod import Pod


@pytest.mark.smoke
@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": "ovms-smoke"}, id="ovms-smoke")],
    indirect=["model_namespace"],
)
class TestOVMSSmokeInOpenShift:
    """
    Test class for OVMS smoke execution inside OpenShift.

    Runs ovms_smoketest.py and smoke.py inside a Pod using the OVMS runtime image,
    with optional image override via --ovms-runtime-image.
    """

    def test_ovms_smoke_runs_in_openshift(self, ovms_smoke_pod: Pod) -> None:
        """
        OVMS smoke scripts run successfully inside an OpenShift Pod.

        Given the OVMS runtime image (from --ovms-runtime-image or template),
        when the smoke Pod runs ovms_smoketest.py and smoke.py in the container,
        then the Pod completes with phase Succeeded and the test passes.

        Args:
            ovms_smoke_pod: The completed Kubernetes Pod that ran the smoke scripts.
        """
        assert ovms_smoke_pod.instance.status.phase == "Succeeded", (
            f"OVMS smoke Pod did not succeed: phase={ovms_smoke_pod.instance.status.phase}"
        )
