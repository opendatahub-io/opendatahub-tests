import pytest
from ocp_resources.pod import Pod


@pytest.mark.usefixtures("capture_notebook_baseline")
class TestPreUpgradeNotebook:
    """Verify a workbench notebook is running before the platform upgrade.

    Steps:
        1. Create a Notebook CR with PVC and supporting resources.
        2. Wait for the notebook pod to reach Ready state.
        3. Capture the pod's creationTimestamp to a ConfigMap for post-upgrade comparison.
    """

    @pytest.mark.pre_upgrade
    def test_notebook_running_before_upgrade(self, upgrade_notebook_pod: Pod) -> None:
        """Given a Notebook CR is created before upgrade,
        When the notebook controller reconciles and starts the pod,
        Then the notebook pod should exist and be in Ready state.
        """
        assert upgrade_notebook_pod.exists, f"Notebook pod '{upgrade_notebook_pod.name}' does not exist"


class TestPostUpgradeNotebook:
    """Verify the workbench notebook survived the platform upgrade.

    Steps:
        1. Verify the notebook pod still exists after upgrade.
        2. Compare the pod's creationTimestamp against the pre-upgrade baseline.
        3. Clean up the Notebook CR.
    """

    @pytest.mark.post_upgrade
    def test_notebook_not_restarted_after_upgrade(
        self,
        upgrade_notebook_pod: Pod,
        upgrade_notebook_baseline: dict[str, str],
    ) -> None:
        """Given a notebook was running before upgrade,
        When the upgrade completes,
        Then the pod's creationTimestamp should match the pre-upgrade baseline.
        """
        assert upgrade_notebook_pod.exists, f"Notebook pod '{upgrade_notebook_pod.name}' no longer exists after upgrade"

        current_timestamp = upgrade_notebook_pod.instance.metadata.creationTimestamp
        saved_timestamp = upgrade_notebook_baseline["ntb_creation_timestamp"]

        assert current_timestamp == saved_timestamp, (
            f"Notebook pod was restarted during upgrade. "
            f"Pre-upgrade creationTimestamp: {saved_timestamp}, "
            f"post-upgrade creationTimestamp: {current_timestamp}"
        )
