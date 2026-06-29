"""JupyterLab N-1 workbench image upgrade survival tests."""

import pytest
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod

from tests.workbenches.notebook_images.upgrade.survival_checks import (
    verify_post_upgrade_exists,
    verify_post_upgrade_health,
    verify_post_upgrade_image_digest,
    verify_post_upgrade_image_selection,
    verify_post_upgrade_notebook_generation,
    verify_post_upgrade_pod_not_recreated,
    verify_post_upgrade_pvc_marker,
    verify_post_upgrade_restart_counts,
    verify_post_upgrade_statefulset,
    verify_pre_upgrade_health,
)
from tests.workbenches.notebook_images.utils import (
    StatefulSet,
    WorkbenchImageBaseline,
    WorkbenchImageSpec,
)

pytestmark = [pytest.mark.tier2, pytest.mark.slow]


class TestPreUpgradeJupyterLabWorkbench:
    """Pre-upgrade survival checks for a JupyterLab workbench pinned to the source image tag."""

    @pytest.mark.pre_upgrade
    def test_jupyterlab_pre_upgrade_health(
        self,
        n1_jupyterlab_notebook: Notebook,
        n1_jupyterlab_pod: Pod,
        n1_jupyterlab_baseline: WorkbenchImageBaseline,
        n1_jupyterlab_case: WorkbenchImageSpec,
    ) -> None:
        """Given a JupyterLab workbench on the source image tag,
        When pre-upgrade validation runs,
        Then the workbench is Ready, healthy, and baseline data is captured.
        """
        verify_pre_upgrade_health(
            notebook=n1_jupyterlab_notebook,
            pod=n1_jupyterlab_pod,
            baseline=n1_jupyterlab_baseline,
            spec=n1_jupyterlab_case,
        )


class TestPostUpgradeJupyterLabWorkbench:
    """Post-upgrade survival checks for a JupyterLab workbench that should remain unchanged."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyterlab_workbench_exists")
    def test_jupyterlab_post_upgrade_exists(
        self,
        n1_jupyterlab_notebook: Notebook,
        n1_jupyterlab_pod: Pod,
    ) -> None:
        """Given the pre-upgrade JupyterLab workbench,
        When the platform upgrade completes,
        Then the Notebook CR and original pod still exist and the pod is Ready.
        """
        verify_post_upgrade_exists(notebook=n1_jupyterlab_notebook, pod=n1_jupyterlab_pod)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyterlab_pod_not_recreated", depends=["jupyterlab_workbench_exists"])
    def test_jupyterlab_post_upgrade_pod_not_recreated(
        self,
        n1_jupyterlab_pod: Pod,
        n1_jupyterlab_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterLab baseline,
        When the upgraded cluster reconciles the workbench,
        Then the original pod object is preserved.
        """
        verify_post_upgrade_pod_not_recreated(pod=n1_jupyterlab_pod, baseline=n1_jupyterlab_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyterlab_image_selection_unchanged", depends=["jupyterlab_pod_not_recreated"])
    def test_jupyterlab_post_upgrade_image_selection_unchanged(
        self,
        n1_jupyterlab_notebook: Notebook,
        n1_jupyterlab_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterLab baseline,
        When the Notebook CR is inspected after upgrade,
        Then the last selected image annotation is unchanged.
        """
        verify_post_upgrade_image_selection(notebook=n1_jupyterlab_notebook, baseline=n1_jupyterlab_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyterlab_image_digest_unchanged", depends=["jupyterlab_image_selection_unchanged"])
    def test_jupyterlab_post_upgrade_image_digest_unchanged(
        self,
        n1_jupyterlab_notebook: Notebook,
        n1_jupyterlab_pod: Pod,
        n1_jupyterlab_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterLab baseline,
        When the running container image is inspected after upgrade,
        Then the resolved digest matches the pre-upgrade digest.
        """
        verify_post_upgrade_image_digest(
            pod=n1_jupyterlab_pod,
            notebook=n1_jupyterlab_notebook,
            baseline=n1_jupyterlab_baseline,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyterlab_restart_counts_unchanged", depends=["jupyterlab_image_digest_unchanged"])
    def test_jupyterlab_post_upgrade_restart_counts_unchanged(
        self,
        n1_jupyterlab_pod: Pod,
        n1_jupyterlab_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterLab baseline,
        When pod restart counts are compared after upgrade,
        Then no container restart count has increased.
        """
        verify_post_upgrade_restart_counts(pod=n1_jupyterlab_pod, baseline=n1_jupyterlab_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(
        name="jupyterlab_notebook_generation_unchanged",
        depends=["jupyterlab_restart_counts_unchanged"],
    )
    def test_jupyterlab_post_upgrade_notebook_generation_unchanged(
        self,
        n1_jupyterlab_notebook: Notebook,
        n1_jupyterlab_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterLab baseline,
        When the Notebook CR generation is compared after upgrade,
        Then the generation is unchanged.
        """
        verify_post_upgrade_notebook_generation(notebook=n1_jupyterlab_notebook, baseline=n1_jupyterlab_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(
        name="jupyterlab_statefulset_healthy",
        depends=["jupyterlab_notebook_generation_unchanged"],
    )
    def test_jupyterlab_post_upgrade_statefulset_healthy(
        self,
        n1_jupyterlab_statefulset: StatefulSet,
    ) -> None:
        """Given the pre-upgrade JupyterLab workbench,
        When the StatefulSet is inspected after upgrade,
        Then readyReplicas matches spec.replicas and no rollout is pending.
        """
        verify_post_upgrade_statefulset(statefulset=n1_jupyterlab_statefulset)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyterlab_pvc_data_survives", depends=["jupyterlab_statefulset_healthy"])
    def test_jupyterlab_post_upgrade_pvc_data_survives(
        self,
        n1_jupyterlab_pod: Pod,
        n1_jupyterlab_baseline: WorkbenchImageBaseline,
        n1_jupyterlab_case: WorkbenchImageSpec,
    ) -> None:
        """Given a marker file was written to the PVC before upgrade,
        When the upgrade completes,
        Then the marker content is still readable from the PVC.
        """
        verify_post_upgrade_pvc_marker(
            pod=n1_jupyterlab_pod,
            baseline=n1_jupyterlab_baseline,
            spec=n1_jupyterlab_case,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["jupyterlab_pvc_data_survives"])
    def test_jupyterlab_post_upgrade_health(
        self,
        n1_jupyterlab_notebook: Notebook,
        n1_jupyterlab_pod: Pod,
        n1_jupyterlab_case: WorkbenchImageSpec,
    ) -> None:
        """Given the surviving JupyterLab workbench after upgrade,
        When logs and in-pod HTTP are checked again,
        Then the health checks pass without new log errors.
        """
        verify_post_upgrade_health(
            notebook=n1_jupyterlab_notebook,
            pod=n1_jupyterlab_pod,
            spec=n1_jupyterlab_case,
        )
