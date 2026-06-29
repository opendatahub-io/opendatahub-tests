"""RStudio N-1 workbench image upgrade survival tests (legacy EUS track only)."""

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


class TestPreUpgradeRStudioWorkbench:
    """Pre-upgrade survival checks for a legacy RStudio workbench pinned to the source image tag."""

    @pytest.mark.pre_upgrade
    def test_rstudio_pre_upgrade_health(
        self,
        n1_rstudio_notebook: Notebook,
        n1_rstudio_pod: Pod,
        n1_rstudio_baseline: WorkbenchImageBaseline,
        n1_rstudio_case: WorkbenchImageSpec,
    ) -> None:
        """Given an RStudio workbench on the source image tag,
        When pre-upgrade validation runs,
        Then the workbench is Ready, logs are clean, and baseline data is captured.
        """
        verify_pre_upgrade_health(
            notebook=n1_rstudio_notebook,
            pod=n1_rstudio_pod,
            baseline=n1_rstudio_baseline,
            spec=n1_rstudio_case,
        )


class TestPostUpgradeRStudioWorkbench:
    """Post-upgrade survival checks for an RStudio workbench that should remain unchanged."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="rstudio_workbench_exists")
    def test_rstudio_post_upgrade_exists(
        self,
        n1_rstudio_notebook: Notebook,
        n1_rstudio_pod: Pod,
    ) -> None:
        """Given the pre-upgrade RStudio workbench,
        When the platform upgrade completes,
        Then the Notebook CR and original pod still exist and the pod is Ready.
        """
        verify_post_upgrade_exists(notebook=n1_rstudio_notebook, pod=n1_rstudio_pod)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="rstudio_pod_not_recreated", depends=["rstudio_workbench_exists"])
    def test_rstudio_post_upgrade_pod_not_recreated(
        self,
        n1_rstudio_pod: Pod,
        n1_rstudio_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade RStudio baseline,
        When the upgraded cluster reconciles the workbench,
        Then the original pod object is preserved.
        """
        verify_post_upgrade_pod_not_recreated(pod=n1_rstudio_pod, baseline=n1_rstudio_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="rstudio_image_selection_unchanged", depends=["rstudio_pod_not_recreated"])
    def test_rstudio_post_upgrade_image_selection_unchanged(
        self,
        n1_rstudio_notebook: Notebook,
        n1_rstudio_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade RStudio baseline,
        When the Notebook CR is inspected after upgrade,
        Then the last selected image annotation is unchanged.
        """
        verify_post_upgrade_image_selection(notebook=n1_rstudio_notebook, baseline=n1_rstudio_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="rstudio_image_digest_unchanged", depends=["rstudio_image_selection_unchanged"])
    def test_rstudio_post_upgrade_image_digest_unchanged(
        self,
        n1_rstudio_notebook: Notebook,
        n1_rstudio_pod: Pod,
        n1_rstudio_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade RStudio baseline,
        When the running container image is inspected after upgrade,
        Then the resolved digest matches the pre-upgrade digest.
        """
        verify_post_upgrade_image_digest(
            pod=n1_rstudio_pod,
            notebook=n1_rstudio_notebook,
            baseline=n1_rstudio_baseline,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="rstudio_restart_counts_unchanged", depends=["rstudio_image_digest_unchanged"])
    def test_rstudio_post_upgrade_restart_counts_unchanged(
        self,
        n1_rstudio_pod: Pod,
        n1_rstudio_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade RStudio baseline,
        When pod restart counts are compared after upgrade,
        Then no container restart count has increased.
        """
        verify_post_upgrade_restart_counts(pod=n1_rstudio_pod, baseline=n1_rstudio_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(
        name="rstudio_notebook_generation_unchanged",
        depends=["rstudio_restart_counts_unchanged"],
    )
    def test_rstudio_post_upgrade_notebook_generation_unchanged(
        self,
        n1_rstudio_notebook: Notebook,
        n1_rstudio_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade RStudio baseline,
        When the Notebook CR generation is compared after upgrade,
        Then the generation is unchanged.
        """
        verify_post_upgrade_notebook_generation(notebook=n1_rstudio_notebook, baseline=n1_rstudio_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(
        name="rstudio_statefulset_healthy",
        depends=["rstudio_notebook_generation_unchanged"],
    )
    def test_rstudio_post_upgrade_statefulset_healthy(
        self,
        n1_rstudio_statefulset: StatefulSet,
    ) -> None:
        """Given the pre-upgrade RStudio workbench,
        When the StatefulSet is inspected after upgrade,
        Then readyReplicas matches spec.replicas and no rollout is pending.
        """
        verify_post_upgrade_statefulset(statefulset=n1_rstudio_statefulset)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="rstudio_pvc_data_survives", depends=["rstudio_statefulset_healthy"])
    def test_rstudio_post_upgrade_pvc_data_survives(
        self,
        n1_rstudio_pod: Pod,
        n1_rstudio_baseline: WorkbenchImageBaseline,
        n1_rstudio_case: WorkbenchImageSpec,
    ) -> None:
        """Given a marker file was written to the PVC before upgrade,
        When the upgrade completes,
        Then the marker content is still readable from the PVC.
        """
        verify_post_upgrade_pvc_marker(
            pod=n1_rstudio_pod,
            baseline=n1_rstudio_baseline,
            spec=n1_rstudio_case,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["rstudio_pvc_data_survives"])
    def test_rstudio_post_upgrade_health(
        self,
        n1_rstudio_notebook: Notebook,
        n1_rstudio_pod: Pod,
        n1_rstudio_case: WorkbenchImageSpec,
    ) -> None:
        """Given the surviving RStudio workbench after upgrade,
        When workbench logs are checked again,
        Then no new errors are reported.
        """
        verify_post_upgrade_health(
            notebook=n1_rstudio_notebook,
            pod=n1_rstudio_pod,
            spec=n1_rstudio_case,
        )
