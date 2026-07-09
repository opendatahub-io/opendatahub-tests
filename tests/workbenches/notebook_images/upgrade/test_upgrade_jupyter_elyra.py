"""JupyterLab with Elyra N-1 workbench image upgrade survival tests."""

import pytest
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod

from tests.workbenches.notebook_images.upgrade.elyra_utils import (
    verify_post_upgrade_elyra_extensions_preserved,
    verify_post_upgrade_elyra_runtime_configs_preserved,
    verify_pre_upgrade_elyra_installed,
)
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


class TestPreUpgradeJupyterElyraWorkbench:
    """Pre-upgrade survival checks for a JupyterElyra workbench pinned to the source image tag."""

    @pytest.mark.pre_upgrade
    def test_jupyter_elyra_pre_upgrade_health(
        self,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
        n1_jupyter_elyra_case: WorkbenchImageSpec,
    ) -> None:
        """Given a JupyterElyra workbench on the source image tag,
        When pre-upgrade validation runs,
        Then the workbench is Ready, healthy, and baseline data is captured.
        """
        verify_pre_upgrade_health(
            notebook=n1_jupyter_elyra_notebook,
            pod=n1_jupyter_elyra_pod,
            baseline=n1_jupyter_elyra_baseline,
            spec=n1_jupyter_elyra_case,
        )

    @pytest.mark.pre_upgrade
    def test_jupyter_elyra_elyra_installed(
        self,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_case: WorkbenchImageSpec,
    ) -> None:
        """Verify Elyra extensions are installed and healthy.

        REQUIRED: This test fails if Elyra is not found, since the
        datascience workbench image must have Elyra pre-installed.
        """
        verify_pre_upgrade_elyra_installed(
            pod=n1_jupyter_elyra_pod,
            notebook=n1_jupyter_elyra_notebook,
            spec=n1_jupyter_elyra_case,
        )


class TestPostUpgradeJupyterElyraWorkbench:
    """Post-upgrade survival checks for a JupyterElyra workbench that should remain unchanged."""

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyter_elyra_workbench_exists")
    def test_jupyter_elyra_post_upgrade_exists(
        self,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_pod: Pod,
    ) -> None:
        """Given the pre-upgrade JupyterElyra workbench,
        When the platform upgrade completes,
        Then the Notebook CR and original pod still exist and the pod is Ready.
        """
        verify_post_upgrade_exists(notebook=n1_jupyter_elyra_notebook, pod=n1_jupyter_elyra_pod)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyter_elyra_pod_not_recreated", depends=["jupyter_elyra_workbench_exists"])
    def test_jupyter_elyra_post_upgrade_pod_not_recreated(
        self,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterElyra baseline,
        When the upgraded cluster reconciles the workbench,
        Then the original pod object is preserved.
        """
        verify_post_upgrade_pod_not_recreated(pod=n1_jupyter_elyra_pod, baseline=n1_jupyter_elyra_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyter_elyra_image_selection_unchanged", depends=["jupyter_elyra_pod_not_recreated"])
    def test_jupyter_elyra_post_upgrade_image_selection_unchanged(
        self,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterElyra baseline,
        When the Notebook CR is inspected after upgrade,
        Then the last selected image annotation is unchanged.
        """
        verify_post_upgrade_image_selection(notebook=n1_jupyter_elyra_notebook, baseline=n1_jupyter_elyra_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(
        name="jupyter_elyra_image_digest_unchanged", depends=["jupyter_elyra_image_selection_unchanged"]
    )
    def test_jupyter_elyra_post_upgrade_image_digest_unchanged(
        self,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterElyra baseline,
        When the running container image is inspected after upgrade,
        Then the resolved digest matches the pre-upgrade digest.
        """
        verify_post_upgrade_image_digest(
            pod=n1_jupyter_elyra_pod,
            notebook=n1_jupyter_elyra_notebook,
            baseline=n1_jupyter_elyra_baseline,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(
        name="jupyter_elyra_restart_counts_unchanged", depends=["jupyter_elyra_image_digest_unchanged"]
    )
    def test_jupyter_elyra_post_upgrade_restart_counts_unchanged(
        self,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterElyra baseline,
        When pod restart counts are compared after upgrade,
        Then no container restart count has increased.
        """
        verify_post_upgrade_restart_counts(pod=n1_jupyter_elyra_pod, baseline=n1_jupyter_elyra_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(
        name="jupyter_elyra_notebook_generation_unchanged",
        depends=["jupyter_elyra_restart_counts_unchanged"],
    )
    def test_jupyter_elyra_post_upgrade_notebook_generation_unchanged(
        self,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade JupyterElyra baseline,
        When the Notebook CR generation is compared after upgrade,
        Then the generation is unchanged.
        """
        verify_post_upgrade_notebook_generation(notebook=n1_jupyter_elyra_notebook, baseline=n1_jupyter_elyra_baseline)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(
        name="jupyter_elyra_statefulset_healthy",
        depends=["jupyter_elyra_notebook_generation_unchanged"],
    )
    def test_jupyter_elyra_post_upgrade_statefulset_healthy(
        self,
        n1_jupyter_elyra_statefulset: StatefulSet,
    ) -> None:
        """Given the pre-upgrade JupyterElyra workbench,
        When the StatefulSet is inspected after upgrade,
        Then readyReplicas matches spec.replicas and no rollout is pending.
        """
        verify_post_upgrade_statefulset(statefulset=n1_jupyter_elyra_statefulset)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyter_elyra_pvc_data_survives", depends=["jupyter_elyra_statefulset_healthy"])
    def test_jupyter_elyra_post_upgrade_pvc_data_survives(
        self,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
        n1_jupyter_elyra_case: WorkbenchImageSpec,
    ) -> None:
        """Given a marker file was written to the PVC before upgrade,
        When the upgrade completes,
        Then the marker content is still readable from the PVC.
        """
        verify_post_upgrade_pvc_marker(
            pod=n1_jupyter_elyra_pod,
            baseline=n1_jupyter_elyra_baseline,
            spec=n1_jupyter_elyra_case,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["jupyter_elyra_pvc_data_survives"])
    def test_jupyter_elyra_post_upgrade_health(
        self,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_case: WorkbenchImageSpec,
    ) -> None:
        """Given the surviving JupyterElyra workbench after upgrade,
        When logs and in-pod HTTP are checked again,
        Then the health checks pass without new log errors.
        """
        verify_post_upgrade_health(
            notebook=n1_jupyter_elyra_notebook,
            pod=n1_jupyter_elyra_pod,
            spec=n1_jupyter_elyra_case,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="jupyter_elyra_elyra_extensions_preserved", depends=["jupyter_elyra_workbench_exists"])
    def test_jupyter_elyra_post_upgrade_elyra_extensions_preserved(
        self,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade Elyra baseline,
        When Elyra extensions are checked after upgrade,
        Then all extensions are preserved with unchanged status.
        """
        verify_post_upgrade_elyra_extensions_preserved(
            pod=n1_jupyter_elyra_pod,
            notebook=n1_jupyter_elyra_notebook,
            baseline=n1_jupyter_elyra_baseline,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["jupyter_elyra_elyra_extensions_preserved"])
    def test_jupyter_elyra_post_upgrade_elyra_runtime_configs_preserved(
        self,
        n1_jupyter_elyra_pod: Pod,
        n1_jupyter_elyra_notebook: Notebook,
        n1_jupyter_elyra_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade Elyra baseline,
        When runtime configurations are checked after upgrade,
        Then all configs are preserved with unchanged content.
        """
        verify_post_upgrade_elyra_runtime_configs_preserved(
            pod=n1_jupyter_elyra_pod,
            notebook=n1_jupyter_elyra_notebook,
            baseline=n1_jupyter_elyra_baseline,
        )
