"""Parametrized N-1 workbench image upgrade survival tests.

Replaces per-IDE test modules (JupyterLab, Code Server, RStudio) with a single
parametrized suite driven by ``get_workbench_image_specs()``.
"""

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
    get_workbench_image_specs,
    verify_kernel_variable,
)

pytestmark = [
    pytest.mark.tier2,
    pytest.mark.slow,
    pytest.mark.parametrize(
        "n1_workbench_spec",
        get_workbench_image_specs(),
        indirect=True,
        ids=lambda spec: spec.ide,
    ),
]


class TestPreUpgradeWorkbench:
    """Pre-upgrade survival checks for a workbench pinned to the source image tag."""

    @pytest.mark.pre_upgrade
    def test_pre_upgrade_health(
        self,
        n1_notebook: Notebook,
        n1_pod: Pod,
        n1_baseline: WorkbenchImageBaseline,
        n1_workbench_spec: WorkbenchImageSpec,
    ) -> None:
        """Given a workbench on the source image tag,
        When pre-upgrade validation runs,
        Then the workbench is Ready, healthy, and baseline data is captured.
        """
        verify_pre_upgrade_health(
            notebook=n1_notebook,
            pod=n1_pod,
            baseline=n1_baseline,
            spec=n1_workbench_spec,
        )

    @pytest.mark.pre_upgrade
    def test_pre_upgrade_kernel_started(
        self,
        n1_kernel_id: str,
    ) -> None:
        """Given a JupyterLab workbench on the source image tag,
        When a Jupyter kernel is started and ``a = 3 + 4`` is executed,
        Then the kernel ID is captured for post-upgrade verification.
        """
        assert n1_kernel_id


class TestPostUpgradeWorkbench:
    """Post-upgrade survival checks for a workbench that should remain unchanged."""

    @pytest.mark.post_upgrade
    def test_post_upgrade_exists(
        self,
        n1_notebook: Notebook,
        n1_pod: Pod,
    ) -> None:
        """Given the pre-upgrade workbench,
        When the platform upgrade completes,
        Then the Notebook CR and original pod still exist and the pod is Ready.
        """
        verify_post_upgrade_exists(notebook=n1_notebook, pod=n1_pod)

    @pytest.mark.post_upgrade
    def test_post_upgrade_pod_not_recreated(
        self,
        n1_pod: Pod,
        n1_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade baseline,
        When the upgraded cluster reconciles the workbench,
        Then the original pod object is preserved.
        """
        verify_post_upgrade_pod_not_recreated(pod=n1_pod, baseline=n1_baseline)

    @pytest.mark.post_upgrade
    def test_post_upgrade_image_selection_unchanged(
        self,
        n1_notebook: Notebook,
        n1_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade baseline,
        When the Notebook CR is inspected after upgrade,
        Then the last selected image annotation is unchanged.
        """
        verify_post_upgrade_image_selection(notebook=n1_notebook, baseline=n1_baseline)

    @pytest.mark.post_upgrade
    def test_post_upgrade_image_digest_unchanged(
        self,
        n1_notebook: Notebook,
        n1_pod: Pod,
        n1_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade baseline,
        When the running container image is inspected after upgrade,
        Then the resolved digest matches the pre-upgrade digest.
        """
        verify_post_upgrade_image_digest(
            pod=n1_pod,
            notebook=n1_notebook,
            baseline=n1_baseline,
        )

    @pytest.mark.post_upgrade
    def test_post_upgrade_restart_counts_unchanged(
        self,
        n1_pod: Pod,
        n1_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade baseline,
        When pod restart counts are compared after upgrade,
        Then no container restart count has increased.
        """
        verify_post_upgrade_restart_counts(pod=n1_pod, baseline=n1_baseline)

    @pytest.mark.post_upgrade
    def test_post_upgrade_notebook_generation_unchanged(
        self,
        n1_notebook: Notebook,
        n1_baseline: WorkbenchImageBaseline,
    ) -> None:
        """Given the pre-upgrade baseline,
        When the Notebook CR generation is compared after upgrade,
        Then the generation is unchanged.
        """
        verify_post_upgrade_notebook_generation(notebook=n1_notebook, baseline=n1_baseline)

    @pytest.mark.post_upgrade
    def test_post_upgrade_statefulset_healthy(
        self,
        n1_statefulset: StatefulSet,
    ) -> None:
        """Given the pre-upgrade workbench,
        When the StatefulSet is inspected after upgrade,
        Then readyReplicas matches spec.replicas and no rollout is pending.
        """
        verify_post_upgrade_statefulset(statefulset=n1_statefulset)

    @pytest.mark.post_upgrade
    def test_post_upgrade_pvc_data_survives(
        self,
        n1_pod: Pod,
        n1_baseline: WorkbenchImageBaseline,
        n1_workbench_spec: WorkbenchImageSpec,
    ) -> None:
        """Given a marker file was written to the PVC before upgrade,
        When the upgrade completes,
        Then the marker content is still readable from the PVC.
        """
        verify_post_upgrade_pvc_marker(
            pod=n1_pod,
            baseline=n1_baseline,
            spec=n1_workbench_spec,
        )

    @pytest.mark.post_upgrade
    def test_post_upgrade_health(
        self,
        n1_notebook: Notebook,
        n1_pod: Pod,
        n1_workbench_spec: WorkbenchImageSpec,
    ) -> None:
        """Given the surviving workbench after upgrade,
        When logs and in-pod HTTP are checked again,
        Then the health checks pass without new log errors.
        """
        verify_post_upgrade_health(
            notebook=n1_notebook,
            pod=n1_pod,
            spec=n1_workbench_spec,
        )

    @pytest.mark.post_upgrade
    def test_post_upgrade_kernel_state_intact(
        self,
        n1_pod: Pod,
        n1_workbench_spec: WorkbenchImageSpec,
        n1_kernel_id: str,
    ) -> None:
        """Given a Jupyter kernel was running with ``a = 7`` before upgrade,
        When the kernel is reconnected after upgrade,
        Then ``print(a * 6)`` returns ``42``.
        """
        result = verify_kernel_variable(
            pod=n1_pod,
            container_name=n1_workbench_spec.notebook_name,
            kernel_id=n1_kernel_id,
        )
        assert result == "42"
