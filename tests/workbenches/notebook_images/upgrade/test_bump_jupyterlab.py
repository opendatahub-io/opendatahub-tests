"""Dashboard-driven N-1 to N workbench image bump tests (JupyterLab).

After the platform upgrade, this suite applies the same JSON patch the
Dashboard uses to bump a workbench from N-1 to N, then verifies the
workbench restarts healthy with PVC data intact.
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from tests.workbenches.notebook_images.utils import (
    ResolvedWorkbenchImage,
    WorkbenchImageBaseline,
    apply_dashboard_image_patch,
    build_dashboard_image_patch,
    get_container_image_digest,
    grab_and_check_pod_logs,
    read_pvc_upgrade_marker,
    wait_for_controller_reconciliation,
    wait_for_http_inside_pod,
)
from utilities.constants import Timeout

pytestmark = [pytest.mark.tier2, pytest.mark.slow]


class TestPreUpgradeBumpWorkbench:
    """Pre-upgrade setup for the dashboard image bump scenario."""

    @pytest.mark.pre_upgrade
    def test_bump_pre_upgrade_health(
        self,
        n1_bump_notebook: Notebook,
        n1_bump_pod: Pod,
        n1_bump_baseline: WorkbenchImageBaseline,
        n1_bump_marker_written: str,
    ) -> None:
        """Given a JupyterLab workbench on the N-1 image,
        When the pre-upgrade validation runs,
        Then the pod is Ready, logs are clean, the HTTP probe succeeds,
        a marker file is written to the PVC, and the baseline is captured.
        """
        assert n1_bump_notebook.exists
        assert n1_bump_pod.exists
        assert n1_bump_baseline.image_tag
        assert n1_bump_marker_written

        container_name = n1_bump_notebook.name
        wait_for_http_inside_pod(
            pod=n1_bump_pod,
            container_name=container_name,
            namespace=n1_bump_notebook.namespace,
            notebook_name=n1_bump_notebook.name,
        )
        grab_and_check_pod_logs(pod=n1_bump_pod, container_name=container_name)


class TestPostUpgradeBumpWorkbench:
    """Post-upgrade dashboard-driven image bump tests.

    After the platform upgrade, applies the Dashboard JSON patch to bump
    the workbench from N-1 to N, then verifies restart, image update,
    health, and PVC data integrity.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="bump_workbench_exists")
    def test_bump_post_upgrade_workbench_exists(
        self,
        n1_bump_notebook: Notebook,
        n1_bump_pod: Pod,
    ) -> None:
        """Given the pre-upgrade bump workbench,
        When the platform upgrade has finished,
        Then the Notebook CR and pod still exist.
        """
        assert n1_bump_notebook.exists, f"Notebook CR '{n1_bump_notebook.name}' not found after upgrade"
        assert n1_bump_pod.exists, f"Pod '{n1_bump_pod.name}' not found after upgrade"

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="bump_image_patched", depends=["bump_workbench_exists"])
    def test_bump_post_upgrade_apply_image_patch(
        self,
        admin_client: DynamicClient,
        unprivileged_client: DynamicClient,
        n1_bump_notebook: Notebook,
        n1_bump_pod: Pod,
        n1_bump_baseline: WorkbenchImageBaseline,
        n1_bump_target_image: ResolvedWorkbenchImage,
    ) -> None:
        """Given the surviving N-1 bump workbench and the resolved N image,
        When the Dashboard-equivalent JSON patch is applied,
        Then the old pod terminates and a new pod reaches Ready.
        """
        old_pod_uid = n1_bump_pod.instance.metadata.uid

        patch_ops = build_dashboard_image_patch(
            notebook=n1_bump_notebook,
            resolved_image=n1_bump_target_image,
        )
        apply_dashboard_image_patch(
            notebook=n1_bump_notebook,
            patch_ops=patch_ops,
        )

        pod_ref = Pod(
            client=unprivileged_client,
            name=f"{n1_bump_notebook.name}-0",
            namespace=n1_bump_notebook.namespace,
        )
        for sample in TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_5MIN,
            sleep=5,
            func=lambda: pod_ref.exists and pod_ref.instance.metadata.uid != old_pod_uid,
        ):
            if sample:
                break

        wait_for_controller_reconciliation(
            admin_client=admin_client,
            notebook_name=n1_bump_notebook.name,
            notebook_namespace=n1_bump_notebook.namespace,
            notebook_pod=pod_ref,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["bump_image_patched"])
    def test_bump_post_upgrade_image_updated(
        self,
        unprivileged_client: DynamicClient,
        n1_bump_notebook: Notebook,
        n1_bump_target_image: ResolvedWorkbenchImage,
    ) -> None:
        """Given the workbench was patched to the N image,
        When the Notebook CR annotations and running container image are inspected,
        Then both reflect the target N image.
        """
        annotations = n1_bump_notebook.instance.metadata.annotations or {}
        actual_selection = annotations.get("notebooks.opendatahub.io/last-image-selection")
        assert actual_selection == n1_bump_target_image.image_selection, (
            f"last-image-selection annotation not updated. "
            f"Expected '{n1_bump_target_image.image_selection}', got '{actual_selection}'"
        )

        new_pod = Pod(
            client=unprivileged_client,
            name=f"{n1_bump_notebook.name}-0",
            namespace=n1_bump_notebook.namespace,
        )
        actual_digest = get_container_image_digest(pod=new_pod, container_name=n1_bump_notebook.name)
        assert actual_digest == n1_bump_target_image.image_digest, (
            f"Running container digest does not match target N image. "
            f"Expected '{n1_bump_target_image.image_digest}', got '{actual_digest}'"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["bump_image_patched"])
    def test_bump_post_upgrade_health(
        self,
        unprivileged_client: DynamicClient,
        n1_bump_notebook: Notebook,
    ) -> None:
        """Given the bump workbench is running on the N image,
        When logs and the in-pod HTTP endpoint are checked,
        Then no unexpected errors are found and the workbench responds.
        """
        new_pod = Pod(
            client=unprivileged_client,
            name=f"{n1_bump_notebook.name}-0",
            namespace=n1_bump_notebook.namespace,
        )
        container_name = n1_bump_notebook.name
        wait_for_http_inside_pod(
            pod=new_pod,
            container_name=container_name,
            namespace=n1_bump_notebook.namespace,
            notebook_name=n1_bump_notebook.name,
        )
        grab_and_check_pod_logs(pod=new_pod, container_name=container_name)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["bump_image_patched"])
    def test_bump_post_upgrade_pvc_data_intact(
        self,
        unprivileged_client: DynamicClient,
        n1_bump_notebook: Notebook,
        n1_bump_marker_written: str,
    ) -> None:
        """Given a marker file was written to the PVC before the upgrade,
        When the marker file is read from the new pod after the image bump,
        Then its content matches the pre-upgrade value.
        """
        new_pod = Pod(
            client=unprivileged_client,
            name=f"{n1_bump_notebook.name}-0",
            namespace=n1_bump_notebook.namespace,
        )
        actual_marker = read_pvc_upgrade_marker(
            pod=new_pod,
            container_name=n1_bump_notebook.name,
        )
        assert actual_marker == n1_bump_marker_written, (
            f"PVC marker file content mismatch after image bump. "
            f"Expected '{n1_bump_marker_written}', got '{actual_marker}'"
        )
