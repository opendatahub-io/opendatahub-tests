"""N-1 workbench image survival tests across RHOAI platform upgrades."""

from typing import Any

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod

from tests.workbenches.notebook_images.utils import (
    UPGRADE_MARKER_CONTENT,
    WorkbenchImageSpec,
    get_workbench_image_specs,
    read_pvc_upgrade_marker,
    verify_workbench_survival,
)
from utilities.constants import Timeout

_WORKBENCH_IMAGE_SPECS = get_workbench_image_specs()
pytestmark = [
    pytest.mark.tier2,
    pytest.mark.slow,
    pytest.mark.parametrize(
        argnames="workbench_image_spec",
        argvalues=_WORKBENCH_IMAGE_SPECS,
        ids=[spec.ide for spec in _WORKBENCH_IMAGE_SPECS],
        indirect=True,
    ),
]


@pytest.mark.usefixtures("capture_n_minus_one_baseline")
class TestPreUpgradeNMinusOneWorkbench:
    """Launch workbenches on N-1 images before the platform upgrade."""

    @pytest.mark.pre_upgrade
    def test_workbench_running_on_n_minus_one_image(
        self,
        n_minus_one_pod: Pod,
        n_minus_one_image: str,
        n_minus_one_notebook: Notebook,
        workbench_image_spec: WorkbenchImageSpec,
    ) -> None:
        """Given a Notebook CR on an N-1 workbench image,
        When the notebook controller reconciles,
        Then the pod should exist, be Ready, and use the expected image.
        """
        assert n_minus_one_pod.exists, f"Pod for {workbench_image_spec.ide} was not created"

        annotations = n_minus_one_notebook.instance.metadata.annotations or {}
        selected_image = annotations.get("notebooks.opendatahub.io/last-image-selection")
        assert selected_image == n_minus_one_image, (
            f"Notebook image mismatch for {workbench_image_spec.ide}. "
            f"Expected '{n_minus_one_image}', got '{selected_image}'"
        )


class TestPostUpgradeNMinusOneWorkbench:
    """Verify N-1 workbench images remain healthy after the platform upgrade."""

    @pytest.mark.post_upgrade
    def test_pod_still_ready(
        self,
        n_minus_one_pod: Pod,
        workbench_image_spec: WorkbenchImageSpec,
    ) -> None:
        """Given a workbench was running on an N-1 image before upgrade,
        When the upgrade completes,
        Then the pod should still exist and be Ready.
        """
        assert n_minus_one_pod.exists, f"Pod '{workbench_image_spec.notebook_name}-0' no longer exists after upgrade"
        n_minus_one_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_5MIN,
        )

    @pytest.mark.post_upgrade
    def test_pod_not_recreated_after_upgrade(
        self,
        n_minus_one_pod: Pod,
        n_minus_one_baseline: dict[str, Any],
        workbench_image_spec: WorkbenchImageSpec,
    ) -> None:
        """Given a workbench pod was running before upgrade,
        When the upgrade completes,
        Then the pod object should not be recreated.
        """
        current_timestamp = n_minus_one_pod.instance.metadata.creationTimestamp
        saved_timestamp = n_minus_one_baseline["pod_creation_timestamp"]

        assert current_timestamp == saved_timestamp, (
            f"Workbench pod for {workbench_image_spec.ide} was recreated during upgrade. "
            f"Pre-upgrade: {saved_timestamp}, post-upgrade: {current_timestamp}"
        )

    @pytest.mark.post_upgrade
    def test_pvc_data_survives_upgrade(
        self,
        n_minus_one_pod: Pod,
        n_minus_one_baseline: dict[str, Any],
        workbench_image_spec: WorkbenchImageSpec,
    ) -> None:
        """Given a marker file was written to the workbench PVC before upgrade,
        When the upgrade completes,
        Then the marker content should still be readable from the PVC.
        """
        marker_content = read_pvc_upgrade_marker(
            pod=n_minus_one_pod,
            container_name=workbench_image_spec.notebook_name,
        )
        expected_marker = n_minus_one_baseline["upgrade_marker"]
        assert marker_content == expected_marker == UPGRADE_MARKER_CONTENT, (
            f"PVC marker mismatch for {workbench_image_spec.ide}. Expected '{expected_marker}', got '{marker_content}'"
        )

    @pytest.mark.post_upgrade
    def test_workbench_http_and_logs_after_upgrade(
        self,
        n_minus_one_pod: Pod,
        n_minus_one_namespace: Namespace,
        workbench_image_spec: WorkbenchImageSpec,
    ) -> None:
        """Given a workbench was running on an N-1 image before upgrade,
        When the upgrade completes,
        Then logs should be clean and, for Jupyter-style IDEs, HTTP should respond.
        """
        http_path = (
            f"/notebook/{n_minus_one_namespace.name}/{workbench_image_spec.notebook_name}/"
            if workbench_image_spec.probe_http
            else None
        )
        verify_workbench_survival(
            pod=n_minus_one_pod,
            container_name=workbench_image_spec.notebook_name,
            http_path=http_path,
        )
