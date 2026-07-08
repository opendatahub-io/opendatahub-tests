"""Shared validation steps for N-1 workbench image upgrade survival tests."""

from ocp_resources.notebook import Notebook
from ocp_resources.pod import Pod

from tests.workbenches.notebook_images.utils import (
    UPGRADE_MARKER_CONTENT,
    StatefulSet,
    WorkbenchImageBaseline,
    WorkbenchImageSpec,
    grab_and_check_pod_logs,
    read_pvc_upgrade_marker,
    verify_notebook_generation_unchanged,
    verify_notebook_image_digest_unchanged,
    verify_notebook_image_selection_unchanged,
    verify_notebook_pod_not_recreated,
    verify_notebook_restart_counts_unchanged,
    verify_statefulset_healthy,
    wait_for_http_inside_pod,
)


def verify_pre_upgrade_health(
    notebook: Notebook,
    pod: Pod,
    baseline: WorkbenchImageBaseline,
    spec: WorkbenchImageSpec,
) -> None:
    """Validate workbench readiness and capture baseline data before upgrade."""
    assert notebook.exists
    assert pod.exists
    assert baseline.image_tag

    container_name = spec.notebook_name
    grab_and_check_pod_logs(pod=pod, container_name=container_name)
    if spec.probe_http:
        wait_for_http_inside_pod(
            pod=pod,
            container_name=container_name,
            namespace=notebook.namespace,
            notebook_name=notebook.name,
        )
    grab_and_check_pod_logs(pod=pod, container_name=container_name)


def verify_post_upgrade_exists(notebook: Notebook, pod: Pod) -> None:
    """Verify the Notebook CR and pod still exist and the pod is Ready after upgrade."""
    assert notebook.exists
    assert pod.exists
    pod.wait_for_condition(
        condition=Pod.Condition.READY,
        status=Pod.Condition.Status.TRUE,
        timeout=300,
    )


def verify_post_upgrade_pod_not_recreated(pod: Pod, baseline: WorkbenchImageBaseline) -> None:
    """Verify the original pod object was not recreated during upgrade."""
    verify_notebook_pod_not_recreated(pod=pod, baseline=baseline)


def verify_post_upgrade_image_selection(notebook: Notebook, baseline: WorkbenchImageBaseline) -> None:
    """Verify the Notebook image selection annotation is unchanged."""
    verify_notebook_image_selection_unchanged(notebook=notebook, baseline=baseline)


def verify_post_upgrade_image_digest(
    pod: Pod,
    notebook: Notebook,
    baseline: WorkbenchImageBaseline,
) -> None:
    """Verify the running container digest matches the pre-upgrade baseline."""
    verify_notebook_image_digest_unchanged(
        pod=pod,
        container_name=notebook.name,
        baseline=baseline,
    )


def verify_post_upgrade_restart_counts(pod: Pod, baseline: WorkbenchImageBaseline) -> None:
    """Verify container restart counts did not increase across the upgrade."""
    verify_notebook_restart_counts_unchanged(pod=pod, baseline=baseline)


def verify_post_upgrade_notebook_generation(notebook: Notebook, baseline: WorkbenchImageBaseline) -> None:
    """Verify the Notebook CR generation is unchanged."""
    verify_notebook_generation_unchanged(notebook=notebook, baseline=baseline)


def verify_post_upgrade_statefulset(statefulset: StatefulSet) -> None:
    """Verify the workbench StatefulSet is healthy after upgrade."""
    verify_statefulset_healthy(statefulset=statefulset)


def verify_post_upgrade_pvc_marker(
    pod: Pod,
    baseline: WorkbenchImageBaseline,
    spec: WorkbenchImageSpec,
) -> None:
    """Verify the PVC marker file written pre-upgrade is still readable."""
    marker_content = read_pvc_upgrade_marker(
        pod=pod,
        container_name=spec.notebook_name,
    )
    assert marker_content == baseline.upgrade_marker == UPGRADE_MARKER_CONTENT, (
        f"PVC marker mismatch for {spec.ide}. Expected '{baseline.upgrade_marker}', got '{marker_content}'"
    )


def verify_post_upgrade_health(
    notebook: Notebook,
    pod: Pod,
    spec: WorkbenchImageSpec,
) -> None:
    """Re-run log and HTTP health checks after upgrade."""
    container_name = spec.notebook_name
    grab_and_check_pod_logs(pod=pod, container_name=container_name)
    if spec.probe_http:
        wait_for_http_inside_pod(
            pod=pod,
            container_name=container_name,
            namespace=notebook.namespace,
            notebook_name=notebook.name,
        )
    grab_and_check_pod_logs(pod=pod, container_name=container_name)
