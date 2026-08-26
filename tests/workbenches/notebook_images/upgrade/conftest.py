"""Fixtures for N-1 workbench image upgrade survival tests."""

from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import config as py_config

from tests.workbenches.notebook_images.utils import (
    UPGRADE_BASELINE_CM_NAME,
    UPGRADE_MARKER_CONTENT,
    UPGRADE_NAMESPACE,
    ResolvedWorkbenchImage,
    StatefulSet,
    WorkbenchImageBaseline,
    WorkbenchImageSpec,
    capture_or_load_workbench_baseline,
    get_ready_upgrade_notebook_pod,
    get_workbench_image_spec_by_ide,
    is_legacy_track_tag,
    manage_upgrade_notebook,
    manage_upgrade_persistent_volume_claim,
    resolve_current_image,
    resolve_workbench_image,
    resolve_workbench_upgrade_track,
    should_skip_workbench_spec,
    start_kernel_and_set_variable,
    write_pvc_upgrade_marker,
)
from utilities.infra import create_ns


@pytest.fixture(scope="session")
def workbench_upgrade_track(admin_client: DynamicClient) -> str:
    """Return the configured workbench upgrade track."""
    return resolve_workbench_upgrade_track(admin_client=admin_client)


@pytest.fixture(scope="session")
def n1_notebook_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace used by workbench image survival upgrade tests."""
    ns = Namespace(client=unprivileged_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        if teardown_resources:
            Namespace(client=admin_client, name=UPGRADE_NAMESPACE).clean_up()
    else:
        if Namespace(client=admin_client, name=UPGRADE_NAMESPACE).exists:
            raise AssertionError(
                f"Namespace '{UPGRADE_NAMESPACE}' already exists. "
                "This indicates a previous test run did not clean up properly."
            )

        with create_ns(
            admin_client=admin_client,
            unprivileged_client=unprivileged_client,
            name=UPGRADE_NAMESPACE,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as created_namespace:
            yield created_namespace


@pytest.fixture(scope="session")
def n1_image_baseline_configmap(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    teardown_resources: bool,
) -> Generator[ConfigMap, Any, Any]:
    """Shared ConfigMap that carries notebook baselines across the upgrade boundary."""
    config_map = ConfigMap(
        client=admin_client,
        name=UPGRADE_BASELINE_CM_NAME,
        namespace=n1_notebook_namespace.name,
        data={},
        ensure_exists=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    )

    if pytestconfig.option.post_upgrade:
        yield config_map
    else:
        with config_map:
            yield config_map


@pytest.fixture(scope="session")
def n1_workbench_spec(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    workbench_upgrade_track: str,
    pytestconfig: pytest.Config,
) -> WorkbenchImageSpec:
    """Resolve the workbench spec for the current parametrize ID and skip if unsupported."""
    spec: WorkbenchImageSpec = request.param
    if skip_reason := should_skip_workbench_spec(
        admin_client=admin_client,
        spec=spec,
        post_upgrade=pytestconfig.option.post_upgrade,
        workbench_upgrade_track=workbench_upgrade_track,
    ):
        pytest.skip(skip_reason)
    return spec


@pytest.fixture(scope="session")
def n1_image(
    admin_client: DynamicClient,
    n1_workbench_spec: WorkbenchImageSpec,
    pytestconfig: pytest.Config,
) -> ResolvedWorkbenchImage | None:
    """Resolved pre-upgrade image for the current IDE (None post-upgrade)."""
    if pytestconfig.option.post_upgrade:
        return None
    resolved_image = resolve_workbench_image(admin_client=admin_client, spec=n1_workbench_spec)
    if n1_workbench_spec.require_eus_track and not is_legacy_track_tag(tag_name=resolved_image.tag_name):
        pytest.skip(f"{n1_workbench_spec.ide} workbench survival tests require a legacy EUS workbench image tag")
    return resolved_image


@pytest.fixture(scope="session")
def n1_pvc(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_workbench_spec: WorkbenchImageSpec,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC backing the current IDE's upgrade workbench."""
    yield from manage_upgrade_persistent_volume_claim(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_workbench_spec,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_workbench_spec: WorkbenchImageSpec,
    n1_image: ResolvedWorkbenchImage,
    n1_pvc: PersistentVolumeClaim,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR pinned to the current IDE's survival image."""
    del n1_pvc
    yield from manage_upgrade_notebook(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_workbench_spec,
        resolved_image=n1_image,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_pod(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    n1_workbench_spec: WorkbenchImageSpec,
    n1_notebook: Notebook,
) -> Pod:
    """Ready pod for the current IDE's upgrade workbench."""
    return get_ready_upgrade_notebook_pod(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        spec=n1_workbench_spec,
        notebook=n1_notebook,
    )


@pytest.fixture(scope="session")
def n1_statefulset(
    unprivileged_client: DynamicClient,
    n1_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet owned by the current IDE's Notebook CR."""
    return StatefulSet(
        client=unprivileged_client,
        name=n1_notebook.name,
        namespace=n1_notebook.namespace,
    )


@pytest.fixture(scope="session")
def n1_baseline(
    pytestconfig: pytest.Config,
    n1_image_baseline_configmap: ConfigMap,
    n1_workbench_spec: WorkbenchImageSpec,
    n1_notebook: Notebook,
    n1_pod: Pod,
    n1_image: ResolvedWorkbenchImage,
) -> WorkbenchImageBaseline:
    """Pre/post-upgrade baseline for the current IDE's workbench."""
    return capture_or_load_workbench_baseline(
        pytestconfig=pytestconfig,
        config_map=n1_image_baseline_configmap,
        spec=n1_workbench_spec,
        notebook=n1_notebook,
        pod=n1_pod,
        resolved_image=n1_image,
    )


@pytest.fixture(scope="session")
def n1_kernel_id(
    pytestconfig: pytest.Config,
    n1_workbench_spec: WorkbenchImageSpec,
    n1_notebook: Notebook,
    n1_pod: Pod,
    n1_baseline: WorkbenchImageBaseline,
    n1_image_baseline_configmap: ConfigMap,
) -> str:
    """Start a Jupyter kernel pre-upgrade; return its ID for post-upgrade verification."""
    if n1_workbench_spec.ide != "jupyterlab":
        pytest.skip("Kernel state test is only applicable to JupyterLab workbenches")

    cm_key = f"{n1_workbench_spec.baseline_prefix}_kernel_id"

    if pytestconfig.option.post_upgrade:
        data = dict(n1_image_baseline_configmap.instance.data or {})
        kernel_id = data.get(cm_key)
        if not kernel_id:
            pytest.skip("No kernel_id stored in baseline ConfigMap — pre-upgrade kernel test was not run")
        return kernel_id

    kernel_id = start_kernel_and_set_variable(
        pod=n1_pod,
        container_name=n1_workbench_spec.notebook_name,
        namespace=n1_notebook.namespace,
        notebook_name=n1_notebook.name,
    )
    current_data = dict(n1_image_baseline_configmap.instance.data or {})
    current_data[cm_key] = kernel_id
    ResourceEditor(patches={n1_image_baseline_configmap: {"data": current_data}}).update()
    return kernel_id


def _workbench_case_fixture(
    ide: str,
    admin_client: DynamicClient,
    workbench_upgrade_track: str,
    pytestconfig: pytest.Config,
) -> WorkbenchImageSpec:
    """Return a workbench case or skip when the IDE is unsupported on this cluster."""
    spec = get_workbench_image_spec_by_ide(ide=ide)
    if skip_reason := should_skip_workbench_spec(
        admin_client=admin_client,
        spec=spec,
        post_upgrade=pytestconfig.option.post_upgrade,
        workbench_upgrade_track=workbench_upgrade_track,
    ):
        pytest.skip(skip_reason)
    return spec


def _workbench_image_fixture(
    admin_client: DynamicClient,
    spec: WorkbenchImageSpec,
    pytestconfig: pytest.Config,
) -> ResolvedWorkbenchImage:
    """Resolve the N-1 image for a workbench case."""
    resolved_image = resolve_workbench_image(admin_client=admin_client, spec=spec)
    if (
        not pytestconfig.option.post_upgrade
        and spec.require_eus_track
        and not is_legacy_track_tag(tag_name=resolved_image.tag_name)
    ):
        pytest.skip(f"{spec.ide} workbench survival tests require a legacy EUS workbench image tag")
    return resolved_image


@pytest.fixture(scope="session")
def n1_jupyter_elyra_case(
    admin_client: DynamicClient,
    workbench_upgrade_track: str,
    pytestconfig: pytest.Config,
) -> WorkbenchImageSpec:
    """Metadata for the JupyterLab Elyra upgrade-survival scenario."""
    return _workbench_case_fixture(
        ide="jupyter-elyra",
        admin_client=admin_client,
        workbench_upgrade_track=workbench_upgrade_track,
        pytestconfig=pytestconfig,
    )


@pytest.fixture(scope="session")
def n1_jupyter_elyra_image(
    admin_client: DynamicClient,
    n1_jupyter_elyra_case: WorkbenchImageSpec,
    pytestconfig: pytest.Config,
) -> ResolvedWorkbenchImage:
    """Resolved pre-upgrade JupyterLab Elyra image."""
    return _workbench_image_fixture(
        admin_client=admin_client,
        spec=n1_jupyter_elyra_case,
        pytestconfig=pytestconfig,
    )


@pytest.fixture(scope="session")
def n1_jupyter_elyra_persistent_volume_claim(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_jupyter_elyra_case: WorkbenchImageSpec,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC backing the JupyterLab Elyra upgrade workbench."""
    yield from manage_upgrade_persistent_volume_claim(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_jupyter_elyra_case,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_jupyter_elyra_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_jupyter_elyra_case: WorkbenchImageSpec,
    n1_jupyter_elyra_image: ResolvedWorkbenchImage,
    n1_jupyter_elyra_persistent_volume_claim: PersistentVolumeClaim,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR pinned to the JupyterLab Elyra survival image."""
    del n1_jupyter_elyra_persistent_volume_claim
    yield from manage_upgrade_notebook(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_jupyter_elyra_case,
        resolved_image=n1_jupyter_elyra_image,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_jupyter_elyra_pod(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    n1_jupyter_elyra_case: WorkbenchImageSpec,
    n1_jupyter_elyra_notebook: Notebook,
) -> Pod:
    """Ready pod for the JupyterLab Elyra upgrade workbench."""
    return get_ready_upgrade_notebook_pod(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        spec=n1_jupyter_elyra_case,
        notebook=n1_jupyter_elyra_notebook,
    )


@pytest.fixture(scope="session")
def n1_jupyter_elyra_statefulset(
    unprivileged_client: DynamicClient,
    n1_jupyter_elyra_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet owned by the JupyterLab Elyra Notebook CR."""
    return StatefulSet(
        client=unprivileged_client,
        name=n1_jupyter_elyra_notebook.name,
        namespace=n1_jupyter_elyra_notebook.namespace,
    )


@pytest.fixture(scope="session")
def n1_jupyter_elyra_baseline(
    pytestconfig: pytest.Config,
    n1_image_baseline_configmap: ConfigMap,
    n1_jupyter_elyra_case: WorkbenchImageSpec,
    n1_jupyter_elyra_notebook: Notebook,
    n1_jupyter_elyra_pod: Pod,
    n1_jupyter_elyra_image: ResolvedWorkbenchImage,
) -> WorkbenchImageBaseline:
    """Pre/post-upgrade baseline for the JupyterLab Elyra workbench."""
    return capture_or_load_workbench_baseline(
        pytestconfig=pytestconfig,
        config_map=n1_image_baseline_configmap,
        spec=n1_jupyter_elyra_case,
        notebook=n1_jupyter_elyra_notebook,
        pod=n1_jupyter_elyra_pod,
        resolved_image=n1_jupyter_elyra_image,
    )


def _bump_jupyterlab_spec() -> WorkbenchImageSpec:
    """Return a WorkbenchImageSpec for the JupyterLab dashboard bump scenario."""
    is_upstream = py_config.get("distribution") == "upstream"
    return WorkbenchImageSpec(
        ide="jupyterlab",
        imagestream_name="jupyter-minimal-notebook" if is_upstream else "s2i-minimal-notebook",
        notebook_name="upgrade-bump-jlab",
        baseline_prefix="bump_jlab",
        pvc_name="upgrade-bump-jlab-storage",
    )


@pytest.fixture(scope="session")
def n1_bump_spec() -> WorkbenchImageSpec:
    """Metadata for the JupyterLab dashboard image bump scenario."""
    return _bump_jupyterlab_spec()


@pytest.fixture(scope="session")
def n1_bump_image(
    admin_client: DynamicClient,
    n1_bump_spec: WorkbenchImageSpec,
) -> ResolvedWorkbenchImage:
    """Resolved pre-upgrade (N-1) JupyterLab image for the bump workbench."""
    return resolve_workbench_image(admin_client=admin_client, spec=n1_bump_spec)


@pytest.fixture(scope="session")
def n1_bump_pvc(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_bump_spec: WorkbenchImageSpec,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC backing the dashboard image bump workbench."""
    yield from manage_upgrade_persistent_volume_claim(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_bump_spec,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_bump_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_bump_spec: WorkbenchImageSpec,
    n1_bump_image: ResolvedWorkbenchImage,
    n1_bump_pvc: PersistentVolumeClaim,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR pinned to the N-1 JupyterLab image for the bump test."""
    del n1_bump_pvc
    yield from manage_upgrade_notebook(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_bump_spec,
        resolved_image=n1_bump_image,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_bump_pod(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    n1_bump_spec: WorkbenchImageSpec,
    n1_bump_notebook: Notebook,
) -> Pod:
    """Ready pod for the bump workbench."""
    return get_ready_upgrade_notebook_pod(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        spec=n1_bump_spec,
        notebook=n1_bump_notebook,
    )


@pytest.fixture(scope="session")
def n1_bump_baseline(
    pytestconfig: pytest.Config,
    n1_image_baseline_configmap: ConfigMap,
    n1_bump_spec: WorkbenchImageSpec,
    n1_bump_notebook: Notebook,
    n1_bump_pod: Pod,
    n1_bump_image: ResolvedWorkbenchImage,
) -> WorkbenchImageBaseline:
    """Pre/post-upgrade baseline for the bump workbench."""
    return capture_or_load_workbench_baseline(
        pytestconfig=pytestconfig,
        config_map=n1_image_baseline_configmap,
        spec=n1_bump_spec,
        notebook=n1_bump_notebook,
        pod=n1_bump_pod,
        resolved_image=n1_bump_image,
    )


@pytest.fixture(scope="session")
def n1_bump_marker_written(
    pytestconfig: pytest.Config,
    n1_bump_spec: WorkbenchImageSpec,
    n1_bump_pod: Pod,
    n1_image_baseline_configmap: ConfigMap,
) -> str:
    """Write a marker file to PVC pre-upgrade; return marker content for post-upgrade comparison."""
    marker_key = f"{n1_bump_spec.baseline_prefix}_pvc_marker"

    if pytestconfig.option.post_upgrade:
        data = dict(n1_image_baseline_configmap.instance.data or {})
        if marker_key not in data:
            raise AssertionError(f"PVC marker content not found in baseline ConfigMap (key: {marker_key})")
        return data[marker_key]

    write_pvc_upgrade_marker(pod=n1_bump_pod, container_name=n1_bump_spec.notebook_name)
    marker_content = UPGRADE_MARKER_CONTENT
    current_data = dict(n1_image_baseline_configmap.instance.data or {})
    current_data[marker_key] = marker_content
    ResourceEditor(patches={n1_image_baseline_configmap: {"data": current_data}}).update()
    return marker_content


@pytest.fixture(scope="session")
def n1_bump_target_image(
    admin_client: DynamicClient,
    n1_bump_spec: WorkbenchImageSpec,
) -> ResolvedWorkbenchImage:
    """Post-upgrade (N) image to bump the workbench to."""
    return resolve_current_image(
        admin_client=admin_client,
        imagestream_name=n1_bump_spec.imagestream_name,
    )
