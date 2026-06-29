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

from tests.workbenches.notebook_images.utils import (
    UPGRADE_BASELINE_CM_NAME,
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
    resolve_workbench_image,
    resolve_workbench_upgrade_track,
    should_skip_workbench_spec,
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
def n1_jupyterlab_case(
    admin_client: DynamicClient,
    workbench_upgrade_track: str,
    pytestconfig: pytest.Config,
) -> WorkbenchImageSpec:
    """Metadata for the JupyterLab upgrade-survival scenario."""
    return _workbench_case_fixture(
        ide="jupyterlab",
        admin_client=admin_client,
        workbench_upgrade_track=workbench_upgrade_track,
        pytestconfig=pytestconfig,
    )


@pytest.fixture(scope="session")
def n1_jupyterlab_image(
    admin_client: DynamicClient,
    n1_jupyterlab_case: WorkbenchImageSpec,
    pytestconfig: pytest.Config,
) -> ResolvedWorkbenchImage:
    """Resolved pre-upgrade JupyterLab image."""
    return _workbench_image_fixture(
        admin_client=admin_client,
        spec=n1_jupyterlab_case,
        pytestconfig=pytestconfig,
    )


@pytest.fixture(scope="session")
def n1_jupyterlab_persistent_volume_claim(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_jupyterlab_case: WorkbenchImageSpec,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC backing the JupyterLab upgrade workbench."""
    yield from manage_upgrade_persistent_volume_claim(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_jupyterlab_case,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_jupyterlab_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_jupyterlab_case: WorkbenchImageSpec,
    n1_jupyterlab_image: ResolvedWorkbenchImage,
    n1_jupyterlab_persistent_volume_claim: PersistentVolumeClaim,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR pinned to the JupyterLab survival image."""
    # Declare the PVC dependency so storage exists before the Notebook CR is created.
    del n1_jupyterlab_persistent_volume_claim
    yield from manage_upgrade_notebook(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_jupyterlab_case,
        resolved_image=n1_jupyterlab_image,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_jupyterlab_pod(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    n1_jupyterlab_case: WorkbenchImageSpec,
    n1_jupyterlab_notebook: Notebook,
) -> Pod:
    """Ready pod for the JupyterLab upgrade workbench."""
    return get_ready_upgrade_notebook_pod(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        spec=n1_jupyterlab_case,
        notebook=n1_jupyterlab_notebook,
    )


@pytest.fixture(scope="session")
def n1_jupyterlab_statefulset(
    unprivileged_client: DynamicClient,
    n1_jupyterlab_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet owned by the JupyterLab Notebook CR."""
    return StatefulSet(
        client=unprivileged_client,
        name=n1_jupyterlab_notebook.name,
        namespace=n1_jupyterlab_notebook.namespace,
    )


@pytest.fixture(scope="session")
def n1_jupyterlab_baseline(
    pytestconfig: pytest.Config,
    n1_image_baseline_configmap: ConfigMap,
    n1_jupyterlab_case: WorkbenchImageSpec,
    n1_jupyterlab_notebook: Notebook,
    n1_jupyterlab_pod: Pod,
    n1_jupyterlab_image: ResolvedWorkbenchImage,
) -> WorkbenchImageBaseline:
    """Pre/post-upgrade baseline for the JupyterLab workbench."""
    return capture_or_load_workbench_baseline(
        pytestconfig=pytestconfig,
        config_map=n1_image_baseline_configmap,
        spec=n1_jupyterlab_case,
        notebook=n1_jupyterlab_notebook,
        pod=n1_jupyterlab_pod,
        resolved_image=n1_jupyterlab_image,
    )


@pytest.fixture(scope="session")
def n1_codeserver_case(
    admin_client: DynamicClient,
    workbench_upgrade_track: str,
    pytestconfig: pytest.Config,
) -> WorkbenchImageSpec:
    """Metadata for the Code Server upgrade-survival scenario."""
    return _workbench_case_fixture(
        ide="code-server",
        admin_client=admin_client,
        workbench_upgrade_track=workbench_upgrade_track,
        pytestconfig=pytestconfig,
    )


@pytest.fixture(scope="session")
def n1_codeserver_image(
    admin_client: DynamicClient,
    n1_codeserver_case: WorkbenchImageSpec,
    pytestconfig: pytest.Config,
) -> ResolvedWorkbenchImage:
    """Resolved pre-upgrade Code Server image."""
    return _workbench_image_fixture(
        admin_client=admin_client,
        spec=n1_codeserver_case,
        pytestconfig=pytestconfig,
    )


@pytest.fixture(scope="session")
def n1_codeserver_persistent_volume_claim(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_codeserver_case: WorkbenchImageSpec,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC backing the Code Server upgrade workbench."""
    yield from manage_upgrade_persistent_volume_claim(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_codeserver_case,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_codeserver_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_codeserver_case: WorkbenchImageSpec,
    n1_codeserver_image: ResolvedWorkbenchImage,
    n1_codeserver_persistent_volume_claim: PersistentVolumeClaim,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR pinned to the Code Server survival image."""
    # Declare the PVC dependency so storage exists before the Notebook CR is created.
    del n1_codeserver_persistent_volume_claim
    yield from manage_upgrade_notebook(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_codeserver_case,
        resolved_image=n1_codeserver_image,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_codeserver_pod(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    n1_codeserver_case: WorkbenchImageSpec,
    n1_codeserver_notebook: Notebook,
) -> Pod:
    """Ready pod for the Code Server upgrade workbench."""
    return get_ready_upgrade_notebook_pod(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        spec=n1_codeserver_case,
        notebook=n1_codeserver_notebook,
    )


@pytest.fixture(scope="session")
def n1_codeserver_statefulset(
    unprivileged_client: DynamicClient,
    n1_codeserver_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet owned by the Code Server Notebook CR."""
    return StatefulSet(
        client=unprivileged_client,
        name=n1_codeserver_notebook.name,
        namespace=n1_codeserver_notebook.namespace,
    )


@pytest.fixture(scope="session")
def n1_codeserver_baseline(
    pytestconfig: pytest.Config,
    n1_image_baseline_configmap: ConfigMap,
    n1_codeserver_case: WorkbenchImageSpec,
    n1_codeserver_notebook: Notebook,
    n1_codeserver_pod: Pod,
    n1_codeserver_image: ResolvedWorkbenchImage,
) -> WorkbenchImageBaseline:
    """Pre/post-upgrade baseline for the Code Server workbench."""
    return capture_or_load_workbench_baseline(
        pytestconfig=pytestconfig,
        config_map=n1_image_baseline_configmap,
        spec=n1_codeserver_case,
        notebook=n1_codeserver_notebook,
        pod=n1_codeserver_pod,
        resolved_image=n1_codeserver_image,
    )


@pytest.fixture(scope="session")
def n1_rstudio_case(
    admin_client: DynamicClient,
    workbench_upgrade_track: str,
    pytestconfig: pytest.Config,
) -> WorkbenchImageSpec:
    """Metadata for the legacy RStudio upgrade-survival scenario."""
    return _workbench_case_fixture(
        ide="rstudio",
        admin_client=admin_client,
        workbench_upgrade_track=workbench_upgrade_track,
        pytestconfig=pytestconfig,
    )


@pytest.fixture(scope="session")
def n1_rstudio_image(
    admin_client: DynamicClient,
    n1_rstudio_case: WorkbenchImageSpec,
    pytestconfig: pytest.Config,
) -> ResolvedWorkbenchImage:
    """Resolved pre-upgrade RStudio image."""
    return _workbench_image_fixture(
        admin_client=admin_client,
        spec=n1_rstudio_case,
        pytestconfig=pytestconfig,
    )


@pytest.fixture(scope="session")
def n1_rstudio_persistent_volume_claim(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_rstudio_case: WorkbenchImageSpec,
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC backing the RStudio upgrade workbench."""
    yield from manage_upgrade_persistent_volume_claim(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_rstudio_case,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_rstudio_notebook(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n1_notebook_namespace: Namespace,
    n1_rstudio_case: WorkbenchImageSpec,
    n1_rstudio_image: ResolvedWorkbenchImage,
    n1_rstudio_persistent_volume_claim: PersistentVolumeClaim,
    teardown_resources: bool,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR pinned to the RStudio survival image."""
    # Declare the PVC dependency so storage exists before the Notebook CR is created.
    del n1_rstudio_persistent_volume_claim
    yield from manage_upgrade_notebook(
        pytestconfig=pytestconfig,
        unprivileged_client=unprivileged_client,
        namespace_name=n1_notebook_namespace.name,
        spec=n1_rstudio_case,
        resolved_image=n1_rstudio_image,
        teardown_resources=teardown_resources,
    )


@pytest.fixture(scope="session")
def n1_rstudio_pod(
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    n1_rstudio_case: WorkbenchImageSpec,
    n1_rstudio_notebook: Notebook,
) -> Pod:
    """Ready pod for the RStudio upgrade workbench."""
    return get_ready_upgrade_notebook_pod(
        admin_client=admin_client,
        unprivileged_client=unprivileged_client,
        spec=n1_rstudio_case,
        notebook=n1_rstudio_notebook,
    )


@pytest.fixture(scope="session")
def n1_rstudio_statefulset(
    unprivileged_client: DynamicClient,
    n1_rstudio_notebook: Notebook,
) -> StatefulSet:
    """StatefulSet owned by the RStudio Notebook CR."""
    return StatefulSet(
        client=unprivileged_client,
        name=n1_rstudio_notebook.name,
        namespace=n1_rstudio_notebook.namespace,
    )


@pytest.fixture(scope="session")
def n1_rstudio_baseline(
    pytestconfig: pytest.Config,
    n1_image_baseline_configmap: ConfigMap,
    n1_rstudio_case: WorkbenchImageSpec,
    n1_rstudio_notebook: Notebook,
    n1_rstudio_pod: Pod,
    n1_rstudio_image: ResolvedWorkbenchImage,
) -> WorkbenchImageBaseline:
    """Pre/post-upgrade baseline for the RStudio workbench."""
    return capture_or_load_workbench_baseline(
        pytestconfig=pytestconfig,
        config_map=n1_image_baseline_configmap,
        spec=n1_rstudio_case,
        notebook=n1_rstudio_notebook,
        pod=n1_rstudio_pod,
        resolved_image=n1_rstudio_image,
    )
