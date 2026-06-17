"""Fixtures for N-1 workbench image upgrade survival tests."""

from collections.abc import Generator
from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.workbenches.notebook_images.utils import (
    UPGRADE_MARKER_CONTENT,
    WorkbenchImageSpec,
    build_notebook_dict,
    merge_baseline_entry,
    notebook_service_account,
    resolve_n_minus_one_image,
    should_skip_workbench_spec,
    write_pvc_upgrade_marker,
)
from utilities import constants
from utilities.constants import Timeout
from utilities.general import collect_pod_information
from utilities.infra import create_ns

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-notebook-images"


@pytest.fixture(scope="session")
def workbench_image_spec(request: pytest.FixtureRequest) -> WorkbenchImageSpec:
    """Parametrized workbench IDE configuration."""
    return request.param


@pytest.fixture(scope="session")
def n_minus_one_namespace(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    teardown_resources: bool,
) -> Generator[Namespace, Any, Any]:
    """Namespace shared by all N-1 workbench image upgrade tests."""
    ns = Namespace(client=unprivileged_client, name=UPGRADE_NAMESPACE)
    existing_ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        if teardown_resources:
            existing_ns.clean_up()
    elif existing_ns.exists:
        LOGGER.info(f"Namespace {UPGRADE_NAMESPACE} already exists, reusing it")
        yield ns
    else:
        with create_ns(
            admin_client=admin_client,
            unprivileged_client=unprivileged_client,
            name=UPGRADE_NAMESPACE,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session", autouse=True)
def ensure_ide_supported(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    workbench_image_spec: WorkbenchImageSpec,
) -> None:
    """Skip unsupported IDE and cluster combinations before fixture setup."""
    skip_reason = should_skip_workbench_spec(
        admin_client=admin_client,
        spec=workbench_image_spec,
        post_upgrade=pytestconfig.option.post_upgrade,
    )
    if skip_reason:
        pytest.skip(skip_reason)


@pytest.fixture(scope="session")
def n_minus_one_baseline_data(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    n_minus_one_namespace: Namespace,
    workbench_image_spec: WorkbenchImageSpec,
) -> dict[str, Any]:
    """Load pre-upgrade baseline during post-upgrade runs; skip when absent."""
    if not pytestconfig.option.post_upgrade:
        return {}

    from tests.workbenches.notebook_images.utils import load_baseline_entry

    try:
        return load_baseline_entry(
            admin_client=admin_client,
            namespace=n_minus_one_namespace.name,
            notebook_name=workbench_image_spec.notebook_name,
        )
    except AssertionError as error:
        pytest.skip(f"No pre-upgrade baseline for {workbench_image_spec.ide}: {error}")


@pytest.fixture(scope="session")
def n_minus_one_image(
    admin_client: DynamicClient,
    workbench_image_spec: WorkbenchImageSpec,
) -> str:
    """Resolved N-1 image reference for the parametrized IDE."""
    return resolve_n_minus_one_image(admin_client=admin_client, spec=workbench_image_spec)


@pytest.fixture(scope="session")
def n_minus_one_pvc(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n_minus_one_namespace: Namespace,
    workbench_image_spec: WorkbenchImageSpec,
    n_minus_one_baseline_data: dict[str, Any],
    teardown_resources: bool,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    """PVC backing the N-1 workbench."""
    pvc_kwargs = {
        "client": unprivileged_client,
        "name": workbench_image_spec.notebook_name,
        "namespace": n_minus_one_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        yield PersistentVolumeClaim(**pvc_kwargs)
    else:
        existing_pvc = PersistentVolumeClaim(**pvc_kwargs)
        if existing_pvc.exists:
            LOGGER.info(f"PVC '{workbench_image_spec.notebook_name}' already exists, reusing it")
            yield existing_pvc
            return

        with PersistentVolumeClaim(
            **pvc_kwargs,
            label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
            accessmodes=PersistentVolumeClaim.AccessMode.RWO,
            size="1Gi",
            volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
            teardown=teardown_resources,
        ) as pvc:
            yield pvc


@pytest.fixture(scope="session")
def n_minus_one_notebook(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    n_minus_one_namespace: Namespace,
    n_minus_one_pvc: PersistentVolumeClaim,
    n_minus_one_baseline_data: dict[str, Any],
    workbench_image_spec: WorkbenchImageSpec,
    teardown_resources: bool,
    request: pytest.FixtureRequest,
) -> Generator[Notebook, Any, Any]:
    """Notebook CR launched on the N-1 workbench image."""
    notebook_kwargs = {
        "client": unprivileged_client,
        "name": workbench_image_spec.notebook_name,
        "namespace": n_minus_one_namespace.name,
    }

    if pytestconfig.option.post_upgrade:
        nb = Notebook(**notebook_kwargs)
        yield nb
        if teardown_resources:
            nb.client = admin_client
            nb.clean_up()
    else:
        n_minus_one_image = request.getfixturevalue(argname="n_minus_one_image")
        existing_notebook = Notebook(**notebook_kwargs)
        if existing_notebook.exists:
            annotations = existing_notebook.instance.metadata.annotations or {}
            selected_image = annotations.get("notebooks.opendatahub.io/last-image-selection")
            if selected_image == n_minus_one_image:
                LOGGER.info(f"Notebook '{workbench_image_spec.notebook_name}' already exists, reusing it")
                with notebook_service_account(
                    client=unprivileged_client,
                    name=workbench_image_spec.notebook_name,
                    namespace=n_minus_one_namespace.name,
                    teardown=False,
                ):
                    yield existing_notebook
                return

            LOGGER.warning(
                f"Notebook '{workbench_image_spec.notebook_name}' exists with image "
                f"'{selected_image}' but expected '{n_minus_one_image}'; recreating notebook"
            )
            existing_notebook.delete()
            for _ in TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_5MIN,
                sleep=5,
                func=lambda: not Notebook(**notebook_kwargs).exists,
            ):
                break

        notebook_dict = build_notebook_dict(
            namespace=n_minus_one_namespace.name,
            name=workbench_image_spec.notebook_name,
            image_path=n_minus_one_image,
        )
        with notebook_service_account(
            client=unprivileged_client,
            name=workbench_image_spec.notebook_name,
            namespace=n_minus_one_namespace.name,
            teardown=teardown_resources,
        ):
            with Notebook(client=unprivileged_client, kind_dict=notebook_dict, teardown=teardown_resources) as nb:
                yield nb


@pytest.fixture(scope="session")
def n_minus_one_pod(
    pytestconfig: pytest.Config,
    unprivileged_client: DynamicClient,
    n_minus_one_notebook: Notebook,
    workbench_image_spec: WorkbenchImageSpec,
) -> Pod:
    """Notebook pod for N-1 survival tests."""
    notebook_pod = Pod(
        client=unprivileged_client,
        namespace=n_minus_one_notebook.namespace,
        name=f"{n_minus_one_notebook.name}-0",
    )

    if pytestconfig.option.post_upgrade:
        return notebook_pod

    try:
        notebook_pod.wait()
        notebook_pod.wait_for_condition(
            condition=Pod.Condition.READY,
            status=Pod.Condition.Status.TRUE,
            timeout=Timeout.TIMEOUT_10MIN,
        )
    except (TimeoutError, TimeoutExpiredError) as error:
        if notebook_pod.exists:
            collect_pod_information(notebook_pod)
            raise AssertionError(
                f"Pod '{workbench_image_spec.notebook_name}-0' failed to reach Ready state "
                f"within {Timeout.TIMEOUT_10MIN} seconds.\nOriginal error: {error}\n"
                "Pod information collected to must-gather directory for debugging."
            ) from error

        raise AssertionError(
            f"Pod '{workbench_image_spec.notebook_name}-0' was not created. Check notebook controller logs."
        ) from error

    return notebook_pod


@pytest.fixture(scope="session")
def capture_n_minus_one_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    n_minus_one_namespace: Namespace,
    n_minus_one_notebook: Notebook,
    n_minus_one_pod: Pod,
    n_minus_one_image: str,
    workbench_image_spec: WorkbenchImageSpec,
) -> None:
    """Capture pod metadata and PVC marker before upgrade."""
    if pytestconfig.option.post_upgrade:
        return

    creation_timestamp = n_minus_one_pod.instance.metadata.creationTimestamp
    assert creation_timestamp, f"Pod '{n_minus_one_pod.name}' has no creationTimestamp"

    write_pvc_upgrade_marker(pod=n_minus_one_pod, container_name=workbench_image_spec.notebook_name)

    baseline_entry = {
        "ide": workbench_image_spec.ide,
        "image": n_minus_one_image,
        "pod_creation_timestamp": creation_timestamp,
        "upgrade_marker": UPGRADE_MARKER_CONTENT,
    }
    merge_baseline_entry(
        admin_client=admin_client,
        namespace=n_minus_one_namespace.name,
        notebook_name=workbench_image_spec.notebook_name,
        baseline_entry=baseline_entry,
    )
    LOGGER.info(f"Saved N-1 baseline for {workbench_image_spec.ide}: {baseline_entry}")


@pytest.fixture(scope="session")
def n_minus_one_baseline(n_minus_one_baseline_data: dict[str, Any]) -> dict[str, Any]:
    """Baseline values captured before upgrade."""
    return n_minus_one_baseline_data
