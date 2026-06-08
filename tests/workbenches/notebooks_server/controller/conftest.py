from typing import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from tests.workbenches.notebooks_server.controller.utils import (
    build_notebook_dict,
    get_dashboard_route_host,
    get_username,
    resolve_notebook_image,
)
from utilities import constants

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="function")
def users_persistent_volume_claim(
    request: pytest.FixtureRequest, unprivileged_model_namespace: Namespace, unprivileged_client: DynamicClient
) -> Generator[PersistentVolumeClaim, None, None]:
    with PersistentVolumeClaim(
        client=unprivileged_client,
        name=request.param["name"],
        namespace=unprivileged_model_namespace.name,
        label={constants.Labels.OpenDataHub.DASHBOARD: "true"},
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        size="10Gi",
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
    ) as pvc:
        yield pvc


@pytest.fixture(scope="function")
def minimal_image() -> Generator[str, None, None]:
    """Provides a full image name of a minimal workbench image (name:tag only, no registry prefix)."""
    image_name = "jupyter-minimal-notebook" if py_config.get("distribution") == "upstream" else "s2i-minimal-notebook"
    image_tag = py_config.get("workbench_image_tag", "2025.2")

    yield f"{image_name}:{image_tag}"


@pytest.fixture(scope="function")
def default_notebook(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    unprivileged_client: DynamicClient,
    minimal_image: str,
) -> Generator[Notebook, None, None]:
    """Returns a new Notebook CR for a given namespace, name, and image"""
    namespace = request.param["namespace"]
    name = request.param["name"]

    oauth_annotations = request.param.get("oauth_annotations", {})

    route_host = get_dashboard_route_host(admin_client=admin_client)

    username = get_username(client=unprivileged_client)
    assert username, "Failed to determine username from the cluster"

    image_path = resolve_notebook_image(admin_client=admin_client)

    notebook_dict = build_notebook_dict(
        namespace=namespace,
        name=name,
        image_path=image_path,
        route_host=route_host,
        username=username,
        extra_annotations=oauth_annotations or None,
    )

    with Notebook(kind_dict=notebook_dict) as nb:
        yield nb
