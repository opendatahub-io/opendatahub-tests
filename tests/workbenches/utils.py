#
# Copyright Skodjob authors.
# License: Apache License 2.0 (see the file LICENSE or http://apache.org/licenses/LICENSE-2.0.html).
#
from __future__ import annotations

import io
import pathlib
from contextlib import contextmanager
from typing import Generator

from pytest_testconfig import config as py_config

from kubernetes.dynamic import DynamicClient, Resource, ResourceInstance

from ocp_resources.route import Route
from ocp_resources.notebook import Notebook

from utilities.constants import INTERNAL_IMAGE_REGISTRY_PATH


def get_notebook_image(image_name: str, image_tag: str) -> str:
    controllers_namespace = py_config["applications_namespace"]
    if py_config.get("distribution") == "upstream":
        image_dict = {"jupyter-minimal-notebook": "jupyter-minimal-notebook"}
    else:
        image_dict = {"jupyter-minimal-notebook": "s2i-minimal-notebook"}
    return f"{INTERNAL_IMAGE_REGISTRY_PATH}/{controllers_namespace}/{image_dict[image_name]}:{image_tag}"


def load_default_notebook(dyn_client: DynamicClient, namespace: str, name: str, image: str) -> Notebook:
    notebook_string = (pathlib.Path(__file__).parent / "notebook-controller/test_data/notebook.yaml").read_text()
    notebook_string = notebook_string.replace("my-project", namespace).replace("my-workbench", name)
    # Set new Route url
    route_name = "odh-dashboard" if py_config.get("distribution") == "upstream" else "rhods-dashboard"
    route_host: str = list(
        Route.get(dyn_client=dyn_client, name=route_name, namespace=py_config["applications_namespace"])
    )[0].host
    notebook_string = notebook_string.replace("odh_dashboard_route", "https://" + route_host)
    # Set the correct username
    username = get_username(dyn_client=dyn_client)
    notebook_string = notebook_string.replace("odh_user", username)
    # Replace image
    notebook_string = notebook_string.replace("notebook_image_placeholder", image)

    return Notebook(yaml_file=io.StringIO(notebook_string))


def get_username(dyn_client: DynamicClient) -> str:
    """Gets the username for the client (see kubectl -v8 auth whoami)"""
    self_subject_review_resource: Resource = dyn_client.resources.get(
        api_version="authentication.k8s.io/v1", kind="SelfSubjectReview"
    )
    self_subject_review: ResourceInstance = dyn_client.create(self_subject_review_resource)
    username: str = self_subject_review.status.userInfo.username
    return username


@contextmanager
def step(description: str) -> Generator[None, None, None]:
    yield
