#
# Copyright Skodjob authors.
# License: Apache License 2.0 (see the file LICENSE or http://apache.org/licenses/LICENSE-2.0.html).
#
from __future__ import annotations

import io
import pathlib

import kubernetes.client
from kubernetes.dynamic import DynamicClient

from ocp_resources.route import Route

import pytest
from pytest_testconfig import config as py_config

from tests.workbenches.docs import TestDoc, SuiteDoc, Contact, Desc, Step
from tests.workbenches.resources import Notebook
from tests.workbenches.utils import step, PodUtils, OcpResourceManager
from utilities import constants


@SuiteDoc(
    description=Desc("Verifies deployments of Notebooks via GitOps approach"),
    before_test_steps={
        Step(value="Deploy Pipelines Operator", expected="Pipelines operator is available on the cluster"),
        Step(value="Deploy ServiceMesh Operator", expected="ServiceMesh operator is available on the cluster"),
        Step(value="Deploy Serverless Operator", expected="Serverless operator is available on the cluster"),
        Step(value="Install ODH operator", expected="Operator is up and running and is able to serve it's operands"),
        Step(value="Deploy DSCI", expected="DSCI is created and ready"),
        Step(value="Deploy DSC", expected="DSC is created and ready"),
    },
    after_test_steps={
        Step(
            value="Delete all created resources",
            expected="All created resources are removed",
        )
    },
)
class TestNotebookST:
    NTB_NAME: str = "test-odh-notebook"
    NTB_NAMESPACE: str = "test-odh-notebook"

    @pytest.mark.parametrize(
        "users_namespace",
        [
            pytest.param(
                {"name": NTB_NAMESPACE},
            )
        ],
        indirect=True,
    )
    @pytest.mark.parametrize(
        "users_persistent_volume_claim",
        [
            pytest.param(
                {"name": NTB_NAME, "namespace": NTB_NAMESPACE},
            )
        ],
        indirect=True,
    )
    @TestDoc(
        description=Desc("Create simple Notebook with all needed resources and see if Operator creates it properly"),
        contact=Contact(name="Jakub Stejskal", email="jstejska@redhat.com"),
        steps={
            Step(
                value="Create namespace for Notebook resources with proper name, labels and annotations",
                expected="Namespace is created",
            ),
            Step(value="Create PVC with proper labels and data for Notebook", expected="PVC is created"),
            Step(
                value="Create Notebook resource with Jupyter Minimal image in pre-defined namespace",
                expected="Notebook resource is created",
            ),
            Step(
                value="Wait for Notebook pods readiness",
                expected="Notebook pods are up and running, Notebook is in ready state",
            ),
        },
    )
    def test_create_simple_notebook(
        self, admin_client, unprivileged_client, users_namespace, users_persistent_volume_claim
    ):
        with OcpResourceManager() as test_frame:
            with step("Create Notebook CR"):
                notebook_image: str = _get_notebook_image("jupyter-minimal-notebook", "2024.2")
                notebook = _load_default_notebook(
                    client=unprivileged_client, namespace=self.NTB_NAMESPACE, name=self.NTB_NAME, image=notebook_image
                )
                test_frame.enter(resource=notebook)

            with step("Wait for Notebook pod readiness"):
                PodUtils.wait_for_pods_ready(
                    client=admin_client,
                    namespace_name=self.NTB_NAMESPACE,
                    label_selector=f"app={self.NTB_NAME}",
                    expect_pods_count=1,
                )


def _get_notebook_image(image_name: str, image_tag: str) -> str:
    registry_path = "image-registry.openshift-image-registry.svc:5000"
    controllers_namespace = py_config["applications_namespace"]
    if py_config.get("distribution") == "upstream":
        image_dict = {"jupyter-minimal-notebook": "jupyter-minimal-notebook"}
    else:
        image_dict = {"jupyter-minimal-notebook": "s2i-minimal-notebook"}
    return registry_path + "/" + controllers_namespace + "/" + image_dict[image_name] + ":" + image_tag


def _load_default_notebook(client: DynamicClient, namespace: str, name: str, image: str) -> Notebook:
    notebook_string = (pathlib.Path(__file__).parent / "test_data/notebook.yaml").read_text()
    notebook_string = notebook_string.replace("my-project", namespace).replace("my-workbench", name)
    # Set new Route url
    route_host: str = list(
        Route.get(client=client, name=constants.Dashboard.route_name, namespace=py_config["applications_namespace"])
    )[0].host
    notebook_string = notebook_string.replace("odh_dashboard_route", "https://" + route_host)
    # Set the correct username
    username = _get_username(client=client)
    notebook_string = notebook_string.replace("odh_user", username)
    # Replace image
    notebook_string = notebook_string.replace("notebook_image_placeholder", image)

    nb = Notebook(yaml_file=io.StringIO(notebook_string))

    return nb


def _get_username(client: DynamicClient) -> str:
    """Gets the username for the client (see kubectl -v8 auth whoami)"""
    self_subject_review_resource: kubernetes.dynamic.Resource = client.resources.get(
        api_version="authentication.k8s.io/v1", kind="SelfSubjectReview"
    )
    self_subject_review: kubernetes.dynamic.ResourceInstance = client.create(self_subject_review_resource)
    username: str = self_subject_review.status.userInfo.username
    return username
