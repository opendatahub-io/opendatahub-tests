#
# Copyright Skodjob authors.
# License: Apache License 2.0 (see the file LICENSE or http://apache.org/licenses/LICENSE-2.0.html).
#
from __future__ import annotations

import pytest

from ocp_resources.pod import Pod

from tests.workbenches.utils import get_notebook_image, load_default_notebook, step


class TestNotebook:
    """
    # description
    Verifies deployments of Notebooks via GitOps approach

    # before_test_steps
        1. Step(value="Deploy Pipelines Operator", expected="Pipelines operator is available on the cluster"),
        2. Step(value="Deploy ServiceMesh Operator", expected="ServiceMesh operator is available on the cluster"),
        3. Step(value="Deploy Serverless Operator", expected="Serverless operator is available on the cluster"),
        4. Step(value="Install ODH operator", expected="Operator is up and running and is able to serve it's operands"),
        5. Step(value="Deploy DSCI", expected="DSCI is created and ready"),
        6. Step(value="Deploy DSC", expected="DSC is created and ready"),

    # after_test_steps
        1. Step(value="Delete all created resources", expected="All created resources are removed"),
    """

    NTB_NAME: str = "test-odh-notebook"
    NTB_NAMESPACE: str = "test-odh-notebook"

    @pytest.mark.parametrize(
        "users_namespace, users_persistent_volume_claim",
        [
            pytest.param(
                {"name": NTB_NAMESPACE},
                {"name": NTB_NAME},

            )
        ],
        indirect=True,
    )
    def test_create_simple_notebook(
        self, admin_client, unprivileged_client, users_namespace, users_persistent_volume_claim
    ):
        """
        # description
        Create simple Notebook with all needed resources and see if Operator creates it properly

        # contact
        Contact(name="Jakub Stejskal", email="jstejska@redhat.com"),

        # steps
            1. Step(
                value="Create namespace for Notebook resources with proper name, labels and annotations",
                expected="Namespace is created",
            ),
            2. Step(value="Create PVC with proper labels and data for Notebook", expected="PVC is created"),
            3. Step(
                value="Create Notebook resource with Jupyter Minimal image in pre-defined namespace",
                expected="Notebook resource is created",
            ),
            4. Step(
                value="Wait for Notebook pods readiness",
                expected="Notebook pods are up and running, Notebook is in ready state",
            ),
        """
        with step("Create Notebook CR"):
            notebook_image: str = get_notebook_image(image_name="jupyter-minimal-notebook", image_tag="2024.2")
            notebook = load_default_notebook(
                dyn_client=admin_client, namespace=self.NTB_NAMESPACE, name=self.NTB_NAME, image=notebook_image
            )

        with step("Wait for Notebook pod readiness"):
            with notebook:
                pods = Pod.get(
                    dyn_client=unprivileged_client, namespace=self.NTB_NAMESPACE, label_selector=f"app={self.NTB_NAME}"
                )
                for pod in pods:
                    pod.wait_for_condition(condition="Ready", status="True")
