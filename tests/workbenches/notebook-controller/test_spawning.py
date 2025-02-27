from __future__ import annotations

import pytest

from ocp_resources.pod import Pod

from tests.workbenches.utils import get_notebook_image, load_default_notebook


class TestNotebook:
    @pytest.mark.parametrize(
        "users_namespace, users_persistent_volume_claim",
        [
            pytest.param(
                {"name": "test-odh-notebook"},
                {"name": "test-odh-notebook"},
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
        notebook_image: str = get_notebook_image(image_name="jupyter-minimal-notebook", image_tag="2024.2")
        notebook = load_default_notebook(
            dyn_client=admin_client, namespace=users_namespace.name, name=users_namespace.name, image=notebook_image
        )

        with notebook:
            pods = Pod.get(
                dyn_client=unprivileged_client,
                namespace=users_namespace.name,
                label_selector=f"app={users_namespace.name}",
            )
            assert pods, "The expected notebook pods were not found"
            for pod in pods:
                pod.wait_for_condition(condition=pod.Condition.READY, status=pod.Condition.Status.TRUE)
