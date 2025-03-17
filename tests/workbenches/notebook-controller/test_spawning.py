from __future__ import annotations

import pytest
from pytest_testconfig import config as py_config

from ocp_resources.pod import Pod

from utilities.constants import INTERNAL_IMAGE_REGISTRY_PATH
from tests.workbenches.utils import load_default_notebook


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
        Create simple Notebook with all needed resources and see if Operator creates it properly
        """
        image_name = (
            "jupyter-minimal-notebook" if py_config.get("distribution") == "upstream" else "s2i-minimal-notebook"
        )
        notebook_image: str = (
            f"{INTERNAL_IMAGE_REGISTRY_PATH}/{py_config['applications_namespace']}/{image_name}:{'2024.2'}"
        )
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
