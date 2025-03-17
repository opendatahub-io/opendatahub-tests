from __future__ import annotations

import pytest

from ocp_resources.pod import Pod


class TestNotebook:
    @pytest.mark.parametrize(
        "users_namespace,users_persistent_volume_claim,default_notebook",
        [
            pytest.param(
                {"name": "test-odh-notebook"},
                {"name": "test-odh-notebook"},
                {
                    "namespace": "test-odh-notebook",
                    "name": "test-odh-notebook",
                },
            )
        ],
        indirect=True,
    )
    def test_create_simple_notebook(
        self, admin_client, unprivileged_client, users_namespace, users_persistent_volume_claim, default_notebook
    ):
        """
        Create simple Notebook with all needed resources and see if Operator creates it properly
        """
        with default_notebook:
            pods = Pod.get(
                dyn_client=unprivileged_client,
                namespace=users_namespace.name,
                label_selector=f"app={users_namespace.name}",
            )
            assert pods, "The expected notebook pods were not found"
            for pod in pods:
                pod.wait_for_condition(condition=pod.Condition.READY, status=pod.Condition.Status.TRUE)
