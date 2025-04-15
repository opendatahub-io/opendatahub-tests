import pytest

from kubernetes.dynamic.client import DynamicClient

from ocp_resources.pod import Pod
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim


class TestNotebook:
    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook",
        [
            pytest.param(
                {
                    "name": "test-odh-notebook",
                    "add-dashboard-label": True,
                },
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
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ):
        """
        Create a simple Notebook CR with all necessary resources and see if the Notebook Operator creates it properly
        """
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=default_notebook.namespace,
            name=f"{default_notebook.name}-0",
        )
        notebook_pod.wait()
        notebook_pod.wait_for_condition(condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE)
