import pytest

from timeout_sampler import TimeoutExpiredError

from ocp_resources.pod import Pod


class TestNotebook:
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
        unprivileged_client,
        unprivileged_model_namespace,
        users_persistent_volume_claim,
        default_notebook,
    ):
        """
        Create a simple Notebook CR with all necessary resources and see if the Notebook Operator creates it properly
        """
        pods = Pod.get(
            dyn_client=unprivileged_client,
            namespace=unprivileged_model_namespace.name,
            label_selector=f"app={unprivileged_model_namespace.name}",
        )
        assert pods, "The expected notebook pods were not found"

        failed_pods = []
        for pod in pods:
            try:
                pod.wait_for_condition(condition=pod.Condition.READY, status=pod.Condition.Status.TRUE)
            except TimeoutExpiredError:
                failed_pods.append(pod)
        assert not failed_pods, f"The following pods failed to get READY when starting the notebook: {failed_pods}"
