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

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook",
        [
            pytest.param(
                {
                    "name": "test-oauth-notebook",
                    "add-dashboard-label": True,
                },
                {"name": "test-oauth-notebook"},
                {
                    "namespace": "test-oauth-notebook",
                    "name": "test-oauth-notebook",
                    "oauth_annotations": {
                        "notebooks.opendatahub.io/auth-sidecar-cpu-request": "200m",
                        "notebooks.opendatahub.io/auth-sidecar-memory-request": "128Mi",
                        "notebooks.opendatahub.io/auth-sidecar-cpu-limit": "500m",
                        "notebooks.opendatahub.io/auth-sidecar-memory-limit": "256Mi",
                    },
                },
            )
        ],
        indirect=True,
    )
    def test_oauth_container_resource_customization(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ):
        """
        Test that OAuth container resource requests and limits can be customized using annotations.

        This test verifies that when a Notebook CR is created with custom OAuth container resource
        annotations, the spawned pod has the OAuth container with the specified resource values.
        """
        notebook_pod = Pod(
            client=unprivileged_client,
            namespace=default_notebook.namespace,
            name=f"{default_notebook.name}-0",
        )
        notebook_pod.wait()
        notebook_pod.wait_for_condition(condition=Pod.Condition.READY, status=Pod.Condition.Status.TRUE)

        # Verify OAuth container has the expected resource values
        oauth_container = self._get_oauth_container(pod=notebook_pod)
        assert oauth_container, "OAuth proxy container not found in the pod"

        # Check CPU request
        assert oauth_container.resources.requests["cpu"] == "200m", (
            f"Expected CPU request '200m', got '{oauth_container.resources.requests['cpu']}'"
        )

        # Check memory request
        assert oauth_container.resources.requests["memory"] == "128Mi", (
            f"Expected memory request '128Mi', got '{oauth_container.resources.requests['memory']}'"
        )

        # Check CPU limit
        assert oauth_container.resources.limits["cpu"] == "500m", (
            f"Expected CPU limit '500m', got '{oauth_container.resources.limits['cpu']}'"
        )

        # Check memory limit
        assert oauth_container.resources.limits["memory"] == "256Mi", (
            f"Expected memory limit '256Mi', got '{oauth_container.resources.limits['memory']}'"
        )

    def _get_oauth_container(self, pod: Pod):
        """
        Find and return the OAuth proxy container from the pod spec.

        Args:
            pod: The pod instance to search

        Returns:
            The OAuth container if found, None otherwise
        """
        containers = pod.instance.spec.containers
        for container in containers:
            if container.name == "oauth-proxy":
                return container
        return None
