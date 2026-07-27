from typing import Any

import pytest
from ocp_resources.namespace import Namespace
from ocp_resources.notebook import Notebook
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.pod import Pod

from utilities.constants import Timeout


class TestNotebookResourceLimits:
    """Verify that resource requests/limits on the Notebook CR propagate to the pod."""

    @pytest.mark.tier1
    @pytest.mark.parametrize(
        "unprivileged_model_namespace,users_persistent_volume_claim,default_notebook,notebook_pod",
        [
            pytest.param(
                {
                    "name": "test-nb-resources-a",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-resources-a"},
                {
                    "namespace": "test-nb-resources-a",
                    "name": "test-nb-resources-a",
                    "resources": {
                        "limits": {"cpu": "200m", "memory": "256Mi"},
                        "requests": {"cpu": "100m", "memory": "128Mi"},
                    },
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_resources_profile_a",
            ),
            pytest.param(
                {
                    "name": "test-nb-resources-b",
                    "add-dashboard-label": True,
                },
                {"name": "test-nb-resources-b"},
                {
                    "namespace": "test-nb-resources-b",
                    "name": "test-nb-resources-b",
                    "resources": {
                        "limits": {"cpu": "400m", "memory": "512Mi"},
                        "requests": {"cpu": "200m", "memory": "256Mi"},
                    },
                },
                {"timeout": Timeout.TIMEOUT_2MIN},
                id="test_resources_profile_b",
            ),
        ],
        indirect=True,
    )
    def test_notebook_pod_resources_match_spec(
        self,
        notebook_pod: Pod,
        unprivileged_model_namespace: Namespace,
        users_persistent_volume_claim: PersistentVolumeClaim,
        default_notebook: Notebook,
    ) -> None:
        """Verify notebook pod container resources match the Notebook CR spec.

        Given a Notebook CR is created with explicit resource requests and limits,
        When the controller reconciles and the pod becomes Ready,
        Then the notebook container's resources should exactly match the CR spec.
        """
        notebook_container = self._find_container_by_name(
            containers=notebook_pod.instance.spec.containers, name=default_notebook.name
        )
        assert notebook_container, (
            f"Notebook container '{default_notebook.name}' not found in pod. "
            f"Available: {[c.name for c in notebook_pod.instance.spec.containers]}"
        )

        expected_container = self._find_container_by_name(
            containers=default_notebook.instance.spec.template.spec.containers, name=default_notebook.name
        )
        assert expected_container, f"Notebook container '{default_notebook.name}' not found in CR spec."
        expected_resources = expected_container.resources
        assert expected_resources, (
            f"Notebook CR container '{default_notebook.name}' has no resources in spec — "
            f"test parametrization or build_notebook_dict is broken"
        )
        expected_limits = dict(expected_resources.get("limits", {}) or {})
        expected_requests = dict(expected_resources.get("requests", {}) or {})

        actual_resources = notebook_container.resources
        assert actual_resources, (
            f"Pod container '{default_notebook.name}' has no resources — "
            f"controller did not propagate resources from CR to pod"
        )
        actual_limits = dict(actual_resources.get("limits", {}) or {})
        actual_requests = dict(actual_resources.get("requests", {}) or {})

        assert actual_limits.get("cpu") == expected_limits.get("cpu"), (
            f"CPU limit mismatch: expected '{expected_limits.get('cpu')}', got '{actual_limits.get('cpu')}'"
        )
        assert actual_limits.get("memory") == expected_limits.get("memory"), (
            f"Memory limit mismatch: expected '{expected_limits.get('memory')}', got '{actual_limits.get('memory')}'"
        )
        assert actual_requests.get("cpu") == expected_requests.get("cpu"), (
            f"CPU request mismatch: expected '{expected_requests.get('cpu')}', got '{actual_requests.get('cpu')}'"
        )
        assert actual_requests.get("memory") == expected_requests.get("memory"), (
            f"Memory request mismatch: expected '{expected_requests.get('memory')}', "
            f"got '{actual_requests.get('memory')}'"
        )

    @staticmethod
    def _find_container_by_name(containers: list[Any], name: str) -> Any | None:
        """Find a container by name in a list of container specs."""
        return next((c for c in containers if c.name == name), None)
