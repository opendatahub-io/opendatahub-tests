import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config

EXPECTED_REPLICAS = 1
MANAGER_CONTAINER_NAME = "manager"

CONTROLLER_CONFIGS = {
    "odh-notebook-controller": {
        "label_selector": "app=odh-notebook-controller",
        "expected_containers": 1,
        "limits": {"cpu": "500m", "memory": "4Gi"},
        "requests": {"cpu": "500m", "memory": "256Mi"},
    },
    "notebook-controller": {
        "label_selector": "app=notebook-controller",
        "expected_containers": 1,
        "limits": {"cpu": "500m", "memory": "4Gi"},
        "requests": {"cpu": "500m", "memory": "256Mi"},
    },
}


pytestmark = pytest.mark.smoke


class TestNotebookControllerDeployment:
    """Verify notebook controller deployments are healthy and correctly configured."""

    @pytest.mark.parametrize(
        "controller_name",
        [
            pytest.param("odh-notebook-controller", id="test_odh_notebook_controller_replicas"),
            pytest.param("notebook-controller", id="test_kf_notebook_controller_replicas"),
        ],
    )
    def test_controller_replicas_ready(
        self,
        admin_client: DynamicClient,
        controller_name: str,
    ) -> None:
        """Verify notebook controller deployment has the expected number of ready replicas.

        Given the notebook controller deployment exists in the applications namespace,
        When checking the deployment status,
        Then the number of ready replicas should match the expected count.
        """
        controller_config = CONTROLLER_CONFIGS[controller_name]
        namespace = py_config["applications_namespace"]

        all_pods = list(
            Pod.get(
                client=admin_client,
                namespace=namespace,
                label_selector=controller_config["label_selector"],
            )
        )
        pods = [pod for pod in all_pods if not pod.instance.metadata.deletionTimestamp]

        assert len(pods) == EXPECTED_REPLICAS, (
            f"Expected {EXPECTED_REPLICAS} non-terminating pod(s) for {controller_name}, found {len(pods)}"
        )

        for pod in pods:
            pod_phase = pod.instance.status.phase
            assert pod_phase == Pod.Status.RUNNING, f"Pod {pod.name} is in phase '{pod_phase}', expected 'Running'"

    @pytest.mark.parametrize(
        "controller_name",
        [
            pytest.param("odh-notebook-controller", id="test_odh_notebook_controller_containers"),
            pytest.param("notebook-controller", id="test_kf_notebook_controller_containers"),
        ],
    )
    def test_controller_container_count(
        self,
        admin_client: DynamicClient,
        controller_name: str,
    ) -> None:
        """Verify notebook controller pod has the expected number of containers.

        Given the notebook controller pod is running,
        When inspecting its container spec,
        Then the container count should match the expected value.
        """
        controller_config = CONTROLLER_CONFIGS[controller_name]
        namespace = py_config["applications_namespace"]

        all_pods = list(
            Pod.get(
                client=admin_client,
                namespace=namespace,
                label_selector=controller_config["label_selector"],
            )
        )
        pods = [pod for pod in all_pods if not pod.instance.metadata.deletionTimestamp]
        assert pods, f"No non-terminating pods found for {controller_name} in namespace {namespace}"

        for pod in pods:
            containers = pod.instance.spec.containers
            assert len(containers) == controller_config["expected_containers"], (
                f"Pod {pod.name}: expected {controller_config['expected_containers']} "
                f"container(s), found {len(containers)}"
            )

    @pytest.mark.parametrize(
        "controller_name",
        [
            pytest.param("odh-notebook-controller", id="test_odh_notebook_controller_resources"),
            pytest.param("notebook-controller", id="test_kf_notebook_controller_resources"),
        ],
    )
    def test_controller_container_resources(
        self,
        admin_client: DynamicClient,
        controller_name: str,
    ) -> None:
        """Verify the manager container has correct CPU/memory resource requests and limits.

        Given the notebook controller pod is running,
        When inspecting the manager container's resource spec,
        Then requests and limits should match the expected values.
        """
        controller_config = CONTROLLER_CONFIGS[controller_name]
        namespace = py_config["applications_namespace"]

        all_pods = list(
            Pod.get(
                client=admin_client,
                namespace=namespace,
                label_selector=controller_config["label_selector"],
            )
        )
        pods = [pod for pod in all_pods if not pod.instance.metadata.deletionTimestamp]
        assert pods, f"No non-terminating pods found for {controller_name} in namespace {namespace}"

        for pod in pods:
            manager_container = self._find_container(pod=pod, name=MANAGER_CONTAINER_NAME)
            assert manager_container, (
                f"Pod {pod.name}: container '{MANAGER_CONTAINER_NAME}' not found. "
                f"Available: {[c.name for c in pod.instance.spec.containers]}"
            )

            resources = manager_container.resources
            limits = (resources.limits if resources else {}) or {}
            requests = (resources.requests if resources else {}) or {}

            actual_limits = {
                "cpu": limits.get("cpu"),
                "memory": limits.get("memory"),
            }
            actual_requests = {
                "cpu": requests.get("cpu"),
                "memory": requests.get("memory"),
            }

            assert actual_limits == controller_config["limits"], (
                f"Pod {pod.name}: manager container limits mismatch. "
                f"Expected {controller_config['limits']}, got {actual_limits}"
            )
            assert actual_requests == controller_config["requests"], (
                f"Pod {pod.name}: manager container requests mismatch. "
                f"Expected {controller_config['requests']}, got {actual_requests}"
            )

    @staticmethod
    def _find_container(pod: Pod, name: str):
        """Find a container by name in a pod's spec."""
        for container in pod.instance.spec.containers:
            if container.name == name:
                return container
        return None
