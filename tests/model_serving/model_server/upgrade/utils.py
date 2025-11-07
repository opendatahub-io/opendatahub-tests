from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config

from utilities.exceptions import PodContainersRestartError, ResourceMismatchError
from utilities.infra import get_inference_serving_runtime


def verify_pod_containers_not_restarted(client: DynamicClient, component_name: str) -> None:
    """
    Verify pod containers not restarted.

    Args:
        client (DynamicClient): DynamicClient instance
        component_name (str): Name of the component

    Raises:
        AssertionError: If pod containers are restarted

    """
    restarted_containers = {}

    for pod in Pod.get(
        dyn_client=client,
        namespace=py_config["applications_namespace"],
        label_selector=f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of={component_name}",
    ):
        if _restarted_containers := [
            container.name for container in pod.instance.status.containerStatuses if container.restartCount > 0
        ]:
            restarted_containers[pod.name] = _restarted_containers

    if restarted_containers:
        raise PodContainersRestartError(f"Containers {restarted_containers} restarted")


def verify_inference_generation(isvc: InferenceService, expected_generation: int) -> None:
    """
    Verify that inference generation is equal to expected generation.

    Args:
        isvc (InferenceService): InferenceService instance
        expected_generation (int): Expected generation

    Raises:
        ResourceMismatch: If inference generation is not equal to expected generation
    """
    if isvc.instance.status.observedGeneration != expected_generation:
        raise ResourceMismatchError(f"Inference service {isvc.name} was modified")


def verify_serving_runtime_generation(isvc: InferenceService, expected_generation: int) -> None:
    """
    Verify that serving runtime generation is equal to expected generation.
    Args:
        isvc (InferenceService): InferenceService instance
        expected_generation (int): Expected generation

    Raises:
        ResourceMismatch: If serving runtime generation is not equal to expected generation
    """
    runtime = get_inference_serving_runtime(isvc=isvc)
    if runtime.instance.metadata.generation != expected_generation:
        raise ResourceMismatchError(f"Serving runtime {runtime.name} was modified")
