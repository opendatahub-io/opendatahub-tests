import re
import shlex

from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import retry

from tests.model_serving.model_server.kserve.multi_node.constants import HEAD_POD_ROLE, SUPPORTED_ROLES, WORKER_POD_ROLE
from utilities.constants import Timeout
from utilities.infra import get_pods_by_isvc_label

LOGGER = get_logger(name=__name__)


def verify_ray_status(pods: list[Pod]) -> None:
    """
    Verify ray status is correct

    Args:
        pods (list[Pod]): pods to verify

    Raises:
        AssertionError: If ray status is not correct

    """
    cmd = shlex.split("ray status")
    ray_failures: dict[str, list[str]] = {}
    res = None
    for pod in pods:
        res = pod.execute(command=cmd)
        if res_regex := re.search(
            r"Active:\n(?P<active>.*)\nPending:\n(?P<pending>.*)\nRecent.*CPU\n(?P<gpu>.*)GPU",
            res,
            re.IGNORECASE | re.DOTALL,
        ):
            ray_formatted_result = res_regex.groupdict()
            if len(ray_formatted_result["active"].split("\n")) != len(pods):
                ray_failures.setdefault(pod.name, []).append("Wrong number of active nodes")

            if "no pending nodes" not in ray_formatted_result["pending"]:
                ray_failures.setdefault(pod.name, []).append("Some nodes are pending")

            if (gpus := ray_formatted_result["gpu"].strip().split("/")) and gpus[0] != gpus[1]:
                ray_failures.setdefault(pod.name, []).append("Wrong number of GPUs")

    assert not ray_failures, f"Failure in ray status check: {ray_failures}, {res}"


def verify_nvidia_gpu_status(pod: Pod) -> None:
    """
    Verify nvidia-smi status is correct

    Args:
        pod (Pod): pod to verify

    Raises:
        AssertionError: If nvidia-smi status is not correct

    """
    res = pod.execute(command=shlex.split("nvidia-smi --query-gpu=memory.used --format=csv"))
    mem_regex = re.search(r"(\d+)", res)

    if not mem_regex:
        raise ValueError(f"Could not find memory usage in response, {res}")

    elif mem_regex and int(mem_regex.group(1)) == 0:
        raise ValueError(f"GPU memory is not used, {res}")


def delete_multi_node_pod_by_role(client: DynamicClient, isvc: InferenceService, role: str) -> None:
    """
    Delete multi node pod by role

    Worker pods have {WORKER_POD_ROLE} str in their name, head pod does not have an identifier in the name.

    Args:
        client (DynamicClient): Dynamic client
        isvc (InferenceService): InferenceService object
        role (str): pod role

    """
    if role not in SUPPORTED_ROLES:
        raise ValueError(f"Role {role} is not supported; supported roles are {SUPPORTED_ROLES}")

    pods = get_pods_by_isvc_label(client=client, isvc=isvc)

    for pod in pods:
        if (
            role == WORKER_POD_ROLE
            and WORKER_POD_ROLE in pod.name
            or role == HEAD_POD_ROLE
            and WORKER_POD_ROLE not in pod.name
        ):
            pod.delete(wait=True)


@retry(wait_timeout=Timeout.TIMEOUT_2MIN, sleep=5)
def get_pods_by_isvc_generation(client: DynamicClient, isvc: InferenceService) -> list[Pod]:
    """
    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService):InferenceService object.

    Returns:
            list[Pod]: A list of all matching pods

    Raises:
            ResourceNotFoundError: if no pods are found.

    """
    isvc_generation = str(isvc.instance.metadata.generation)

    if pods := list(
        Pod.get(
            client=client,
            namespace=isvc.namespace,
            label_selector=f"isvc.generation={isvc_generation}",
        )
    ):
        return pods

    raise ResourceNotFoundError(f"InferenceService {isvc.name} generation {isvc_generation} has no pods")


def is_arg_in_model_spec(client: DynamicClient, isvc: InferenceService, arg: str) -> bool:
    """
    Check if arg is in model spec; spec.model.args are only added to head pod

    Args:
        client (DynamicClient): OCP Client to use.
        isvc (InferenceService): InferenceService object.
        arg (str): arg to check

    Returns:
        bool: True if arg is in model spec, False otherwise

    Raises:
        ResourceNotFoundError if no head pod found.

    """
    for pod in get_pods_by_isvc_generation(client=client, isvc=isvc):
        if WORKER_POD_ROLE not in pod.name:
            return arg in pod.instance.spec.containers[0].args

    raise ResourceNotFoundError(f"InferenceService {isvc.name} has no head pod")
