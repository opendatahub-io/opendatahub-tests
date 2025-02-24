import re
import shlex

from ocp_resources.pod import Pod


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
