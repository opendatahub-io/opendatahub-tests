import subprocess
from typing import List

from kubernetes.dynamic import DynamicClient
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.pod import Pod

from utilities.constants import Timeout
from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


def get_lmevaljob_pod(client: DynamicClient, lmevaljob: LMEvalJob, timeout: int = Timeout.TIMEOUT_2MIN) -> Pod:
    """
    Gets the pod corresponding to a given LMEvalJob and waits for it to be ready.

    Args:
        client: The Kubernetes client to use
        lmevaljob: The LMEvalJob that the pod is associated with
        timeout: How long to wait for the pod, defaults to TIMEOUT_2MIN

    Returns:
        Pod resource
    """
    lmeval_pod = Pod(
        client=client,
        namespace=lmevaljob.namespace,
        name=lmevaljob.name,
    )

    lmeval_pod.wait(timeout=timeout)

    return lmeval_pod

def get_lmeval_tasks() -> List:
    """
    Gets the list of LM-Eval tasks

    Returns:
        List of LM-Eval task names
    """
    result = subprocess.check_output(["lm_eval", "--tasks", "list"])
    lines = result.decode("utf-8").splitlines()[3:]
    lmeval_tasks = []
    for line in lines:
        if '|' not in line or '---' in line:
            continue
        parts = line.split('|')
        if len(parts) > 1:
            task = parts[1].strip()
            if task:
                lmeval_tasks.append(task)

    return lmeval_tasks
    