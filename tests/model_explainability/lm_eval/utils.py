from typing import List
from collections import Counter

from kubernetes.dynamic import DynamicClient
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.pod import Pod

from utilities.constants import Timeout
from simple_logger.logger import get_logger

import pandas as pd


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

def get_lmeval_tasks(min_downloads: int = 10000) -> List[str]:
    """
    Gets the list of unique LM-Eval tasks that have above a certain number of downloads.

    Args:
        min_downloads: The minimum number of downloads

    Returns:
        List of LM-Eval task names
    """
    if min_downloads < 1:
        raise ValueError("Min downloads must be greater than 0")
    # TO-DO: update list of tasks since some of them don't exist
    lmeval_tasks = pd.read_csv("tests/model_explainability/lm_eval/data/lm_eval_tasks.csv")

    # filter rows by download count
    filtered_df = lmeval_tasks[lmeval_tasks['HF dataset downloads'] >= min_downloads]

    def get_group_names(task_names: List[str]):
        # count how many times each base prefix appears
        group_counts = Counter(task.split('_')[0] for task in task_names)

        unique_tasks = set()

        for name in task_names:
            group = name.split('_')[0]
            if group_counts[group] > 1:
                # if multiple tasks share this base, keep only the base prefix once
                unique_tasks.add(group)
            else:
                # if base prefix unique, keep the full name
                unique_tasks.add(name)

        return unique_tasks

    group_names = list(get_group_names(filtered_df['Name']))
    LOGGER.info(f"Number of unique LMEval tasks with more than {min_downloads} downloads: {len(group_names)}")

    return group_names
