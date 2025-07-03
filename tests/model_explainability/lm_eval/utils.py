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

def get_lmeval_tasks(min_downloads: int = 10000, tier: int = 1) -> List[str]:
    """
    Gets the list of unique LM-Eval tasks that have above a certain number of downloads.

    Args:
        min_downloads: The minimum number of downloads
        tier: The tier of support offered

    Returns:
        List of LM-Eval task names
    """
    if min_downloads < 1:
        raise ValueError("Min downloads must be greater than 0")

    if tier < 1 or tier > 3:
        raise ValueError("Valid tier values are: 1, 2, or 3")

    lmeval_tasks = pd.read_csv("tests/model_explainability/lm_eval/data/new_task_list.csv")

    filtered_df = lmeval_tasks[
        (lmeval_tasks['Exists']) &
        (lmeval_tasks['Tier'] == tier) &
        (lmeval_tasks['HF dataset downloads'] >= min_downloads) &
        (lmeval_tasks['OpenLLM leaderboard'])
    ]

    LOGGER.info(f"Number of unique LMEval tasks with more than {min_downloads} downloads: {len(filtered_df)}")

    return filtered_df['Name'].tolist()
