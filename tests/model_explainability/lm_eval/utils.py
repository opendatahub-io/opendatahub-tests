from typing import List

from kubernetes.dynamic import DynamicClient
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.pod import Pod

from utilities.constants import Timeout
from simple_logger.logger import get_logger

import pandas as pd


LOGGER = get_logger(name=__name__)


def get_lmevaljob_pod(client: DynamicClient, lmevaljob: LMEvalJob, timeout: int = Timeout.TIMEOUT_10MIN) -> Pod:
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
    Gets the list of supported LM-Eval tasks that have above a certain number of minimum downloads on HuggingFace.

    Args:
        min_downloads: The minimum number of downloads

    Returns:
        List of LM-Eval task names
    """
    if min_downloads < 1:
        raise ValueError("Minimum downloads must be greater than 0")

    lmeval_tasks = pd.read_csv(filepath_or_buffer="tests/model_explainability/lm_eval/data/new_task_list.csv")

    # filter for tasks that either exceed (min_downloads OR exist on the OpenLLM leaderboard)
    # AND exist on LMEval AND do not include image data

    filtered_df = lmeval_tasks[
        lmeval_tasks["Exists"]
        & (lmeval_tasks["Dataset"] != "MMMU/MMMU")
        & ((lmeval_tasks["HF dataset downloads"] >= min_downloads) | (lmeval_tasks["OpenLLM leaderboard"]))
    ]

    # group tasks by dataset and extract the task with shortest name in the group
    unique_tasks = filtered_df.loc[filtered_df.groupby("Dataset")["Name"].apply(lambda x: x.str.len().idxmin())]

    unique_tasks = unique_tasks["Name"].tolist()

    LOGGER.info(f"Number of unique LMEval tasks with more than {min_downloads} downloads: {len(unique_tasks)}")

    return unique_tasks
