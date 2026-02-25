import re

import pandas as pd
from kubernetes.dynamic import DynamicClient
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.pod import Pod
from pyhelper_utils.general import tts
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError

from utilities.constants import Timeout
from utilities.exceptions import PodLogMissMatchError, UnexpectedFailureError

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


def get_lmeval_tasks(min_downloads: float, max_downloads: float | None = None) -> list[str]:
    """
    Gets the list of supported LM-Eval tasks that have above a certain number of minimum downloads on HuggingFace.

    Args:
        min_downloads: The minimum number of downloads or the percentile of downloads to use as a minimum
        max_downloads: The maximum number of downloads or the percentile of downloads to use as a maximum

    Returns:
        List of LM-Eval task names
    """
    if min_downloads <= 0:
        raise ValueError("Minimum downloads must be greater than 0")

    lmeval_tasks = pd.read_csv(filepath_or_buffer="tests/model_explainability/lm_eval/data/new_task_list.csv")

    if isinstance(min_downloads, float):
        if not 0 <= min_downloads <= 1:
            raise ValueError("Minimum downloads as a percentile must be between 0 and 1")
        min_downloads = lmeval_tasks["HF dataset downloads"].quantile(q=min_downloads)

    # filter for tasks that either exceed min_downloads OR exist on the OpenLLM leaderboard
    # AND exist on LMEval AND do not include image data
    filtered_df = lmeval_tasks[
        lmeval_tasks["Exists"]
        & (lmeval_tasks["Dataset"] != "MMMU/MMMU")
        & ((lmeval_tasks["HF dataset downloads"] >= min_downloads) | (lmeval_tasks["OpenLLM leaderboard"]))
    ]

    # if max_downloads is provided, filter for tasks that have less than
    # or equal to the maximum number of downloads
    if max_downloads is not None:
        if max_downloads <= 0 or max_downloads > max(lmeval_tasks["HF dataset downloads"]):
            raise ValueError("Maximum downloads must be greater than 0 and less than the maximum number of downloads")
        if isinstance(max_downloads, float):
            if not 0 <= max_downloads <= 1:
                raise ValueError("Maximum downloads as a percentile must be between 0 and 1")
            max_downloads = lmeval_tasks["HF dataset downloads"].quantile(q=max_downloads)
        filtered_df = filtered_df[filtered_df["HF dataset downloads"] <= max_downloads]

    # group tasks by dataset and extract the task with shortest name in the group
    unique_tasks = filtered_df.loc[filtered_df.groupby("Dataset")["Name"].apply(lambda x: x.str.len().idxmin())]

    unique_tasks = unique_tasks["Name"].tolist()

    LOGGER.info(f"Number of unique LMEval tasks with more than {min_downloads} downloads: {len(unique_tasks)}")

    return unique_tasks


def validate_lmeval_job_pod_and_logs(lmevaljob_pod: Pod) -> None:
    """Validate LMEval job pod success and presence of corresponding logs.

    Args:
        lmevaljob_pod: The LMEvalJob pod.

    Returns: None
    """
    pod_success_log_regex = (
        r"INFO\sdriver\supdate status: job completed\s\{\"state\":\s\{\"state\""
        r":\"Complete\",\"reason\":\"Succeeded\",\"message\":\"job completed\""
    )
    lmevaljob_pod.wait_for_status(status=lmevaljob_pod.Status.RUNNING, timeout=tts("10m"))
    try:
        lmevaljob_pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=tts("1h"))
    except TimeoutExpiredError as e:
        raise UnexpectedFailureError("LMEval job pod failed from a running state.") from e
    if not bool(re.search(pod_success_log_regex, lmevaljob_pod.log())):
        raise PodLogMissMatchError("LMEval job pod failed.")
