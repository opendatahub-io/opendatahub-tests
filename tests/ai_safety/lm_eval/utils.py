import re

import pandas as pd
import structlog
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.pod import Pod
from pyhelper_utils.general import tts
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.lm_eval.constants import (
    CA_BUNDLE_MOUNT_PATH,
    CA_BUNDLE_VOLUME_NAME,
    MERGED_CA_BUNDLE_KEY,
    MERGED_CA_CONFIGMAP_SUFFIX,
)
from utilities.exceptions import (
    PodLogMissMatchError,
    UnexpectedFailureError,
    UnexpectedResourceCountError,
)
from utilities.general import collect_pod_information

LOGGER = structlog.get_logger(name=__name__)


def get_lmevaljob_pod(client: DynamicClient, lmevaljob: LMEvalJob, timeout: int = 600) -> Pod:
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

    lmeval_tasks = pd.read_csv(filepath_or_buffer="tests/ai_safety/lm_eval/data/new_task_list.csv")

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


def validate_ca_bundle_injected(pod: Pod, job_name: str) -> None:
    """Assert the pod has CA bundle volume, mount, and REQUESTS_CA_BUNDLE env var.

    Also verifies the merged CA ConfigMap exists in the namespace with the expected key.

    Args:
        pod: The LMEvalJob pod to inspect.
        job_name: The name of the LMEvalJob (used to derive the merged ConfigMap name).
    """
    merged_cm_name = f"{job_name}{MERGED_CA_CONFIGMAP_SUFFIX}"

    volume_names = [volume.name for volume in pod.instance.spec.volumes]
    assert CA_BUNDLE_VOLUME_NAME in volume_names, (
        f"Expected volume '{CA_BUNDLE_VOLUME_NAME}' not found. Volumes: {volume_names}"
    )

    ca_volume = next(volume for volume in pod.instance.spec.volumes if volume.name == CA_BUNDLE_VOLUME_NAME)
    assert ca_volume.configMap is not None, "CA bundle volume must reference a ConfigMap"
    assert ca_volume.configMap.name == merged_cm_name, (
        f"Expected ConfigMap '{merged_cm_name}', got '{ca_volume.configMap.name}'"
    )

    main_container = pod.instance.spec.containers[0]
    mount_names = [mount.name for mount in main_container.volumeMounts]
    assert CA_BUNDLE_VOLUME_NAME in mount_names, (
        f"Expected volume mount '{CA_BUNDLE_VOLUME_NAME}' not found. Mounts: {mount_names}"
    )

    ca_mount = next(mount for mount in main_container.volumeMounts if mount.name == CA_BUNDLE_VOLUME_NAME)
    assert ca_mount.mountPath == CA_BUNDLE_MOUNT_PATH
    assert ca_mount.subPath == MERGED_CA_BUNDLE_KEY
    assert ca_mount.readOnly is True

    env_map = {env_var.name: env_var.value for env_var in (main_container.env or []) if env_var.value is not None}
    assert "REQUESTS_CA_BUNDLE" in env_map, "REQUESTS_CA_BUNDLE env var not found"
    assert env_map["REQUESTS_CA_BUNDLE"] == CA_BUNDLE_MOUNT_PATH

    merged_cm = ConfigMap(
        client=pod.client,
        name=merged_cm_name,
        namespace=pod.namespace,
        ensure_exists=True,
    )
    assert merged_cm.exists, f"Merged CA ConfigMap '{merged_cm_name}' does not exist"
    assert MERGED_CA_BUNDLE_KEY in merged_cm.instance.data, (
        f"Key '{MERGED_CA_BUNDLE_KEY}' not found in merged CA ConfigMap"
    )


def validate_ca_bundle_not_injected(pod: Pod, job_name: str) -> None:
    """Assert the pod has no CA bundle volume, REQUESTS_CA_BUNDLE env var, or merged ConfigMap.

    Args:
        pod: The LMEvalJob pod to inspect.
        job_name: The name of the LMEvalJob (used to derive the merged ConfigMap name).
    """
    volume_names = [volume.name for volume in pod.instance.spec.volumes]
    assert CA_BUNDLE_VOLUME_NAME not in volume_names, (
        f"Unexpected CA bundle volume '{CA_BUNDLE_VOLUME_NAME}' found on pod"
    )

    main_container = pod.instance.spec.containers[0]
    env_names = [env_var.name for env_var in (main_container.env or [])]
    assert "REQUESTS_CA_BUNDLE" not in env_names, "Unexpected REQUESTS_CA_BUNDLE env var found on pod"

    merged_cm_name = f"{job_name}{MERGED_CA_CONFIGMAP_SUFFIX}"
    merged_cm = ConfigMap(
        client=pod.client,
        name=merged_cm_name,
        namespace=pod.namespace,
    )
    assert not merged_cm.exists, f"Unexpected merged CA ConfigMap '{merged_cm_name}' found in namespace"


def wait_for_lmevaljob_state(
    lmevaljob: LMEvalJob,
    state: str,
    timeout: int = 600,
) -> None:
    """Wait for an LMEvalJob CR to reach a specific state.

    Args:
        lmevaljob: The LMEvalJob resource to watch.
        state: The target state (e.g. "Complete", "New", "Scheduled").
        timeout: Maximum time to wait in seconds.

    Raises:
        TimeoutExpiredError: If the job does not reach the expected state within the timeout.
    """
    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: lmevaljob.instance.status.state,
        ):
            if sample == state:
                return
    except TimeoutExpiredError:
        current_state = lmevaljob.instance.status.state if lmevaljob.instance.status else "unknown"
        LOGGER.error(
            f"LMEvalJob '{lmevaljob.name}' did not reach state '{state}' within {timeout}s. "
            f"Current state: {current_state}"
        )
        raise


def wait_for_vllm_model_ready(
    client: DynamicClient,
    namespace: str,
    inference_service_name: str,
    max_wait_time: int = 600,
    check_interval: int = 10,
) -> Pod:
    """Wait for vLLM model to download and be ready to serve requests.

    Args:
        client: Kubernetes dynamic client
        namespace: Namespace where the inference service is deployed
        inference_service_name: Name of the inference service
        max_wait_time: Maximum time to wait in seconds
        check_interval: Time between checks in seconds

    Returns:
        The predictor pod once model is ready

    Raises:
        ResourceNotFoundError: If no predictor pod is found
        UnexpectedFailureError: If model fails to load or pod encounters errors
    """
    LOGGER.info("Waiting for vLLM model to download and load...")

    predictor_pods = list(
        Pod.get(
            dyn_client=client,
            namespace=namespace,
            label_selector=f"serving.kserve.io/inferenceservice={inference_service_name},component=predictor",
        )
    )

    if not predictor_pods:
        raise ResourceNotFoundError(f"No predictor pod found for inference service '{inference_service_name}'.")

    if len(predictor_pods) != 1:
        raise UnexpectedResourceCountError(
            f"Expected exactly 1 predictor pod for inference service '{inference_service_name}', "
            f"but found {len(predictor_pods)}: {[pod.name for pod in predictor_pods]}"
        )

    predictor_pod = predictor_pods[0]
    LOGGER.info(f"Predictor pod: {predictor_pod.name}")

    def _check_model_ready() -> bool:
        try:
            pod_logs = predictor_pod.log(container="kserve-container")
            if "Uvicorn running on" in pod_logs or "Application startup complete" in pod_logs:
                LOGGER.info("vLLM server is running and ready!")
                return True
            else:
                LOGGER.info("Model still loading..")
                return False
        except (ApiException, OSError) as e:
            LOGGER.info(f"Could not get pod logs yet: {e}")
            return False

    try:
        for sample in TimeoutSampler(
            wait_timeout=max_wait_time,
            sleep=check_interval,
            func=_check_model_ready,
        ):
            if sample:
                break
    except TimeoutExpiredError:
        LOGGER.error(f"vLLM pod failed to start within {max_wait_time} seconds")
        collect_pod_information(pod=predictor_pod)
        raise

    return predictor_pod
