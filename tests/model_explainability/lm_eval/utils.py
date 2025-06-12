import re
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.pod import Pod

from utilities.constants import Timeout
from simple_logger.logger import get_logger

from utilities.general import SHA256_DIGEST_PATTERN

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


def verify_lmeval_pod_images(lmeval_pod: Pod, trustyai_operator_configmap: ConfigMap) -> None:
    """Verifies LMEval Pod images.

    Args:
        lmeval_pod:
        trustyai_operator_configmap:

    Returns:
        None

    Raises:
        AssertionError: if pod images don't match or don't have a sha256 digest.
    """
    assert (
        lmeval_pod.instance.spec.initContainers[0].image
        == trustyai_operator_configmap.instance.data["lmes-driver-image"]
    ), "Invalid LMEval driver image found in pod."
    assert (
        lmeval_pod.instance.spec.containers[0].image == trustyai_operator_configmap.instance.data["lmes-pod-image"]
    ), "Invalid LMEval job image found in running pod."
    assert re.search(SHA256_DIGEST_PATTERN, lmeval_pod.instance.spec.initContainers[0].image), (
        "LMEval Driver image is not pinned using a sha256 digest."
    )
    assert re.search(SHA256_DIGEST_PATTERN, lmeval_pod.instance.spec.containers[0].image), (
        "LMEval job image is not pinned using a sha256 digest."
    )
