from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.lm_eval_job import LMEvalJob
from ocp_resources.pod import Pod

from tests.model_explainability.utils import validate_pod_image_against_tai_configmap_images_and_check_digest
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


def verify_lmeval_pod_images(lmeval_pod: Pod, tai_operator_configmap: ConfigMap) -> None:
    """Verifies LMEval Pod images.

    Args:
        lmeval_pod: LMEval Pod
        tai_operator_configmap: TrustyAI operator configmap

    Returns:
        None

    Raises:
        AssertionError: if pod images don't match or don't have a sha256 digest.
    """
    validate_pod_image_against_tai_configmap_images_and_check_digest(
        pod=lmeval_pod, tai_operator_configmap=tai_operator_configmap, include_init_containers=True
    )
