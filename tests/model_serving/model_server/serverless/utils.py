from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout
from utilities.infra import get_pods_by_isvc_label


LOGGER = get_logger(name=__name__)


def verify_no_inference_pods(client: DynamicClient, isvc: InferenceService) -> None:
    """
    Verify that no inference pods are running for the given InferenceService.

    Args:
        client (DynamicClient): DynamicClient object
        isvc (InferenceService): InferenceService object

    Raises:
        TimeoutError: If pods are exist after the timeout.

    """
    pods = []

    try:
        pods = TimeoutSampler(
            wait_timeout=Timeout.TIMEOUT_4MIN,
            sleep=5,
            func=get_pods_by_isvc_label,
            client=client,
            isvc=isvc,
        )
        if not pods:
            return

    except TimeoutError:
        LOGGER.error(f"{[pod.name for pod in pods]} were not deleted")
        raise
