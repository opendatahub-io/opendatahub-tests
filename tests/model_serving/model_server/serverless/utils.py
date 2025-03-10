from typing import Any

from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import Timeout
from utilities.exceptions import InferenceCanaryTrafficError

LOGGER = get_logger(name=__name__)


def wait_for_canary_rollout(isvc: InferenceService, percentage: int, timeout: int = Timeout.TIMEOUT_5MIN) -> None:
    """
    Wait for inference service to be updated with canary rollout.

    Args:
        isvc (InferenceService): InferenceService object
        percentage (int): Percentage of canary rollout
        timeout (int): Timeout in seconds

    Raises:
        TimeoutExpired: If canary rollout is not updated

    """
    sample = None

    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: isvc.instance.status.components.predictor.get("traffic", []),
        ):
            if sample:
                for traffic_info in sample:
                    if traffic_info.get("latestRevision") and traffic_info.get("percent") == percentage:
                        return

    except TimeoutExpiredError:
        LOGGER.error(
            f"InferenceService {isvc.name} canary rollout is not updated to {percentage}. Traffic info:\n{sample}"
        )
        raise


def verify_canary_traffic(
    isvc: InferenceService,
    inference_config: dict[str, Any],
    inference_type: str,
    protocol: str,
    model_name: str,
    iterations: int,
    percentage: int,
) -> None:
    """
    Verify canary traffic percentage against inference_config.

    Args:
        isvc (InferenceService): Inference service.
        inference_config (dict[str, Any]): Inference config.
        inference_type (str): Inference type.
        protocol (str): Protocol.
        model_name (str): Model name.
        iterations (int): Number of iterations.
        percentage (int): Percentage of canary rollout.

    Raises:
        InferenceCanaryTrafficError: If canary rollout is not updated

    """
    successful_inferences = 0

    for _ in range(iterations):
        try:
            verify_inference_response(
                inference_service=isvc,
                inference_config=inference_config,
                inference_type=inference_type,
                protocol=protocol,
                model_name=model_name,
                use_default_query=True,
            )

            successful_inferences += 1

        except Exception:
            continue

    successful_inferences_percentage = successful_inferences / iterations * 100

    if successful_inferences_percentage != percentage:
        raise InferenceCanaryTrafficError(
            f"Percentage of inference requests {successful_inferences_percentage} "
            f"to the new model does not match the expected percentage {percentage}. "
        )
