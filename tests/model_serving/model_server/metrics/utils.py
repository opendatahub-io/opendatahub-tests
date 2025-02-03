from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger

from tests.model_serving.model_server.utils import verify_inference_response


LOGGER = get_logger(name=__name__)


def run_inference_multiple_times(
    isvc: InferenceService,
    inference_config: dict[str, Any],
    inference_type: str,
    protocol: str,
    model_name: str,
    iterations: int,
    run_in_parallel: bool = False,
) -> None:
    """
    Run inference multiple times and verify the response.

    Args:
        isvc (InferenceService): InferenceService object
        inference_config (dict[str, Any]): Inference configuration
        inference_type (str): Inference type
        protocol (str): Protocol
        model_name (str): Model name
        iterations (int): Number of iterations
        run_in_parallel (bool): Run in parallel

    """
    futures = []

    with ThreadPoolExecutor() as executor:
        for iteration in range(iterations):
            infer_kwargs = {
                "inference_service": isvc,
                "inference_config": inference_config,
                "inference_type": inference_type,
                "protocol": protocol,
                "model_name": model_name,
                "use_default_query": True,
            }

            if run_in_parallel:
                futures.append(executor.submit(verify_inference_response, **infer_kwargs))
            else:
                verify_inference_response(**infer_kwargs)

        if futures:
            for result in as_completed(futures):
                _exception = result.exception()
                if _exception:
                    LOGGER.error(f"Failed to run inference. Error: {_exception}")
