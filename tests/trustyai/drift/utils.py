import json
import logging
import os
import subprocess
from time import sleep
from typing import Any

import requests
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.trustyai_service import TrustyAIService
from requests import RequestException
from timeout_sampler import TimeoutSampler

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from tests.trustyai.constants import MODELMESH_SERVING, TIMEOUT_5MIN

LOGGER = logging.getLogger(__name__)
TIMEOUT_1SEC: int = 1
TIMEOUT_30SEC: int = 30

def get_ocp_token(namespace: Namespace) -> str:
    return subprocess.check_output(["oc", "create", "token", "test-user", "-n", namespace.name]).decode().strip()


def send_request_to_trustyai_service(
    client: DynamicClient, token: str, trustyai_service: TrustyAIService, endpoint: str, method: str, data: Any = None, json: Any = None
) -> Any:
    trustyai_service_route = Route(client=client, namespace=trustyai_service.namespace, name="trustyai-service", ensure_exists=True)

    url = f"https://{trustyai_service_route.host}{endpoint}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    if method == "GET":
        return requests.get(url=url, headers=headers, verify=False)
    elif method == "POST":
        return requests.post(url=url, headers=headers, data=data, json=json, verify=False)
    raise ValueError(f"Unsupported HTTP method: {method}")


def get_trustyai_model_metadata(client: DynamicClient, token: str, trustyai_service: TrustyAIService) -> Any:
    return send_request_to_trustyai_service(
        client=client,
        token=token,
        trustyai_service=trustyai_service,
        endpoint="/info",
        method="GET",
    )


def send_inference_request(
        client: DynamicClient,
        token: str,
        inference_service: InferenceService,
        data_batch: Any,
        file_path: str,
        max_retries: int = 5,
) -> None:
    """
    Send data batch to inference service with retry logic for network errors.

    Args:
        client: DynamicClient instance
        token: Authentication token
        inference_service: InferenceService instance
        data_batch: Data to be sent
        file_path: Path to the file being processed
        max_retries: Maximum number of retry attempts (default: 5)

    Returns:
        None

    Raises:
        RequestException: If all retry attempts fail
    """
    namespace = inference_service.namespace
    inference_route = Route(client=client, namespace=namespace, name=inference_service.name)

    url = f"https://{inference_route.host}{inference_route.instance.spec.path}/infer"
    headers = {"Authorization": f"Bearer {token}"}

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(RequestException),
        before_sleep=lambda retry_state: LOGGER.warning(
            f"Retry attempt {retry_state.attempt_number} for file {file_path} after error. "
            f"Waiting {retry_state.next_action.sleep} seconds..."
        )
    )
    def _make_request():
        try:
            response = requests.post(
                url=url,
                headers=headers,
                data=data_batch,
                verify=False,
                timeout=TIMEOUT_30SEC
            )
            response.raise_for_status()
            return response
        except RequestException as e:
            LOGGER.debug(response.content)
            LOGGER.error(f"Error sending data for file: {file_path}. Error: {str(e)}")
            raise

    try:
        return _make_request()
    except RequestException as e:
        LOGGER.error(f"All {max_retries} retry attempts failed for file: {file_path}")
        raise


def get_trustyai_number_of_observations(client: DynamicClient, token: str, trustyai_service: TrustyAIService) -> int:
    model_metadata = get_trustyai_model_metadata(client=client, token=token, trustyai_service=trustyai_service)

    if not model_metadata:
        return 0

    try:
        # Convert response to JSON
        metadata_json = model_metadata.json()

        # If empty JSON, return 0
        if not metadata_json:
            return 0

        # Get the first model key from the metadata
        model_key = next(iter(metadata_json))
        return metadata_json[model_key]["data"]["observations"]
    except Exception as e:
        raise TypeError(f"Failed to parse response: {str(e)}")


def get_number_of_observations_from_data_batch(data: Any) -> int:
    data_dict = json.loads(data)
    return data_dict['inputs'][0]['shape'][0]

def wait_for_trustyai_to_register_inference_request(client:DynamicClient, token:str, trustyai_service: TrustyAIService, expected_observations: int) -> None:
    current_observations = get_trustyai_number_of_observations(client=client, token=token, trustyai_service=trustyai_service)

    samples = TimeoutSampler(
        wait_timeout=TIMEOUT_30SEC,
        sleep=TIMEOUT_1SEC,
        func=lambda: current_observations == expected_observations,
    )
    for sample in samples:
        if sample:
            return


def send_inference_requests_and_verify_trustyai_service(client:DynamicClient, token: str, data_path: str, trustyai_service: TrustyAIService, inference_service: InferenceService) -> None:
    """Sends all the data batches present in a given directory to an InferenceService, and verifies that TrustyAIService has registered the observations."""

    files_processed = 0
    for root, _, files in os.walk(data_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            with open(file_path, "r") as file:
                data = file.read()

            current_observations = get_trustyai_number_of_observations(client=client, token=token, trustyai_service=trustyai_service)
            send_inference_request(client=client, token=token, inference_service=inference_service, data_batch=data, file_path=file_path)
            wait_for_trustyai_to_register_inference_request(client=client, token=token, trustyai_service=trustyai_service, expected_observations=current_observations+get_number_of_observations_from_data_batch(data))


def wait_for_modelmesh_pods_registered_by_trustyai(client:DynamicClient, namespace: Namespace):
    """Check if all the ModelMesh pods in a given namespace are ready and have been registered by the TrustyAIService in that same namespace."""

    def _check_pods_ready_with_env() -> bool:
        modelmesh_pods = [pod for pod in Pod.get(client=client, namespace=namespace) if
                          MODELMESH_SERVING in pod.name]

        found_pod_with_env = False

        for pod in modelmesh_pods:
            try:
                has_env_var = False
                # Check containers for environment variable
                for container in pod.instance.spec.containers:
                    if container.env is not None and any(
                            env.name == "MM_PAYLOAD_PROCESSORS" for env in container.env
                    ):
                        has_env_var = True
                        found_pod_with_env = True
                        break

                # If pod has env var but isn't running, return False
                if has_env_var and pod.status != Pod.Status.RUNNING:
                    return False

            except NotFoundError:
                # Ignore pods that were deleted during the process
                continue

        # Return True only if we found at least one pod with the env var
        # and all pods with the env var are running
        return found_pod_with_env

    samples = TimeoutSampler(
        wait_timeout=TIMEOUT_5MIN,
        sleep=TIMEOUT_30SEC,
        func=_check_pods_ready_with_env,
    )
    for sample in samples:
        if sample:
            return

