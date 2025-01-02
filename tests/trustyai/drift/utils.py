import http
import json
import os
from typing import Any, Dict, List, Optional

import requests
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.trustyai_service import TrustyAIService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from tests.trustyai.constants import TIMEOUT_5MIN
from utilities.constants import MODELMESH_SERVING

LOGGER = get_logger(name=__name__)
TIMEOUT_30SEC: int = 30


class MetricValidationError(Exception):
    pass


class TrustyAIServiceRequestHandler:
    """
    Class to encapsulate the behaviors associated to the different TrustyAIService requests we make in the tests
    TODO: It will be moved to a more general file when we start using it in new tests.
    """

    def __init__(self, token: str, service: TrustyAIService, client: DynamicClient):
        self.token = token
        self.service = service
        self.headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        self.service_route = Route(
            client=client, namespace=service.namespace, name="trustyai-service", ensure_exists=True
        )

    def _send_request(
        self,
        endpoint: str,
        method: str,
        data: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        url = f"https://{self.service_route.host}{endpoint}"

        if method not in ("GET", "POST"):
            raise ValueError(f"Unsupported HTTP method: {method}")
        if method == "GET":
            return requests.get(url=url, headers=self.headers, verify=False)
        elif method == "POST":
            return requests.post(url=url, headers=self.headers, data=data, json=json, verify=False)

    def get_model_metadata(self) -> Any:
        return self._send_request(endpoint="/info", method="GET")

    def send_drift_request(
        self,
        metric_name: str,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        LOGGER.info(f"Sending request for drift metric: {metric_name}")
        return self._send_request(endpoint=f"/metrics/drift/{metric_name}", method="POST", json=json)


# TODO: Refactor code to be under utilities.inference_utils.Inference
def send_inference_request(
    token: str,
    inference_route: Route,
    data_batch: Any,
    file_path: str,
    max_retries: int = 5,
) -> None:
    """
    Send data batch to inference service with retry logic for network errors.

    Args:
        token: Authentication token
        inference_route: Route of the inference service
        data_batch: Data to be sent
        file_path: Path to the file being processed
        max_retries: Maximum number of retry attempts (default: 5)

    Returns:
        None

    Raises:
        RequestException: If all retry attempts fail
    """
    url: str = f"https://{inference_route.host}{inference_route.instance.spec.path}/infer"
    headers: Dict[str, str] = {"Authorization": f"Bearer {token}"}

    @retry(
        stop=stop_after_attempt(max_retries),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(requests.RequestException),
        before_sleep=lambda retry_state: LOGGER.warning(
            f"Retry attempt {retry_state.attempt_number} for file {file_path} after error. "
            f"Waiting {retry_state.next_action.sleep} seconds..."
        ),
    )
    def _make_request() -> None:
        response: Optional[requests.Response] = None

        try:
            response = requests.post(url=url, headers=headers, data=data_batch, verify=False, timeout=TIMEOUT_30SEC)
            response.raise_for_status()
        except requests.RequestException as e:
            if response:
                LOGGER.error(response.content)
            LOGGER.error(f"Error sending data for file: {file_path}. Error: {str(e)}")
            raise

    try:
        _make_request()
    except requests.RequestException:
        LOGGER.error(f"All {max_retries} retry attempts failed for file: {file_path}")
        raise


def get_trustyai_number_of_observations(client: DynamicClient, token: str, trustyai_service: TrustyAIService) -> int:
    handler = TrustyAIServiceRequestHandler(token=token, service=trustyai_service, client=client)
    model_metadata: requests.Response = handler.get_model_metadata()

    if not model_metadata:
        return 0

    try:
        metadata_json: Any = model_metadata.json()

        if not metadata_json:
            return 0

        model_key: str = next(iter(metadata_json))
        model = metadata_json.get(model_key)
        if not model:
            raise KeyError(f"Model data not found for key: {model_key}")

        if observations := model.get("data", {}).get("observations"):
            return observations

        raise KeyError("Observations data not found in model metadata")
    except Exception as e:
        raise TypeError(f"Failed to parse response: {str(e)}")


def wait_for_trustyai_to_register_inference_request(
    client: DynamicClient, token: str, trustyai_service: TrustyAIService, expected_observations: int
) -> None:
    current_observations: int = get_trustyai_number_of_observations(
        client=client, token=token, trustyai_service=trustyai_service
    )

    samples = TimeoutSampler(
        wait_timeout=TIMEOUT_30SEC,
        sleep=1,
        func=lambda: current_observations == expected_observations,
    )
    for sample in samples:
        if sample:
            return


def send_inference_requests_and_verify_trustyai_service(
    client: DynamicClient,
    token: str,
    data_path: str,
    trustyai_service: TrustyAIService,
    inference_service: InferenceService,
) -> None:
    """
    Sends all the data batches present in a given directory to an InferenceService, and verifies that TrustyAIService has registered the observations.

    Args:
        client (DynamicClient): The client instance for making API calls.
        token (str): Authentication token for API access.
        data_path (str): Directory path containing data batch files.
        trustyai_service (TrustyAIService): TrustyAIService that will register the model.
        inference_service (InferenceService): Model to be registered by TrustyAI.
    """
    inference_route: Route = Route(client=client, namespace=inference_service.namespace, name=inference_service.name)

    for root, _, files in os.walk(data_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            with open(file_path, "r") as file:
                data = file.read()

            current_observations = get_trustyai_number_of_observations(
                client=client, token=token, trustyai_service=trustyai_service
            )
            send_inference_request(token=token, inference_route=inference_route, data_batch=data, file_path=file_path)
            wait_for_trustyai_to_register_inference_request(
                client=client,
                token=token,
                trustyai_service=trustyai_service,
                expected_observations=current_observations + json.loads(data)["inputs"][0]["shape"][0],
            )


def wait_for_modelmesh_pods_registered_by_trustyai(client: DynamicClient, namespace: Namespace) -> None:
    """
    Check if all the ModelMesh pods in a given namespace are ready and have been registered by the TrustyAIService in that same namespace.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        namespace (Namespace): The namespace where ModelMesh pods and TrustyAIService are deployed.
    """

    def _check_pods_ready_with_env() -> bool:
        modelmesh_pods: List[Pod] = [
            pod
            for pod in Pod.get(client=client, namespace=namespace)
            if pod.labels.get("modelmesh-service") == MODELMESH_SERVING
        ]

        found_pod_with_env: bool = False

        for pod in modelmesh_pods:
            try:
                has_env_var = False
                # Check containers for environment variable
                for container in pod.instance.spec.containers:
                    if container.env is not None and any(env.name == "MM_PAYLOAD_PROCESSORS" for env in container.env):
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


def verify_metric_request(
    client: DynamicClient, trustyai_service: TrustyAIService, token: str, metric_name: str, json_data: Any
) -> None:
    """
    Sends a metric request to a TrustyAIService and validates the response.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        trustyai_service (TrustyAIService): The TrustyAI service instance to interact with.
        token (str): Authentication token for the service.
        metric_name (str): Name of the metric to request.
        json_data (Any): JSON payload for the metric request.

    Raise:
        AssertionError if some of the response fields does not have the expected value.
    """

    response = TrustyAIServiceRequestHandler(token=token, service=trustyai_service, client=client).send_drift_request(
        metric_name=metric_name, json=json_data
    )
    LOGGER.info(msg=f"TrustyAI metric request response: {json.dumps(json.loads(response.text), indent=2)}")
    response_data = json.loads(response.text)

    errors = []

    if response.status_code != http.HTTPStatus.OK:
        errors.append(f"Unexpected status code: {response.status_code}")
    if response_data.get("timestamp", "") == "":
        errors.append("Timestamp is empty")
    if (metric_type := response_data.get("type", "")) != "metric":
        errors.append(f"Incorrect type: {metric_type or 'None'}")
    if response_data.get("value", "") == "":
        errors.append("Value is empty")
    if not isinstance(response_data.get("value"), float):
        errors.append("Value must be a float")
    if response_data.get("specificDefinition", "") == "":
        errors.append("Specific definition is empty")
    if (response_metric_name := response_data.get("name", "")) != metric_name:
        errors.append(f"Wrong name: {response_metric_name or 'None'}, expected: {metric_name}")
    if response_data.get("id", "") == "":
        errors.append("ID is empty")
    if response_data.get("thresholds", "") == "":
        errors.append("Thresholds are empty")

    if errors:
        raise MetricValidationError("\n".join(errors))
