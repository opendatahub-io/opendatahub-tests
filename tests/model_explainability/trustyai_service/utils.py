from http import HTTPStatus
import json
import os
from typing import Any, Optional

import requests
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.route import Route
from ocp_resources.trustyai_service import TrustyAIService
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout, Protocols
from utilities.exceptions import MetricValidationError
from utilities.general import create_isvc_label_selector_str
from utilities.inference_utils import UserInference, Inference


LOGGER = get_logger(name=__name__)
TIMEOUT_30SEC: int = 30


class TrustyAIServiceMetrics:
    class Fairness:
        BASE_URL = "/metrics/group/fairness"
        SPD: str = "spd"

    class Drift:
        BASE_URL = "/metrics/drift"
        MEANSHIFT: str = "meanshift"


def _get_metric_base_url(metric_name: str) -> str:
    if hasattr(TrustyAIServiceMetrics.Fairness, metric_name.upper()):
        base_url: str = TrustyAIServiceMetrics.Fairness.BASE_URL
    elif hasattr(TrustyAIServiceMetrics.Drift, metric_name.upper()):
        base_url = TrustyAIServiceMetrics.Drift.BASE_URL
    else:
        raise MetricValidationError(f"Unknown metric: {metric_name}")
    return f"{base_url}/{metric_name}"


class TrustyAIServiceClient:
    """
    Class to encapsulate the behaviors associated to the different TrustyAIService requests used in the tests
    """

    class Endpoints:
        INFO: str = "info"  # Endpoint used to get model metadata
        DATA_UPLOAD: str = "data/upload"  # Endpoint used to upload data to TrustyAIService
        REQUEST: str = (
            "request"  # Endpoint used to schedule a recurrent metric calculation, or to delete a scheduled metric
        )
        REQUESTS: str = "requests"  # Endpoint used to get all scheduled metrics for a given metric type
        INFO_NAMES: str = "info/names"  # Endpoint used to apply name mappings

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
        data: Optional[str] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        url = f"https://{self.service_route.host}/{endpoint}"

        if method not in ("GET", "POST", "DELETE"):
            raise ValueError(f"Unsupported HTTP method: {method}")
        if method == "GET":
            return requests.get(url=url, headers=self.headers, verify=False)
        elif method == "POST":
            return requests.post(url=url, headers=self.headers, data=data, json=json, verify=False)
        elif method == "DELETE":
            return requests.delete(url=url, headers=self.headers, json=json, verify=False)

    def get_model_metadata(self) -> requests.Response:
        return self._send_request(endpoint=self.Endpoints.INFO, method="GET")

    def upload_data(
        self,
        data_path: str,
    ) -> requests.Response:
        with open(data_path, "r") as file:
            data = file.read()

        LOGGER.info(f"Uploading data to TrustyAIService: {data_path}")
        return self._send_request(endpoint=self.Endpoints.DATA_UPLOAD, method="POST", data=data)

    def apply_name_mappings(
        self, model_name: str, input_mappings: dict[str, str], output_mappings: dict[str, str]
    ) -> requests.Response:
        mappings: dict[str, Any] = {
            "modelId": model_name,
            "inputMapping": input_mappings,
            "outputMapping": output_mappings,
        }

        LOGGER.info(f"Applying name mappings: {mappings}")
        return self._send_request(endpoint=self.Endpoints.INFO_NAMES, method="POST", json=mappings)

    def request_metric(
        self,
        metric_name: str,
        json: Optional[dict[str, Any]] = None,
        schedule: bool = False,
    ) -> requests.Response:
        endpoint: str = f"/{_get_metric_base_url(metric_name=metric_name)}/{self.Endpoints.REQUEST if schedule else ''}"
        LOGGER.info(f"Sending request for metric {metric_name} to endpoint {endpoint}")
        return self._send_request(endpoint=endpoint, method="POST", json=json)

    def get_metrics(
        self,
        metric_name: str,
    ) -> requests.Response:
        endpoint: str = f"{_get_metric_base_url(metric_name=metric_name)}/{self.Endpoints.REQUESTS}"
        LOGGER.info(f"Sending request to get drift metrics to endpoint {endpoint}")
        return self._send_request(
            endpoint=endpoint,
            method="GET",
        )

    def delete_metric(self, metric_name: str, request_id: str) -> requests.Response:
        endpoint: str = f"{_get_metric_base_url(metric_name=metric_name)}/{self.Endpoints.REQUEST}"
        LOGGER.info(f"Sending request to delete {metric_name} metric {request_id} to endpoint {endpoint}")
        json_payload = {"requestId": request_id}
        return self._send_request(endpoint=endpoint, method="DELETE", json=json_payload)


def get_trustyai_number_of_observations(client: DynamicClient, token: str, trustyai_service: TrustyAIService) -> int:
    handler = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)
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
        LOGGER.error(f"Failed to parse response: {str(e)}")
        raise


def send_inference_and_verify_trustyai_registered(
    token: str,
    inference_service: InferenceService,
    data_batch: str,
    client: DynamicClient,
    trustyai_service: TrustyAIService,
    expected_observations: int,
    inference_config: dict[str, Any],
    inference_type: str = Inference.INFER,
    protocol: str = Protocols.HTTPS,
) -> None:
    inference = UserInference(
        inference_service=inference_service,
        inference_config=inference_config,
        inference_type=inference_type,
        protocol=protocol,
    )

    res = inference.run_inference_flow(
        model_name=inference_service.name,
        inference_input=data_batch,
        use_default_query=False,
        token=token,
    )
    LOGGER.debug(f"Inference response: {res}")

    samples = TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_5MIN,
        sleep=1,
        func=lambda: get_trustyai_number_of_observations(client=client, token=token, trustyai_service=trustyai_service),
    )

    for obs in samples:
        if obs >= expected_observations:
            return

    raise AssertionError(f"Observations not updated. Current: {obs}, Expected: {expected_observations}")


def send_inference_requests_and_verify_trustyai_service(
    client: DynamicClient,
    token: str,
    data_path: str,
    trustyai_service: TrustyAIService,
    inference_service: InferenceService,
    inference_config: dict[str, Any],
    inference_type: str = Inference.INFER,
    protocol: str = Protocols.HTTPS,
) -> None:
    """
    Sends all the data batches present in a given directory to an InferenceService, and verifies that
    TrustyAIService has registered the observations.

    Args:
        client (DynamicClient): The client instance for making API calls.
        token (str): Authentication token for API access.
        data_path (str): Directory path containing data batch files.
        trustyai_service (TrustyAIService): TrustyAIService that will register the model.
        inference_service (InferenceService): Model to be registered by TrustyAI.
        inference_config (dict[str, Any]): Inference config to be used when sending the inference.
        inference_type (Optional[str]): Inference type to be used when sending the inference
        protocol (Optional[str]): Protocol to be used when sending the inference
    """

    for root, _, files in os.walk(data_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            with open(file_path, "r") as file:
                data = file.read()

            current_observations = get_trustyai_number_of_observations(
                client=client, token=token, trustyai_service=trustyai_service
            )
            expected_observations: int = current_observations + json.loads(data)[0]["shape"][0]

            send_inference_and_verify_trustyai_registered(
                token=token,
                inference_service=inference_service,
                data_batch=data,
                client=client,
                trustyai_service=trustyai_service,
                expected_observations=expected_observations,
                inference_config=inference_config,
                inference_type=inference_type,
                protocol=protocol,
            )


def wait_for_isvc_deployment_registered_by_trustyaiservice(
    client: DynamicClient, isvc: InferenceService, trustyai_service: TrustyAIService, runtime_name: str
) -> None:
    """
    Check if all the ModelMesh pods in a given namespace are
    ready and have been registered by the TrustyAIService in that same namespace.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        isvc (InferenceService): The InferenceService related to the deployment.
        runtime_name (str): The name of the serving runtime of the isvc
    """
    label_selector = create_isvc_label_selector_str(isvc=isvc, resource_type="deployment", runtime_name=runtime_name)

    def _check_deployments_ready() -> bool:
        deployments = list(
            Deployment.get(
                label_selector=label_selector,
                client=client,
                namespace=isvc.namespace,
            )
        )

        if not deployments:
            return False

        for deployment in deployments:
            if (
                deployment.instance.metadata.annotations.get("internal.serving.kserve.io/logger-sink-url")
                == f"http://{trustyai_service.name}.{isvc.namespace}.svc.cluster.local"
            ):
                deployment.wait_for_replicas()
            else:
                if deployment.instance.spec.replicas != 0:
                    return False

        return True

    samples = TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_10MIN,
        sleep=1,
        func=_check_deployments_ready,
    )
    for sample in samples:
        if sample:
            return


def validate_trustyai_response(
    response: Any,
    response_data: dict[str, Any],
    expected_values: Optional[dict[str, Any]] = None,
    required_fields: Optional[list[str]] = None,
) -> list[str]:
    """
    Validates a TrustyAI service response against common criteria.

    Args:
        response: The HTTP response object
        response_data: The parsed JSON response data
        expected_values: Dictionary of field names and their expected values
        required_fields: List of fields that should not be empty

    Returns:
        list: List of error messages found during validation
    """
    errors = []

    # Validate HTTP status
    if response.status_code != HTTPStatus.OK:
        errors.append(f"Unexpected status code: {response.status_code}")

    # Validate required non-empty fields
    if required_fields:
        for field in required_fields:
            if field in response_data and response_data[field] == "":
                errors.append(f"{field.capitalize()} is empty")

    # Validate expected values
    if expected_values:
        for field, expected in expected_values.items():
            if field in response_data:
                actual = response_data.get(field)
                if isinstance(actual, str) and isinstance(expected, str):
                    if actual.lower() != expected.lower():
                        errors.append(f"Wrong {field}: {actual or 'None'}, expected: {expected}")
                else:
                    if actual != expected:
                        errors.append(f"Wrong {field}: {actual or 'None'}, expected: {expected}")

    return errors


def verify_trustyai_metric_request(
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
        MetricValidationError if some of the response fields does not have the expected value.
    """
    response = TrustyAIServiceClient(token=token, service=trustyai_service, client=client).request_metric(
        metric_name=metric_name, json=json_data
    )

    LOGGER.info(msg=f"TrustyAI metric request response: {json.dumps(json.loads(response.text), indent=2)}")
    response_data = json.loads(response.text)

    required_fields = ["timestamp", "value", "specificDefinition", "id", "thresholds"]
    expected_values = {"type": "metric", "name": metric_name}

    errors = validate_trustyai_response(
        response=response, response_data=response_data, expected_values=expected_values, required_fields=required_fields
    )

    if errors:
        raise MetricValidationError("\n".join(errors))


def verify_trustyai_metric_scheduling_request(
    client: DynamicClient, trustyai_service: TrustyAIService, token: str, metric_name: str, json_data: Any
) -> None:
    """
    Schedules a metric request with the TrustyAI service and validates both the scheduling response
    and subsequent metrics retrieval.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        trustyai_service (TrustyAIService): The TrustyAI service instance to interact with.
        token (str): Authentication token for the service.
        metric_name (str): Name of the metric to schedule.
        json_data (Any): JSON payload for the metric scheduling request.

    Raises:
        MetricValidationError: If the scheduling response or metrics retrieval response contain invalid
            or unexpected values, including empty required fields or mismatched request IDs.
    """
    handler = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)
    response = handler.request_metric(
        metric_name=metric_name,
        json=json_data,
        schedule=True,
    )

    response_data = json.loads(response.text)
    LOGGER.info(msg=f"TrustyAI metric scheduling request response: {response_data}")

    required_fields = ["requestId", "timestamp"]
    errors = validate_trustyai_response(response=response, response_data=response_data, required_fields=required_fields)

    request_id = response_data.get("requestId", "")

    # Get and validate metrics
    get_metrics_response = handler.get_metrics(metric_name=metric_name)
    get_metrics_data = json.loads(get_metrics_response.text)
    LOGGER.info(msg=f"TrustyAI scheduled metrics: {get_metrics_data}")

    metrics_errors = validate_trustyai_response(response=get_metrics_response, response_data=get_metrics_data)
    errors.extend(metrics_errors)

    # Validate metrics-specific requirements
    if "requests" not in get_metrics_data or not get_metrics_data["requests"]:
        errors.append("No requests found in metrics response")
    elif len(get_metrics_data["requests"]) != 1:
        errors.append(f"Expected exactly 1 request, got {len(get_metrics_data['requests'])}")
    else:
        metrics_request_id = get_metrics_data["requests"][0]["id"]
        if metrics_request_id != request_id:
            errors.append(f"Request ID mismatch. Expected: {request_id}, Got: {metrics_request_id}")

    if errors:
        raise MetricValidationError("\n".join(errors))


def verify_upload_data_to_trustyai_service(
    client: DynamicClient,
    token: str,
    trustyai_service: TrustyAIService,
    data_path: str,
) -> None:
    """
    Uploads data to the TrustyAI service and verifies the number of observations increased correctly.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        trustyai_service (TrustyAIService): The TrustyAI service instance to interact with.
        token (str): Authentication token for the service.
        data_path (str): Path to the data file to be uploaded.
    """

    with open(data_path, "r") as file:
        data = file.read()

    expected_num_observations: int = (
        get_trustyai_number_of_observations(client=client, token=token, trustyai_service=trustyai_service)
        + json.loads(data)["request"]["inputs"][0]["shape"][0]
    )

    response = TrustyAIServiceClient(token=token, service=trustyai_service, client=client).upload_data(
        data_path=data_path
    )
    assert response.status_code == HTTPStatus.OK

    actual_num_observations: int = get_trustyai_number_of_observations(
        client=client, token=token, trustyai_service=trustyai_service
    )
    assert expected_num_observations >= actual_num_observations


def verify_trustyai_metric_delete_request(
    client: DynamicClient, trustyai_service: TrustyAIService, token: str, metric_name: str
) -> None:
    """
    Deletes a metric request from the TrustyAI service and verifies that the deletion was successful.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        trustyai_service (TrustyAIService): The TrustyAI service instance to interact with.
        token (str): Authentication token for the service.
        metric_name (str): Name of the metric to delete.

    Raises:
        ValueError: If there are no metrics to delete.
        AssertionError: If the deletion request fails or the number of metrics after deletion is not as expected.
    """
    handler = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)

    metrics_response = handler.get_metrics(metric_name=metric_name)
    metrics_data = json.loads(metrics_response.text)
    initial_num_metrics: int = len(metrics_data.get("requests", []))

    if initial_num_metrics < 1:
        raise ValueError(f"No metrics found for {metric_name}. Cannot perform deletion.")

    request_id: str = metrics_data["requests"][0]["id"]

    delete_response = handler.delete_metric(metric_name=metric_name, request_id=request_id)

    assert delete_response.status_code == HTTPStatus.OK, (
        f"Delete request failed with status code: {delete_response.status_code}"
    )

    # Verify the number of metrics after deletion is N-1
    updated_metrics_response = handler.get_metrics(metric_name=metric_name)
    updated_metrics_data = json.loads(updated_metrics_response.text)
    updated_num_metrics: int = len(updated_metrics_data.get("requests", []))

    assert updated_num_metrics == initial_num_metrics - 1, (
        f"Number of metrics after deletion is {updated_num_metrics}, expected {initial_num_metrics - 1}"
    )


def verify_name_mappings(
    client: DynamicClient,
    token: str,
    trustyai_service: TrustyAIService,
    isvc: InferenceService,
    input_mappings: dict[str, str],
    output_mappings: dict[str, str],
) -> None:
    """
    Verifies that input and output name mappings are correctly applied to the TrustyAI service.

    Args:
        client: Kubernetes dynamic client instance
        token: Authentication token used
        trustyai_service: TrustyAI service instance
        isvc: InferenceService instance to verify mappings for
        input_mappings: Dictionary mapping input names
        output_mappings: Dictionary mapping output names

    Raises:
        AssertionError: If mappings don't match expected values
    """
    tas_client: TrustyAIServiceClient = TrustyAIServiceClient(client=client, token=token, service=trustyai_service)
    response: requests.Response = tas_client.apply_name_mappings(
        model_name=isvc.name, input_mappings=input_mappings, output_mappings=output_mappings
    )
    assert response.status_code == HTTPStatus.OK
    response = tas_client.get_model_metadata()

    metadata = json.loads(response.text)
    model_data = metadata[isvc.name]["data"]

    response_input_mappings = model_data["inputSchema"]["nameMapping"]
    assert response_input_mappings == input_mappings, (
        f"Input mappings mismatch. Expected: {input_mappings}, Got: {response_input_mappings}"
    )

    response_output_mappings = model_data["outputSchema"]["nameMapping"]
    assert response_output_mappings == output_mappings, (
        f"Output mappings mismatch. Expected: {output_mappings}, Got: {response_output_mappings}"
    )
