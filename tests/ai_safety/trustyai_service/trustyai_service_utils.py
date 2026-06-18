import gzip
import json
import os
from collections.abc import Callable
from http import HTTPStatus
from typing import Any

import requests
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import TimeoutSampler

from utilities.certificates_utils import create_ca_bundle_file
from utilities.constants import TRUSTYAI_SERVICE_NAME, Protocols, Timeout
from utilities.exceptions import MetricValidationError
from utilities.general import create_isvc_label_selector_str
from utilities.inference_utils import Inference, UserInference

LOGGER = structlog.get_logger(name=__name__)


class NoMetricsFoundError(ValueError):
    """Raised when no metrics are available for the requested operation."""


class TrustyAIServiceMetrics:
    class Fairness:
        SPD: str = "spd"
        DIR: str = "dir"

    class Drift:
        MEANSHIFT: str = "meanshift"
        KSTEST: str = "kstest"
        APPROXKSTEST: str = "approxkstest"
        FOURIERMMD: str = "fouriermmd"


class TrustyAIServiceClient:
    """
    A class to be used as a client to interact with TrustyAIService.
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
            client=client, namespace=service.namespace, name=TRUSTYAI_SERVICE_NAME, ensure_exists=True
        )
        self.cert_path = create_ca_bundle_file(client=client)

    def _get_metric_base_url(self, metric_name: str) -> str:
        """Gets base URL for a given metric type (fairness or drift).

        Args:
            metric_name (str): Name of the metric to get URL for.

        Returns:
            str: Base URL for the metric type.

        Raises:
            MetricValidationError: If metric_name is not a valid fairness or drift metric.
        """

        if hasattr(TrustyAIServiceMetrics.Fairness, metric_name.upper()):
            base_url: str = "metrics/group/fairness"
        elif hasattr(TrustyAIServiceMetrics.Drift, metric_name.upper()):
            base_url = "metrics/drift"
        else:
            raise MetricValidationError(f"Unknown metric: {metric_name}")
        return f"{base_url.rstrip('/')}/{metric_name}"

    def _send_request(
        self,
        endpoint: str,
        method: str,
        data: str | bytes | None = None,
        json: dict[str, Any] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> Any:
        """Sends HTTP request to specified TrustyAIService endpoint.

        Args:
            endpoint (str): API endpoint to send request to.
            method (str): HTTP method (GET, POST, DELETE).
            data (str | bytes | None): Request body data.
            json (dict[str, Any] | None): JSON data to send.
            extra_headers (dict[str, str] | None): Additional headers to merge with defaults.

        Returns:
            Any: Response from the request.

        Raises:
            ValueError: If method is not GET, POST or DELETE.
        """

        url = f"https://{self.service_route.host}/{endpoint.lstrip('/')}"
        headers = {**self.headers, **(extra_headers or {})}
        base_kwargs = {"url": url, "headers": headers, "verify": self.cert_path}

        method = method.upper()
        if method not in ("GET", "POST", "DELETE"):
            raise ValueError(f"Unsupported HTTP method: {method}")
        if method == "GET":
            return requests.get(**base_kwargs)  # type: ignore[arg-type]
        elif method == "POST":
            return requests.post(**base_kwargs, data=data, json=json)  # type: ignore[arg-type]
        elif method == "DELETE":
            return requests.delete(**base_kwargs, json=json)  # type: ignore[arg-type]

    def get_model_metadata(self) -> requests.Response:
        """Gets metadata information about the model from TrustyAIService.

        Returns:
            requests.Response: Response containing model metadata.
        """
        return self._send_request(endpoint=self.Endpoints.INFO, method="GET")

    def upload_data(
        self,
        data_path: str,
    ) -> requests.Response:
        """Uploads data file to TrustyAIService.

        Args:
            data_path (str): Path to data file to upload.

        Returns:
            requests.Response: Response from upload request.
        """

        with open(data_path, "r") as file:
            data = file.read()

        LOGGER.info(f"Uploading data to TrustyAIService: {data_path}")
        return self._send_request(endpoint=self.Endpoints.DATA_UPLOAD, method="POST", data=data)

    def upload_gzip_data(self, data_path: str) -> requests.Response:
        """Uploads gzip-compressed data to TrustyAI's /data/upload endpoint.

        Args:
            data_path (str): Path to the JSON data file to compress and upload.

        Returns:
            requests.Response: Response from upload request.
        """
        with open(data_path, "r") as file:
            raw_data = file.read().encode("utf-8")

        compressed = gzip.compress(data=raw_data)

        LOGGER.info(f"Uploading gzip-compressed data to TrustyAIService: {data_path}")
        return self._send_request(
            endpoint=self.Endpoints.DATA_UPLOAD,
            method="POST",
            data=compressed,
            extra_headers={"Content-Encoding": "gzip"},
        )

    def upload_raw_bytes(self, data: bytes, content_encoding: str = "gzip") -> requests.Response:
        """Uploads raw bytes to /data/upload with specified Content-Encoding header.

        Args:
            data (bytes): Raw bytes to send as request body.
            content_encoding (str): Value for Content-Encoding header. Defaults to "gzip".

        Returns:
            requests.Response: Response from upload request.
        """
        LOGGER.info(f"Uploading raw bytes to TrustyAIService with Content-Encoding: {content_encoding}")
        return self._send_request(
            endpoint=self.Endpoints.DATA_UPLOAD,
            method="POST",
            data=data,
            extra_headers={"Content-Encoding": content_encoding},
        )

    def apply_name_mappings(
        self, model_name: str, input_mappings: dict[str, str], output_mappings: dict[str, str]
    ) -> requests.Response:
        """Applies input and output name mappings for a model registered by TrustyAIService.

        Args:
            model_name (str): Name of model to apply mappings to.
            input_mappings (dict[str, str]): Mappings for input names.
            output_mappings (dict[str, str]): Mappings for output names.

        Returns:
            requests.Response: Response from mapping request.
        """
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
        json: dict[str, Any] | None = None,
        schedule: bool = False,
    ) -> requests.Response:
        """Requests calculation of specified metric.

        Args:
            metric_name (str): Name of metric to calculate.
            json (dict[str, Any] | None): Additional JSON parameters. Defaults to None.
            schedule (bool, optional): Whether to schedule a recurrent metric calculation. Defaults to False.

        Returns:
            requests.Response: Response from metric request.
        """
        base_url = self._get_metric_base_url(metric_name=metric_name)
        endpoint: str = f"{base_url}/{self.Endpoints.REQUEST}" if schedule else base_url
        LOGGER.info(f"Sending request for metric {metric_name} to endpoint {endpoint}")
        return self._send_request(endpoint=endpoint, method="POST", json=json)

    def get_metrics(
        self,
        metric_name: str,
    ) -> requests.Response:
        """Gets all scheduled metrics for specified metric type.

        Args:
            metric_name (str): Name of metric to retrieve.

        Returns:
            requests.Response: Response containing metrics data.
        """
        endpoint: str = f"{self._get_metric_base_url(metric_name=metric_name)}/{self.Endpoints.REQUESTS}"
        LOGGER.info(f"Sending request to get drift metrics to endpoint {endpoint}")
        return self._send_request(
            endpoint=endpoint,
            method="GET",
        )

    def delete_metric(self, metric_name: str, request_id: str) -> requests.Response:
        """Deletes specified metric request.

        Args:
            metric_name (str): Name of metric to delete.
            request_id (str): ID of specific metric request to delete.

        Returns:
            requests.Response: Response from delete request.
        """
        endpoint: str = f"{self._get_metric_base_url(metric_name=metric_name)}/{self.Endpoints.REQUEST}"
        LOGGER.info(f"Sending request to delete {metric_name} metric {request_id} to endpoint {endpoint}")
        json_payload = {"requestId": request_id}
        return self._send_request(endpoint=endpoint, method="DELETE", json=json_payload)


def get_num_observations_from_trustyai_service(
    client: DynamicClient, token: str, trustyai_service: TrustyAIService
) -> int:
    """Gets the number of observations that TrustyAIService has stored for a given model.

    Args:
        client (DynamicClient): Dynamic client instance.
        token (str): Authentication token.
        trustyai_service (TrustyAIService): TrustyAI service instance.

    Returns:
        int: Number of observations, 0 if no metadata found.

    Raises:
        KeyError: If model data or observations not found in metadata.
    """
    tas_client = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)
    model_metadata: requests.Response = tas_client.get_model_metadata()

    if not model_metadata:
        return 0

    try:
        metadata_json: Any = model_metadata.json()
        LOGGER.debug(f"TrustyAIService model metadata: {metadata_json}")

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
        LOGGER.error(f"Failed to parse response: {e!s}")
        raise


def send_inferences_and_verify_trustyai_service_registered(
    client: DynamicClient,
    token: str,
    inference_token: str,
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
        inference_type (str): Inference type to be used when sending the inference
        inference_token(str): Token to be used in the inference request
        protocol (str): Protocol to be used when sending the inference
    """
    for root, _, files in os.walk(data_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)

            with open(file_path, "r") as file:
                data = file.read()

            current_observations = get_num_observations_from_trustyai_service(
                client=client, token=token, trustyai_service=trustyai_service
            )
            expected_observations: int = current_observations + json.loads(data)[0]["shape"][0]

            inference = UserInference(
                inference_service=inference_service,
                inference_config=inference_config,
                inference_type=inference_type,
                protocol=protocol,
            )

            res = inference.run_inference_flow(
                model_name=inference_service.name,
                inference_input=data,
                use_default_query=False,
                token=inference_token,
            )
            LOGGER.debug(f"Inference response: {res}")
            samples = TimeoutSampler(
                wait_timeout=Timeout.TIMEOUT_5MIN,
                sleep=1,
                func=lambda: get_num_observations_from_trustyai_service(
                    client=client, token=token, trustyai_service=trustyai_service
                ),
            )

            for obs in samples:
                if obs >= expected_observations:
                    break
            else:
                raise AssertionError(f"Observations not updated. Current: {obs}, Expected: {expected_observations}")


def wait_for_isvc_deployment_registered_by_trustyai_service(
    client: DynamicClient, isvc: InferenceService, runtime_name: str
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
    pod_label_selector = create_isvc_label_selector_str(isvc=isvc, resource_type="pod", runtime_name=runtime_name)
    trustyai_service = TrustyAIService(name=TRUSTYAI_SERVICE_NAME, namespace=isvc.namespace, ensure_exists=True)

    expected_sink_host = f"{trustyai_service.name}.{isvc.namespace}.svc.cluster.local"

    def _get_deployments() -> list[Deployment]:
        return list(
            Deployment.get(
                label_selector=label_selector,
                client=client,
                namespace=isvc.namespace,
            )
        )

    def _get_pods() -> list[Pod]:
        return list(
            Pod.get(
                client=client,
                namespace=isvc.namespace,
                label_selector=pod_label_selector,
            )
        )

    samples = TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_20MIN,
        sleep=1,
        func=_get_deployments,
    )

    for deployments in samples:
        if not deployments:
            continue

        all_ready = True
        for deployment in deployments:
            sink_url = deployment.instance.metadata.annotations.get("internal.serving.kserve.io/logger-sink-url", "")
            if sink_url.endswith(expected_sink_host):
                deployment.wait_for_replicas()
                deployment.wait_for_condition(condition="Available", status="True")

                pods = _get_pods()
                if len(pods) != 1:
                    all_ready = False
                    break

                pod = pods[0]
                if pod.instance.status.phase != "Running":
                    all_ready = False
                    break

            elif deployment.instance.spec.replicas != 0:
                all_ready = False
                break

        if all_ready:
            return


def verify_trustyai_service_response(
    response: Any,
    response_data: dict[str, Any],
    expected_values: dict[str, Any] | None = None,
    required_fields: list[str] | None = None,
) -> None:
    """
    Validates a TrustyAI service response against common criteria.

    Args:
        response: The HTTP response object
        response_data: The parsed JSON response data
        expected_values: Dictionary of field names and their expected values
        required_fields: List of fields that should not be empty

    Raise:
        MetricValidationError if some of the response fields does not have the expected value.
    """
    errors = []

    # Validate HTTP status
    if response.status_code != HTTPStatus.OK:
        errors.append(f"Unexpected status code: {response.status_code}")

    # Validate required non-empty fields
    if required_fields:
        errors.extend([
            f"{field.capitalize()} is empty"
            for field in required_fields
            if field in response_data and response_data[field] == ""
        ])

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

    if errors:
        raise MetricValidationError("\n".join(errors))


def verify_trustyai_service_metric_request(
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
    expected_values = {"type": "metric", "name": metric_name}  # TODO: Check other fields

    verify_trustyai_service_response(
        response=response, response_data=response_data, expected_values=expected_values, required_fields=required_fields
    )


def verify_trustyai_service_metric_scheduling_request(
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
    tas_client = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)
    response = tas_client.request_metric(
        metric_name=metric_name,
        json=json_data,
        schedule=True,
    )

    response_data = json.loads(response.text)
    LOGGER.info(msg=f"TrustyAI metric scheduling request response: {response_data}")

    required_fields = ["requestId", "timestamp"]
    verify_trustyai_service_response(response=response, response_data=response_data, required_fields=required_fields)

    request_id = response_data.get("requestId", "")

    # Get and validate metrics
    get_metrics_response = tas_client.get_metrics(metric_name=metric_name)
    get_metrics_data = json.loads(get_metrics_response.text)
    LOGGER.info(msg=f"TrustyAI scheduled metrics: {get_metrics_data}")

    verify_trustyai_service_response(response=get_metrics_response, response_data=get_metrics_data)
    errors = []
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


def _verify_upload_and_check_observations(
    client: DynamicClient,
    token: str,
    trustyai_service: TrustyAIService,
    data_path: str,
    upload_fn: Callable[[], requests.Response],
) -> None:
    """Uploads data via the provided function and verifies the observation count increased.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        token (str): Authentication token for the service.
        trustyai_service (TrustyAIService): The TrustyAI service instance.
        data_path (str): Path to the data file to be uploaded.
        upload_fn: Callable that performs the upload and returns a requests.Response.
    """
    with open(data_path, "r") as file:
        data = file.read()

    expected_num_observations: int = (
        get_num_observations_from_trustyai_service(client=client, token=token, trustyai_service=trustyai_service)
        + json.loads(data)["request"]["inputs"][0]["shape"][0]
    )

    response = upload_fn()
    assert response.status_code == HTTPStatus.OK, f"Upload failed with status {response.status_code}: {response.text}"

    actual_num_observations: int = get_num_observations_from_trustyai_service(
        client=client, token=token, trustyai_service=trustyai_service
    )
    assert actual_num_observations == expected_num_observations, (
        f"Observation count mismatch after upload. Expected {expected_num_observations}, got {actual_num_observations}"
    )


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
    tas_client = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)
    _verify_upload_and_check_observations(
        client=client,
        token=token,
        trustyai_service=trustyai_service,
        data_path=data_path,
        upload_fn=lambda: tas_client.upload_data(data_path=data_path),
    )


def verify_trustyai_service_metric_registered_post_upgrade(
    client: DynamicClient, trustyai_service: TrustyAIService, token: str, metric_name: str
) -> None:
    """Verifies that a metric scheduled pre-upgrade was re-registered by TrustyAI after upgrade.

    Fails with a clear message if the metric is missing, pointing to known 3.5 breaking changes:
    referenceTag is now required and must exist in logged data; fitColumns must use original
    column names, not mapped names.
    """
    tas_client = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)
    metrics_response = tas_client.get_metrics(metric_name=metric_name)
    assert metrics_response.status_code == HTTPStatus.OK, (
        f"Failed to fetch {metric_name} metrics: "
        f"HTTP {metrics_response.status_code}. Response body: {metrics_response.text!r}"
    )
    metrics_data = json.loads(metrics_response.text)
    num_metrics = len(metrics_data.get("requests", []))

    assert num_metrics >= 1, (
        f"Pre-upgrade {metric_name} metric was not re-registered by TrustyAI after upgrade. "
        f"Known causes: (1) referenceTag is now required and must exist in logged data, "
        f"(2) fitColumns must use original column names, not mapped names. "
        f"Check TrustyAI service logs for re-registration errors."
    )


def verify_trustyai_service_metric_delete_request(
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
    tas_client = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)

    metrics_response = tas_client.get_metrics(metric_name=metric_name)
    metrics_data = json.loads(metrics_response.text)
    initial_num_metrics: int = len(metrics_data.get("requests", []))

    if initial_num_metrics < 1:
        raise NoMetricsFoundError(f"No metrics found for {metric_name}. Cannot perform deletion.")

    request_id: str = metrics_data["requests"][0]["id"]

    delete_response = tas_client.delete_metric(metric_name=metric_name, request_id=request_id)

    assert delete_response.status_code == HTTPStatus.OK, (
        f"Delete request failed with status code: {delete_response.status_code}. "
        f"Response body: {delete_response.text!r}"
    )

    # Verify the number of metrics after deletion is N-1
    updated_metrics_response = tas_client.get_metrics(metric_name=metric_name)
    updated_metrics_data = json.loads(updated_metrics_response.text)
    updated_num_metrics: int = len(updated_metrics_data.get("requests", []))

    expected_num_metrics: int = initial_num_metrics - 1
    assert updated_num_metrics == expected_num_metrics, (
        f"Number of metrics after deletion is {updated_num_metrics}, expected {expected_num_metrics}"
    )


def verify_trustyai_service_name_mappings(
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


def verify_upload_gzip_data_to_trustyai_service(
    client: DynamicClient,
    token: str,
    trustyai_service: TrustyAIService,
    data_path: str,
) -> None:
    """Uploads gzip-compressed data to TrustyAI and verifies the observation count increased.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        trustyai_service (TrustyAIService): The TrustyAI service instance.
        token (str): Authentication token for the service.
        data_path (str): Path to the JSON data file to compress and upload.
    """
    tas_client = TrustyAIServiceClient(token=token, service=trustyai_service, client=client)
    _verify_upload_and_check_observations(
        client=client,
        token=token,
        trustyai_service=trustyai_service,
        data_path=data_path,
        upload_fn=lambda: tas_client.upload_gzip_data(data_path=data_path),
    )


def verify_upload_malformed_gzip_returns_400(
    client: DynamicClient,
    token: str,
    trustyai_service: TrustyAIService,
) -> None:
    """Sends invalid gzip data to TrustyAI's /data/upload and verifies HTTP 400 response.

    Args:
        client (DynamicClient): The client instance for interacting with the cluster.
        trustyai_service (TrustyAIService): The TrustyAI service instance.
        token (str): Authentication token for the service.
    """
    invalid_gzip_payload = b"not-a-valid-gzip-stream"

    response = TrustyAIServiceClient(token=token, service=trustyai_service, client=client).upload_raw_bytes(
        data=invalid_gzip_payload
    )
    assert response.status_code == HTTPStatus.BAD_REQUEST, (
        f"Expected HTTP 400 for malformed gzip, got {response.status_code}: {response.text}"
    )
    assert "could not be decompressed" in response.text, (
        f"Expected error message about decompression failure, got: {response.text}"
    )
