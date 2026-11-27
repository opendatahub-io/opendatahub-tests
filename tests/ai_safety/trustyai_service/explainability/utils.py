import json
import math
import sys
from collections.abc import Generator
from contextlib import contextmanager
from functools import lru_cache
from http import HTTPStatus
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.exceptions import ExecOnPodError
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor
from ocp_resources.trustyai_service import TrustyAIService
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.ai_safety.trustyai_service.constants import (
    EXPLAINABILITY_TRAINING_DATA,
    GAUSSIAN_CREDIT_MODEL_INPUT_NAME,
    GAUSSIAN_CREDIT_MODEL_OUTPUT_NAME,
)
from tests.ai_safety.trustyai_service.trustyai_service_utils import TrustyAIServiceClient
from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.general import wait_for_pods_by_labels
from utilities.infra import get_services_by_isvc_label

EXPLAINER_ENV_FLAG: str = "TRUSTYAI_ENABLE_EXPLAINER"
LOCAL_EXPLAINER_ENV_FLAG: str = "TRUSTYAI_ENABLE_EXPLAINER_LOCAL"
ALLOWED_HOSTS_ENV: str = "TRUSTYAI_EXPLAINER_ALLOWED_HOSTS"
ALLOWED_PRIVATE_HOSTS_ENV: str = "TRUSTYAI_EXPLAINER_ALLOWED_PRIVATE_HOSTS"

HTTP_PORT_NAMES: frozenset[str] = frozenset({"http", "http1", "http-port", "rest", "rest-port"})
NON_HTTP_PORT_NAMES: frozenset[str] = frozenset({"grpc", "https"})
TRUTHY_FLAG_VALUES: frozenset[str] = frozenset({"1", "true", "yes", "on", "enabled"})

DEPLOYMENT_ENV_TIMEOUT: int = 180
DEPLOYMENT_ROLLOUT_TIMEOUT: int = 600
PREDICTION_ID_TIMEOUT: int = 180
MODEL_URL_PREDICTOR_SUFFIX: str = "-predictor"


def _service_http_port(service: Any) -> int | None:
    """Return the port to dial for a Service, honouring headless Services.

    A headless Service (``clusterIP: None``) has no kube-proxy port
    translation: DNS returns the endpoint addresses directly, so clients must
    dial the target port. Using the Service port against a headless Service
    sends traffic to a closed port, which a deny-by-default model policy drops,
    producing a connect hang rather than a refusal.
    """
    spec = service.instance.to_dict().get("spec") or {}
    ports = spec.get("ports") or []
    headless = str(spec.get("clusterIP") or "").lower() == "none"
    tcp_ports = [
        port
        for port in ports
        if (port.get("protocol") or "TCP").upper() == "TCP"
        and "metrics" not in (port.get("name") or "").lower()
        and not any(marker in (port.get("name") or "").lower() for marker in NON_HTTP_PORT_NAMES)
    ]
    http_ports = [port for port in tcp_ports if (port.get("name") or "").lower() in HTTP_PORT_NAMES]
    for port in http_ports or tcp_ports:
        candidates = (port.get("targetPort"), port.get("port")) if headless else (port.get("port"),)
        for candidate in candidates:
            if isinstance(candidate, int) and not isinstance(candidate, bool):
                return candidate
    return None


def get_internal_model_url(
    client: DynamicClient,
    inference_service: InferenceService,
    runtime_name: str,
) -> str:
    """Build the cluster-local KServe server root URL for a model.

    Args:
        client (DynamicClient): Kubernetes API client.
        inference_service (InferenceService): Raw-deployment model service.
        runtime_name (str): ServingRuntime name used by the model.

    Returns:
        str: Cluster-local KServe server root URL.

    Raises:
        AssertionError: If no HTTP service port is available.
    """
    services = get_services_by_isvc_label(
        client=client,
        isvc=inference_service,
        runtime_name=runtime_name,
    )
    ordered_services = sorted(
        services,
        key=lambda service: (
            not service.name.endswith(MODEL_URL_PREDICTOR_SUFFIX),
            service.name,
        ),
    )
    for service in ordered_services:
        if (port := _service_http_port(service)) is not None:
            return f"http://{service.name}.{inference_service.namespace}.svc.cluster.local:{port}"

    raise AssertionError(
        f"No HTTP service port found for InferenceService {inference_service.name}: "
        f"{[service.name for service in services]}"
    )


def wait_for_deployment_rollout(deployment: Deployment, timeout: int = DEPLOYMENT_ROLLOUT_TIMEOUT) -> None:
    """Wait until the latest Deployment generation is fully rolled out.

    ``Deployment.wait_for_replicas`` can return while an old pod still serves
    traffic: a new, unready pod counts as updated while the old pod counts as
    ready. This also requires the controller to observe the current generation
    and every non-terminated pod to belong to it.

    Args:
        deployment (Deployment): Deployment to wait for.
        timeout (int): Maximum wait in seconds.

    Raises:
        AssertionError: If the rollout does not complete before the timeout.
    """

    def _rolled_out() -> bool:
        instance = deployment.instance
        status = instance.status
        replicas = instance.spec.replicas if instance.spec.replicas is not None else 1
        return (
            status is not None
            and (status.observedGeneration or 0) >= instance.metadata.generation
            and (status.updatedReplicas or 0) == replicas
            and (status.readyReplicas or 0) == replicas
            and (status.availableReplicas or 0) == replicas
            and (status.replicas or 0) == replicas
        )

    try:
        for rolled_out in TimeoutSampler(wait_timeout=timeout, sleep=2, func=_rolled_out):
            if rolled_out:
                return
    except TimeoutExpiredError as error:
        raise AssertionError(f"Deployment {deployment.name} did not finish rolling out") from error


def _trustyai_service_container(deployment: Deployment) -> dict[str, Any]:
    """Return the TrustyAI application container from a Deployment."""
    containers = deployment.instance.to_dict()["spec"]["template"]["spec"]["containers"]
    container = next(
        (item for item in containers if item.get("name") == TRUSTYAI_SERVICE_NAME),
        None,
    )
    if container is None:
        raise AssertionError(f"Container {TRUSTYAI_SERVICE_NAME} not found in TrustyAI deployment")
    return container


def _container_env(container: dict[str, Any]) -> dict[str, str]:
    """Return named environment variables from a container specification."""
    return {
        item["name"]: item.get("value", "")
        for item in container.get("env", [])
        if isinstance(item, dict) and item.get("name")
    }


def local_explainers_enabled_by_deployment(deployment: Deployment) -> bool:
    """Return whether the deployment explicitly enables both local explainer flags.

    Args:
        deployment (Deployment): TrustyAI service Deployment.

    Returns:
        bool: True when both local explainer feature flags are truthy.
    """
    environment = _container_env(container=_trustyai_service_container(deployment=deployment))
    return all(
        environment.get(name, "").strip().lower() in TRUTHY_FLAG_VALUES
        for name in (EXPLAINER_ENV_FLAG, LOCAL_EXPLAINER_ENV_FLAG)
    )


@contextmanager
def configure_local_explainers(
    client: DynamicClient,
    trustyai_service: TrustyAIService,
    model_url: str,
) -> Generator[Deployment, Any, Any]:
    """Enable local explainers and allow the cluster-local model authority.

    The TrustyAIService resource wrapper exposes no application environment
    configuration, so this test-only adapter patches the generated Deployment and
    restores it when the context exits.

    Args:
        client (DynamicClient): Kubernetes API client.
        trustyai_service (TrustyAIService): TrustyAI service resource.
        model_url (str): Cluster-local KServe server root URL.

    Yields:
        Deployment: Patched TrustyAI deployment.

    Raises:
        AssertionError: If the model URL has no authority or the expected
            environment is reverted by the operator.
    """
    model_authority = urlsplit(url=model_url).netloc
    if not model_authority:
        raise AssertionError(f"Model URL has no host authority: {model_url}")

    deployment = Deployment(
        client=client,
        name=TRUSTYAI_SERVICE_NAME,
        namespace=trustyai_service.namespace,
        ensure_exists=True,
    )
    deployment_body = deployment.instance.to_dict()
    containers = deployment_body["spec"]["template"]["spec"]["containers"]
    service_container = next(
        (container for container in containers if container.get("name") == TRUSTYAI_SERVICE_NAME),
        None,
    )
    if service_container is None:
        raise AssertionError(f"Container {TRUSTYAI_SERVICE_NAME} not found in TrustyAI deployment")

    environment = {
        entry["name"]: entry
        for entry in service_container.get("env", [])
        if isinstance(entry, dict) and entry.get("name")
    }
    expected_environment = {
        EXPLAINER_ENV_FLAG: "true",
        LOCAL_EXPLAINER_ENV_FLAG: "true",
        ALLOWED_HOSTS_ENV: model_authority,
        ALLOWED_PRIVATE_HOSTS_ENV: model_authority,
    }
    for name, value in expected_environment.items():
        environment[name] = {"name": name, "value": value}
    service_container["env"] = list(environment.values())

    patch = {"spec": {"template": {"spec": {"containers": containers}}}}

    def _deployment_has_expected_environment() -> bool:
        current_deployment = Deployment(
            client=client,
            name=TRUSTYAI_SERVICE_NAME,
            namespace=trustyai_service.namespace,
            ensure_exists=True,
        )
        current_environment = _container_env(container=_trustyai_service_container(deployment=current_deployment))
        return all(current_environment.get(name) == value for name, value in expected_environment.items())

    with ResourceEditor(patches={deployment: patch}):
        try:
            for observed in TimeoutSampler(
                wait_timeout=DEPLOYMENT_ENV_TIMEOUT,
                sleep=2,
                func=_deployment_has_expected_environment,
            ):
                if observed:
                    break
        except TimeoutExpiredError as error:
            raise AssertionError(
                f"TrustyAI deployment did not retain the local explainer environment: {expected_environment}"
            ) from error
        wait_for_deployment_rollout(deployment=deployment)
        yield deployment

    wait_for_deployment_rollout(deployment=deployment)


def _inference_ids(client: TrustyAIServiceClient, model_name: str) -> set[str]:
    """Return the stored inference IDs for a model or an empty set when absent."""
    response = client.get_inference_ids(model_name=model_name, timeout=30)
    if response.status_code in {HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND}:
        return set()
    if response.status_code != HTTPStatus.OK:
        raise AssertionError(f"Failed to list inference IDs: {response.status_code}: {response.text}")
    return {str(item["id"]) for item in response.json().get("ids", [])}


def _prediction_id_row(id_value: str) -> int:
    """Return the stored row index encoded in a prediction ID, or a large sentinel."""
    suffix = id_value.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else sys.maxsize


def upload_training_data_and_get_prediction_id(
    client: TrustyAIServiceClient,
    data_path: Path,
    model_name: str,
) -> str:
    """Upload organic background data and return the first stored row's ID.

    Stored IDs are ``<upload-uuid>_<row>``. Returning the row-0 ID makes the
    stored instance deterministic, so a test can compare an explanation's
    ``prediction_output`` against an independent prediction for that same row.

    Args:
        client (TrustyAIServiceClient): Authenticated TrustyAI client.
        data_path (Path): Upload payload with a request and response.
        model_name (str): Stored model name.

    Returns:
        str: The prediction ID for row 0 of this upload.

    Raises:
        AssertionError: If the upload fails or no new ID appears before timeout.
    """
    existing_ids = _inference_ids(client=client, model_name=model_name)

    upload_response = client.upload_data(data_path=str(data_path), timeout=300)
    assert upload_response.status_code == HTTPStatus.OK, (
        f"Data upload failed with status {upload_response.status_code}: {upload_response.text}"
    )

    def new_ids() -> list[str]:
        return sorted(_inference_ids(client=client, model_name=model_name) - existing_ids)

    try:
        for observed_ids in TimeoutSampler(
            wait_timeout=PREDICTION_ID_TIMEOUT,
            sleep=2,
            func=new_ids,
        ):
            if observed_ids:
                return min(observed_ids, key=_prediction_id_row)
    except TimeoutExpiredError as error:
        raise AssertionError(f"No new organic prediction ID appeared for model {model_name}") from error

    raise AssertionError(f"No new organic prediction ID appeared for model {model_name}")


def build_local_explanation_payload(
    model_url: str,
    model_name: str,
    prediction_id: str,
    explainer_config: dict[str, Any],
) -> dict[str, Any]:
    """Build one local explanation request for a regression model.

    The model car declares no tensor metadata, so the request supplies both
    tensor names. That activates the service's request-supplied metadata
    fallback; without both names the service returns 400 invalid_request.

    Args:
        model_url (str): Cluster-local KServe server root URL.
        model_name (str): Deployed model name.
        prediction_id (str): Stored target prediction identifier.
        explainer_config (dict[str, Any]): Algorithm-specific explainer options.

    Returns:
        dict[str, Any]: Local explanation request body.
    """
    return {
        "predictionId": prediction_id,
        "config": {
            "model": {
                "base_url": model_url,
                "model_name": model_name,
                "task": "REGRESSION",
                "input_name": GAUSSIAN_CREDIT_MODEL_INPUT_NAME,
                "output_name": GAUSSIAN_CREDIT_MODEL_OUTPUT_NAME,
            },
            "explainer": explainer_config,
        },
    }


def assert_common_explanation_response(
    response_body: dict[str, Any],
    model_name: str,
    prediction_id: str,
) -> None:
    """Assert fields common to successful local explanation responses.

    Args:
        response_body (dict[str, Any]): Parsed explanation response.
        model_name (str): Expected public model identity.
        prediction_id (str): Expected stored target identifier.

    Raises:
        AssertionError: If a common response field does not match.
    """
    assert response_body["prediction_id"] == prediction_id
    assert response_body["model"] == model_name
    assert response_body["task"] == "REGRESSION"
    assert response_body.get("class_index") is None
    assert response_body.get("output_name") is None or isinstance(response_body["output_name"], str)


def assert_finite_attributions(
    response_body: dict[str, Any],
    expected_feature_names: set[str],
) -> list[str]:
    """Assert one finite, unique attribution per expected feature.

    The explanation requests every stored feature (``num_features`` equals the
    feature count), so a subset would let a dropped feature pass unnoticed.

    Args:
        response_body (dict[str, Any]): Parsed explanation response.
        expected_feature_names (set[str]): Feature names expected from stored aliases.

    Returns:
        list[str]: Observed feature names.

    Raises:
        AssertionError: If attributions are missing, malformed, duplicated, or
            do not cover every expected feature.
    """
    attributions = response_body["attributions"]
    assert isinstance(attributions, list) and attributions

    observed_feature_names: list[str] = []
    for attribution in attributions:
        feature_name = attribution["feature_name"]
        importance = attribution["importance"]
        assert isinstance(feature_name, str) and feature_name
        assert isinstance(importance, (int, float)) and not isinstance(importance, bool)
        assert math.isfinite(float(importance))
        assert attribution.get("confidence_lower") is None
        assert attribution.get("confidence_upper") is None
        observed_feature_names.append(feature_name)

    assert len(observed_feature_names) == len(set(observed_feature_names)), (
        f"Attributions repeat a feature: {observed_feature_names}"
    )
    missing_features = expected_feature_names - set(observed_feature_names)
    assert not missing_features, f"Attributions are missing features: {sorted(missing_features)}"
    unexpected_features = set(observed_feature_names) - expected_feature_names
    assert not unexpected_features, f"Attributions contain unexpected features: {sorted(unexpected_features)}"
    assert len(observed_feature_names) == len(expected_feature_names)
    return observed_feature_names


def assert_finite_number(value: object) -> None:
    """Assert that a response value is a finite JSON number.

    Args:
        value (object): Value to validate.

    Raises:
        AssertionError: If the value is not a finite number.
    """
    assert isinstance(value, (int, float)) and not isinstance(value, bool)
    assert math.isfinite(float(value))


@lru_cache(maxsize=1)
def first_training_features() -> tuple[float, ...]:
    """Return the feature values of row 0 in the explainability upload payload.

    Returns:
        tuple[float, ...]: Feature values of the row the prediction ID selects.
    """
    payload = json.loads(Path(EXPLAINABILITY_TRAINING_DATA).read_text())
    return tuple(float(value) for value in payload["request"]["inputs"][0]["data"][0])


def get_deployed_model_prediction(
    client: DynamicClient,
    inference_service: InferenceService,
    feature_row: tuple[float, ...],
) -> float:
    """Ask the deployed model for its prediction on one feature row.

    This is an independent oracle for an explanation's ``prediction_output``. It
    bypasses the explainer and speaks KServe V2 to the model container directly.
    The call runs inside the model pod: the served URL is cluster-internal and is
    not reachable from the test runner.

    Args:
        client (DynamicClient): Kubernetes API client.
        inference_service (InferenceService): Deployed model.
        feature_row (tuple[float, ...]): Feature values for one instance.

    Returns:
        float: The model's scalar prediction for the row.

    Raises:
        AssertionError: If the pod call fails or returns an unusable value.
    """
    pods = wait_for_pods_by_labels(
        admin_client=client,
        namespace=inference_service.namespace,
        label_selector=f"serving.kserve.io/inferenceservice={inference_service.name}",
        expected_num_pods=1,
    )
    model_name = inference_service.name
    body = json.dumps({
        "inputs": [
            {
                "name": GAUSSIAN_CREDIT_MODEL_INPUT_NAME,
                "shape": [1, len(feature_row)],
                "datatype": "FP64",
                "data": [list(feature_row)],
            }
        ]
    })
    code = (
        "import json, urllib.request\n"
        f"name = {model_name!r}\n"
        f"body = json.loads({body!r})\n"
        "req = urllib.request.Request(\n"
        "    f'http://127.0.0.1:8080/v2/models/{name}/infer',\n"
        "    data=json.dumps(body).encode(),\n"
        "    headers={'Content-Type': 'application/json'},\n"
        ")\n"
        "outputs = json.loads(urllib.request.urlopen(req, timeout=30).read())['outputs']\n"
        "print(json.dumps(float(outputs[0]['data'][0])))\n"
    )
    try:
        stdout = pods[0].execute(command=["python", "-c", code], container="kserve-container", timeout=60)
    except ExecOnPodError as error:
        raise AssertionError(f"Deployed model prediction failed: {error}") from error
    return float(json.loads(stdout.strip().splitlines()[-1]))


def assert_matches_deployed_model(
    response_body: dict[str, Any],
    expected_prediction: float,
) -> None:
    """Assert the explanation reports the deployed model's own prediction.

    Args:
        response_body (dict[str, Any]): Parsed explanation response.
        expected_prediction (float): Prediction obtained from the model directly.

    Raises:
        AssertionError: If the reported prediction differs from the model's own.
    """
    reported = response_body["prediction_output"]
    assert_finite_number(value=reported)
    assert math.isclose(float(reported), expected_prediction, rel_tol=1e-5, abs_tol=1e-6), (
        f"prediction_output {reported} does not match the deployed model prediction {expected_prediction}"
    )
