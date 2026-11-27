"""Utility functions for NeMo Guardrails tests."""

import http
import re
from typing import Any

import requests
import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import NotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from requests import Response
from timeout_sampler import TimeoutExpiredError, TimeoutSampler, retry

from tests.ai_safety.nemo_guardrails.constants import (
    ADMIN_SERVICE_PORT,
    INPUT_PROMPT_TEMPLATE,
    OUTPUT_PROMPT_TEMPLATE,
)
from utilities.general import SHA256_DIGEST_PATTERN
from utilities.guardrails import get_auth_headers
from utilities.resources.nemo_guardrails import NemoGuardrails

LOGGER = structlog.get_logger(name=__name__)


def validate_nemo_guardrails_images(pod: Pod, trustyai_operator_configmap: ConfigMap) -> None:
    """
    Validate NeMo Guardrails pod images against TrustyAI operator ConfigMap.

    This validation checks that:
    1. All container images have SHA256 digests
    2. Image names and SHA256 digests match ConfigMap (ignoring registry differences)

    Args:
        pod: NeMo Guardrails pod to validate
        trustyai_operator_configmap: TrustyAI operator ConfigMap containing image references

    Raises:
        AssertionError: If validation fails
    """
    configmap_images = trustyai_operator_configmap.instance.data.values()

    # Extract image name (without registry) and SHA256 from an image string
    def parse_image(image_str: str) -> tuple[str, str]:
        """Parse image into (name_with_sha, sha256_only)."""
        # Remove registry (everything before the first '/')
        # Example: registry.redhat.io/rhoai/odh-image@sha256:abc -> rhoai/odh-image@sha256:abc
        if "/" in image_str:
            name_with_sha = image_str.split("/", 1)[1]
        else:
            name_with_sha = image_str
        # Extract just the SHA256 digest
        sha_match = re.search(SHA256_DIGEST_PATTERN, image_str)
        sha256 = sha_match.group(0) if sha_match else ""
        return name_with_sha, sha256

    for container in pod.instance.spec.containers:
        # Check SHA256 digest exists
        assert re.search(SHA256_DIGEST_PATTERN, container.image), (
            f"{container.name}: {container.image} does not have a valid SHA256 digest"
        )

        # Parse pod image
        pod_name_sha, pod_sha = parse_image(image_str=container.image)

        # Check if image name + SHA exists in ConfigMap (ignoring registry)
        found = False
        for cm_image in configmap_images:
            cm_name_sha, cm_sha = parse_image(image_str=cm_image)
            # Match on image name + SHA256, ignoring registry
            if pod_name_sha == cm_name_sha and pod_sha == cm_sha:
                found = True
                break

        assert found, (
            f"{container.name}: {container.image} not found in TrustyAI operator ConfigMap "
            f"(checked image name and SHA, ignoring registry)"
        )


def create_llm_judge_config(namespace: str, model_isvc_name: str, model_name: str) -> dict[str, str]:
    """
    Create NeMo Guardrails ConfigMap data for LLM-as-a-judge configuration.

    Args:
        namespace: Kubernetes namespace where model is deployed
        model_isvc_name: Name of the InferenceService for the model
        model_name: Name of the model

    Returns:
        Dictionary with config.yaml and actions.py content
    """
    config_yaml = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "parameters": {
                    "base_url": f"http://{model_isvc_name}-predictor.{namespace}.svc.cluster.local/v1",
                    "model_name": model_name,
                },
            }
        ],
        "rails": {
            "input": {"flows": ["self check input"]},
            "output": {"flows": ["self check output"]},
        },
    }

    prompts_yml = {
        "prompts": [
            {"task": "self_check_input", "content": INPUT_PROMPT_TEMPLATE, "max_tokens": 10},
            {"task": "self_check_output", "content": OUTPUT_PROMPT_TEMPLATE, "max_tokens": 10},
        ]
    }
    rails_co = ""

    return {
        "config.yaml": yaml.dump(config_yaml),
        "prompts.yaml": yaml.dump(prompts_yml),
        "rails.co": rails_co,
    }


def create_presidio_config(
    namespace: str,
    model_isvc_name: str,
    model_name: str,
    input_entities: list[str],
    output_entities: list[str],
) -> dict[str, str]:
    """
    Create NeMo Guardrails ConfigMap data for Presidio PII detection configuration.

    Args:
        namespace: Kubernetes namespace where model is deployed
        model_isvc_name: Name of the InferenceService for the model
        model_name: Name of the model
        input_entities: List of PII entities to detect in input
        output_entities: List of PII entities to detect in output

    Returns:
        Dictionary with config.yaml content
    """
    config_yaml = {
        "models": [
            {
                "type": "main",
                "engine": "openai",
                "parameters": {
                    "base_url": f"http://{model_isvc_name}-predictor.{namespace}.svc.cluster.local/v1",
                    "model_name": model_name,
                },
            }
        ],
        "rails": {
            "config": {
                "sensitive_data_detection": {
                    "input": {"entities": input_entities},
                    "output": {"entities": output_entities},
                }
            },
            "input": {"flows": ["detect sensitive data on input"]},
            "output": {"flows": ["detect sensitive data on output"]},
        },
    }

    # Minimal rails.co - Presidio uses built-in flows for PII detection
    rails_co = ""

    return {
        "config.yaml": yaml.dump(config_yaml),
        "rails.co": rails_co,
    }


def create_chat_completion_request(message: str, model: str, configuration: str | None) -> dict[str, Any]:
    """
    Create a chat completion request payload for NeMo Guardrails.

    Args:
        message: User message to send
        model: Model name to use
        configuration: Which config id to use

    Returns:
        Request payload dictionary
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": message}],
        "temperature": 0,
    }

    if configuration is not None:
        payload["guardrails"] = {"config_id": configuration}

    return payload


def build_nemo_admin_service_url(nemo_cr: NemoGuardrails, path: str) -> str:
    """
    Build the in-cluster HTTPS URL for a NeMo Guardrails admin-service path.

    Args:
        nemo_cr: NemoGuardrails CR whose Service exposes the admin port
        path: Admin API path (e.g. /admin/info)

    Returns:
        Cluster-local HTTPS URL targeting the admin service port
    """
    return f"https://{nemo_cr.name}.{nemo_cr.namespace}.svc.cluster.local:{ADMIN_SERVICE_PORT}{path}"


def execute_admin_service_get(
    pod: Pod,
    service_url: str,
    token_file: str | None = None,
) -> tuple[int, str]:
    """
    GET an admin-service URL from an in-cluster caller pod.

    Args:
        pod: Ready caller Pod with curl available
        service_url: Full HTTPS admin service URL
        token_file: Absolute in-pod path to a bearer token file. None sends no
            Authorization header. The token is read inside the pod so Pod.execute
            logs never contain the credential value.

    Returns:
        Tuple of (HTTP status code, response body text)
    """
    curl_command = (
        "curl --insecure --silent --show-error --connect-timeout 10 --max-time 30 --write-out '\\n%{http_code}'"
    )
    if token_file:
        curl_command += ' --header "Authorization: Bearer $(cat "$2")"'
        command = ["sh", "-c", f'{curl_command} "$1"', "sh", service_url, token_file]
    else:
        command = ["sh", "-c", f'{curl_command} "$1"', "sh", service_url]
    output = pod.execute(command=command, ignore_rc=True)
    body, status_text = output.rsplit("\n", 1)
    return int(status_text.strip()), body


def send_admin_get(
    host: str,
    path: str,
    ca_bundle_file: str,
    token: str | None = None,
    timeout: int = 30,
) -> Response:
    """
    Send a GET request to a NeMo Guardrails path via an OpenShift Route.

    Used for user-route deny checks against /admin/* paths.

    Args:
        host: Route hostname
        path: API path (e.g. /admin/info)
        ca_bundle_file: Path to CA bundle file for TLS verification
        token: Bearer token for kube-rbac-proxy (None for unauthenticated requests)
        timeout: Request timeout in seconds

    Returns:
        Response object
    """
    headers = get_auth_headers(token=token) if token else {}
    return requests.get(
        url=f"https://{host}{path}",
        headers=headers,
        verify=ca_bundle_file,
        timeout=timeout,
    )


def assert_admin_info_shape(body: dict[str, Any]) -> None:
    """Assert GET /admin/info has the expected top-level capability document shape."""
    for key in ("identity", "endpoints", "documentation", "integration", "rails", "actions"):
        assert key in body, f"Missing '{key}' in /admin/info response: {list(body.keys())}"
    assert "configs" not in body, "/admin/info must not include per-config payloads"

    identity = body["identity"]
    for key in ("name", "version", "upstream_delta_summary"):
        assert identity.get(key), f"identity.{key} must be present and non-empty"

    assert isinstance(body["endpoints"], list), "endpoints must be a list"
    assert isinstance(body["rails"], list), "rails must be a list"
    assert isinstance(body["actions"], list), "actions must be a list"


def send_request(
    url: str,
    token: str | None,
    ca_bundle_file: str,
    message: str,
    model: str,
    configuration: str | None,
) -> Response:
    """
    Send a chat completion request to NeMo Guardrails.

    Args:
        url: Full URL to NeMo Guardrails endpoint
        token: Authentication token (None if auth disabled)
        ca_bundle_file: Path to CA bundle file for TLS verification
        message: User message to send
        model: Model name to use
        configuration: Which config id to use

    Returns:
        Response object
    """
    payload = create_chat_completion_request(message=message, model=model, configuration=configuration)
    headers = get_auth_headers(token=token) if token else {}

    response = requests.post(
        url=url,
        headers=headers,
        json=payload,
        verify=ca_bundle_file,
        timeout=30,
    )

    return response


def verify_auth_required(response: Response) -> None:
    """
    Verify that authentication is required (request without token should fail).

    Args:
        response: HTTP response from NeMo Guardrails

    Raises:
        AssertionError: If authentication is not required
    """
    assert response.status_code in [
        http.HTTPStatus.UNAUTHORIZED,
        http.HTTPStatus.FORBIDDEN,
    ], f"Expected 401/403, got {response.status_code}"


def verify_health_response(response: Response) -> None:
    """
    Verify that the health endpoint returns a valid response.

    Args:
        response: HTTP response from health endpoint

    Raises:
        AssertionError: If health check fails
    """
    # Accept any response that matches the default k8s health check
    assert 200 <= response.status_code < 400, (
        f"Expected service to be responsive, got {response.status_code}: {response.text[:200]}"
    )

    LOGGER.info(f"Health check passed: {response.status_code} (service is responding)")


@retry(exceptions_dict={AssertionError: []}, wait_timeout=300, sleep=5)
def wait_for_nemo_guardrails_health(
    host: str,
    token: str | None,
    ca_bundle_file: str,
    health_endpoint: str = "/v1/health",
) -> bool:
    """
    Wait for NeMo Guardrails to become healthy.

    Args:
        host: Hostname for NeMo Guardrails
        token: Authentication token (None if auth disabled)
        ca_bundle_file: Path to CA bundle file for TLS verification
        health_endpoint: Health check endpoint path (default: "/")

    Returns:
        True when service is healthy

    Raises:
        AssertionError: If health check fails after retries
    """
    url = f"https://{host}{health_endpoint}"
    headers = get_auth_headers(token=token) if token else {}

    response = requests.get(url=url, headers=headers, verify=ca_bundle_file)
    verify_health_response(response=response)
    LOGGER.info(f"NeMo Guardrails is healthy at {host}")
    return True


def build_api_key_env(secret_name: str) -> list[dict]:
    """
    Build a container env entry that sources the OpenAI API key from a Secret.

    Args:
        secret_name: Name of the Secret containing the "token" key

    Returns:
        List with a single env var dict referencing the Secret
    """
    return [
        {
            "name": "OPENAI_API_KEY",
            "valueFrom": {"secretKeyRef": {"name": secret_name, "key": "token"}},
        }
    ]


def route_exists(client: DynamicClient, name: str, namespace: str) -> bool:
    """
    Check whether a Route exists.

    Args:
        client: DynamicClient to query the cluster
        name: Route name
        namespace: Namespace to look in

    Returns:
        True if the Route exists, False otherwise
    """
    return bool(Route(client=client, name=name, namespace=namespace).exists)


def wait_for_route(client: DynamicClient, name: str, namespace: str, *, present: bool) -> None:
    """
    Wait until a Route's existence matches the expected state.

    Args:
        client: DynamicClient to query the cluster
        name: Route name
        namespace: Namespace to look in
        present: True to wait for the Route to exist, False to wait for it to be gone
    """
    for sample in TimeoutSampler(
        wait_timeout=120,
        sleep=5,
        func=lambda: route_exists(client, name, namespace),
    ):
        if sample == present:
            break


def get_nemo_guardrails_pod(
    client: DynamicClient,
    label_selector: str,
    namespace: str,
) -> Pod:
    """
    Get the first NeMo Guardrails pod matching a label selector.

    Args:
        client: DynamicClient to query the cluster
        label_selector: Kubernetes label selector (e.g. "app=<name>")
        namespace: Namespace to look in

    Returns:
        The first matching Pod

    Raises:
        AssertionError: If no pod matches the label selector
    """
    pods = list(Pod.get(client=client, namespace=namespace, label_selector=label_selector))
    assert pods, f"No pods found with label selector '{label_selector}'"
    return pods[0]


def condition_reason(nemo_cr: NemoGuardrails, condition_type: str) -> str | None:
    """
    Get the reason of a specific status condition on a NemoGuardrails CR.

    Args:
        nemo_cr: NemoGuardrails resource to inspect
        condition_type: Condition "type" to look up (e.g. "RouteReady")

    Returns:
        The condition's "reason" value, or None if the condition is not present
    """
    conditions = (nemo_cr.instance.status or {}).get("conditions", [])
    for cond in conditions:
        if cond.get("type") == condition_type:
            return cond.get("reason")
    return None


def condition_status(nemo_cr: NemoGuardrails, condition_type: str) -> str | None:
    """
    Read the status value of a named condition on a NemoGuardrails server.

    Args:
        nemo_cr: NemoGuardrails resource to inspect
        condition_type: Name of the condition to look up (e.g. "AllowedConsumersReady")

    Returns:
        "True", "False", or "Unknown" if the condition exists. None if it has not been set yet
    """
    conditions = (nemo_cr.instance.status or {}).get("conditions", [])
    for cond in conditions:
        if cond.get("type") == condition_type:
            return cond.get("status")
    return None


def nemo_cr_phase(nemo_cr: NemoGuardrails) -> str | None:
    """
    Read the current phase of a NemoGuardrails server (e.g. "Ready", "Error", "Progressing").

    Args:
        nemo_cr: NemoGuardrails resource to inspect

    Returns:
        The phase string, or None if the operator has not written it yet
    """
    return (nemo_cr.instance.status or {}).get("phase")


def wait_for_deployment_rollout(
    deployment: Deployment,
    timeout: int = 300,
) -> None:
    """
    Wait until the deployment rollout completes.

    Waits for:
    - Kubernetes to notice the latest change (observedGeneration >= generation)
    - All new pods to be ready (updatedReplicas >= desired)
    - No pods to be unavailable
    - Old pods to be removed (total <= updated)

    Args:
        deployment: Deployment to wait for
        timeout: Maximum wait time in seconds

    Raises:
        RuntimeError: If rollout does not finish within timeout
    """
    last_status: Any = None
    try:
        for instance in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: deployment.instance,
            exceptions_dict={NotFoundError: []},
        ):
            status = instance.status
            last_status = status
            if status is None:
                continue
            desired = instance.spec.replicas if instance.spec.replicas is not None else 1
            generation = instance.metadata.generation or 0
            observed = getattr(status, "observedGeneration", None) or 0
            updated = getattr(status, "updatedReplicas", None) or 0
            total = getattr(status, "replicas", None) or 0
            unavailable = getattr(status, "unavailableReplicas", None) or 0
            if observed >= generation and updated >= desired and total <= updated and unavailable == 0:
                return
    except TimeoutExpiredError as err:
        raise RuntimeError(
            f"Deployment {deployment.name} rollout did not finish within {timeout}s. Last status: {last_status}"
        ) from err


def get_ready_nemo_guardrails_pod(client: DynamicClient, namespace: str, label_selector: str) -> Pod | None:
    """
    Return a running, ready NeMo Guardrails pod with instance data loaded.

    After a rollout, old pods can still be listed while they terminate, so this function
    filters for pods that are:
    - Running
    - Not shutting down (no deletionTimestamp)
    - All containers ready

    The pod instance data is loaded and rechecked before returning to ensure readiness
    and image validation use the same observation.

    Args:
        client: DynamicClient to query the cluster
        namespace: Namespace to look in
        label_selector: Kubernetes label selector

    Returns:
        A ready Pod with instance data loaded, or None if no pod is ready yet
    """
    # raw=True reads pod data from a single list call, so a pod deleted mid-check
    # can't raise NotFoundError on a per-pod fetch.
    for pod in Pod.get(client=client, namespace=namespace, label_selector=label_selector, raw=True):
        if pod.metadata.deletionTimestamp or not pod.status or pod.status.phase != Pod.Status.RUNNING:
            continue
        container_statuses = pod.status.containerStatuses or []
        if container_statuses and all(status.ready for status in container_statuses):
            pod_wrapper = Pod(client=client, namespace=namespace, name=pod.metadata.name)
            # Load instance data and recheck readiness on the GET result
            try:
                instance = pod_wrapper.instance
            except NotFoundError:
                # Pod was deleted between list and GET; continue to next candidate
                continue
            # Recheck readiness and deletionTimestamp on GET result to ensure validation
            # uses the same observation
            if instance.metadata.deletionTimestamp:
                continue
            if not instance.status or instance.status.phase != Pod.Status.RUNNING:
                continue
            get_container_statuses = instance.status.containerStatuses or []
            if not (get_container_statuses and all(status.ready for status in get_container_statuses)):
                continue
            return pod_wrapper
    return None
