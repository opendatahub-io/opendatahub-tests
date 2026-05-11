"""Utility functions for NeMo Guardrails tests."""

import http
import re
from typing import Any

import requests
import structlog
import yaml
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from requests import Response
from timeout_sampler import retry

from tests.model_explainability.nemo_guardrails.constants import (
    INPUT_PROMPT_TEMPLATE,
    OUTPUT_PROMPT_TEMPLATE,
)
from utilities.guardrails import get_auth_headers

LOGGER = structlog.get_logger(name=__name__)

# SHA256 digest pattern
SHA256_DIGEST_PATTERN = r"@sha256:[a-f0-9]{64}"


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
                    "openai_api_base": f"http://{model_isvc_name}-predictor.{namespace}.svc.cluster.local/v1",
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
            {"task": "self_check_input", "content": INPUT_PROMPT_TEMPLATE},
            {"task": "self_check_output", "content": OUTPUT_PROMPT_TEMPLATE},
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
                    "openai_api_base": f"http://{model_isvc_name}-predictor.{namespace}.svc.cluster.local/v1",
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
    # Accept any response that indicates the service is running
    # 200 OK = healthy, 404 Not Found = service running but endpoint doesn't exist (still healthy)
    # 401/403 = service running but needs auth (still healthy)
    acceptable_statuses = {
        http.HTTPStatus.OK,
        http.HTTPStatus.NOT_FOUND,
        http.HTTPStatus.UNAUTHORIZED,
        http.HTTPStatus.FORBIDDEN,
        http.HTTPStatus.METHOD_NOT_ALLOWED,
    }
    assert response.status_code in acceptable_statuses, (
        f"Expected service to be responsive, got {response.status_code}: {response.text[:200]}"
    )
    LOGGER.info(f"Health check passed: {response.status_code} (service is responding)")


@retry(exceptions_dict={AssertionError: []}, wait_timeout=300, sleep=5)
def wait_for_nemo_guardrails_health(
    host: str,
    token: str | None,
    ca_bundle_file: str,
    health_endpoint: str = "/",
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
