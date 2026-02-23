import http
import json
from typing import Any

import requests
from requests import Response
from simple_logger.logger import get_logger
from timeout_sampler import retry

from tests.model_explainability.guardrails.constants import GuardrailsDetectionPrompt
from utilities.exceptions import UnexpectedValueError
from utilities.guardrails import get_auth_headers

LOGGER = get_logger(name=__name__)


def get_chat_detections_payload(content: str, model: str, detectors: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Constructs a chat detections payload for a given content string.

    Args:
        content: The user's message content.
        model: The model identifier to be used.
        detectors: Optional. A dictionary specifying detectors to be used.
                   If None, detectors are not included in the payload.

    Returns:
        A dictionary representing the chat detections payload.
    """

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "user", "content": content},
        ],
        "temperature": 0,
    }

    if detectors is not None:
        payload["detectors"] = detectors

    return payload


def verify_and_parse_response(response: Response) -> Any:
    assert response.status_code == http.HTTPStatus.OK, (
        f"Expected status code {http.HTTPStatus.OK}, got {response.status_code}"
    )

    response_json = response.json()
    LOGGER.info(f"Guardrails Orchestrator detection response:\n{json.dumps(response_json, indent=4)}")
    return response_json


def assert_no_errors(errors: list[str], failure_message_prefix: str) -> None:
    assert not errors, f"{failure_message_prefix}:\n" + "\n".join(f"- {error}" for error in errors)


def verify_detection(
    detections_list: list[dict[str, Any]],
    detector_id: str,
    detection_name: str,
    detection_type: str,
    expected_detection_text: str | None = None,
) -> list[str]:
    """
    Helper to verify detection results.

    Args:
        detections_list: List of detection objects
        detector_id: Expected detector ID
        detection_name: Expected detection name
        detection_type: Expected detection type
        expected_detection_text: Expected text (if None, just checks text exists and is non-empty)

    Returns:
        List of error messages
    """
    errors = []

    if len(detections_list) == 0:
        errors.append("Expected detections")
        return errors

    results = detections_list[0].get("results", [])
    if len(results) == 0:
        errors.append("Expected at least one detection result, but got 0.")
        return errors

    detection = results[0]

    if detection["detector_id"] != detector_id:
        errors.append(f"Expected detector_id {detector_id}, got {detection['detector_id']}")

    if detection["detection"] != detection_name:
        errors.append(f"Expected detection name {detection_name}, got {detection['detection']}")

    if detection["detection_type"] != detection_type:
        errors.append(f"Expected detection_type {detection_type}, got {detection['detection_type']}")

    detection_text_actual = detection.get("text", "")
    if expected_detection_text:
        if detection_text_actual != expected_detection_text:
            errors.append(f"Expected text {expected_detection_text}, got {detection_text_actual}")
    else:
        if not detection_text_actual or len(detection_text_actual.strip()) == 0:
            errors.append("Expected detection text to be present and non-empty")

    return errors


def verify_builtin_detector_unsuitable_input_response(
    response: Response, detector_id: str, detection_name: str, detection_type: str, detection_text: str
) -> None:
    """
    Verify that a guardrails response indicates an unsuitable input.

    Args:
        response: The HTTP response object from the guardrails API
        detector_id: Expected detector ID
        detection_name: Expected detection name
        detection_type: Expected detection type
        detection_text: Expected detected text
    """

    response_data = verify_and_parse_response(response=response)
    errors = []

    if not response_data:
        raise UnexpectedValueError("Expected non-empty response data but got an empty response.")

    warnings = response_data.get("warnings")
    unsuitable_input_warning: str = "UNSUITABLE_INPUT"

    if warnings is None:
        raise UnexpectedValueError("Expected warnings in response, got None")
    elif len(warnings) != 1:
        errors.append(f"Expected 1 warning in response, got {len(warnings)}")
    elif warnings[0]["type"] != unsuitable_input_warning:
        errors.append(f"Expected warning type {unsuitable_input_warning}, got {warnings[0]['type']}")

    input_detections = response_data.get("detections", {}).get("input", [])
    if len(input_detections) != 1:
        errors.append(f"Expected 1 input detection, but got {len(input_detections)}.")
    else:
        errors.extend(
            verify_detection(
                detections_list=input_detections,
                detector_id=detector_id,
                detection_name=detection_name,
                detection_type=detection_type,
                expected_detection_text=detection_text,
            )
        )

    assert_no_errors(errors=errors, failure_message_prefix="Input detection verification failed")


def verify_builtin_detector_unsuitable_output_response(
    response: Response, detector_id: str, detection_name: str, detection_type: str
) -> None:
    """
    Verify that a guardrails response indicates an unsuitable output.

    Args:
        response: The HTTP response object from the guardrails API
        detector_id: Expected detector ID
        detection_name: Expected detection name
        detection_type: Expected detection type
    """
    response_data = verify_and_parse_response(response=response)
    errors = []

    unsuitable_output_warning = "UNSUITABLE_OUTPUT"
    warnings = response_data.get("warnings", [])
    if len(warnings) != 1:
        errors.append(f"Expected 1 warning in response, got {len(warnings)}")
    elif warnings[0]["type"] != unsuitable_output_warning:
        errors.append(f"Expected warning type {unsuitable_output_warning}, got {warnings[0]['type']}")

    output_detections = response_data.get("detections", {}).get("output", [])

    if len(output_detections) < 1:
        errors.append(f"Expected at least one output detection, but got {len(output_detections)}.")
    else:
        errors.extend(
            verify_detection(
                detections_list=output_detections,
                detector_id=detector_id,
                detection_name=detection_name,
                detection_type=detection_type,
            )
        )

    assert_no_errors(errors=errors, failure_message_prefix="Unsuitable output detection verification failed")


def verify_negative_detection_response(response: Response) -> None:
    """
    Verify that a guardrails response indicates no PII detection (negative case).

    Args:
        response: The HTTP response object from the guardrails API
    """

    response_data = verify_and_parse_response(response=response)
    errors = []

    warnings = response_data.get("warnings")
    if warnings:
        errors.append(f"Expected no warnings, got {warnings}")

    detections = response_data.get("detections")
    if detections:
        errors.append(f"Expected no detections, got {detections}")

    choices = response_data.get("choices", [])
    if len(choices) != 1:
        errors.append(f"Expected one choice in response, got {len(choices)}")
    else:
        finish_reason = choices[0].get("finish_reason")
        if finish_reason != "stop":
            errors.append(f"Expected finish_reason 'stop', got '{finish_reason}'")

        message = choices[0].get("message", {})
        content = message.get("content")
        if not content:
            errors.append("Expected message content, got none.")

        refusal = message.get("refusal")
        if refusal:
            errors.append(f"Expected refusal to be null, got {refusal}")

    assert_no_errors(errors=errors, failure_message_prefix="Negative detection verification failed")


def create_detector_config(*detector_names: str) -> dict[str, dict[str, Any]]:
    detectors_dict = {name: {} for name in detector_names}
    return {
        "input": detectors_dict.copy(),
        "output": detectors_dict.copy(),
    }


def verify_health_info_response(host, token, ca_bundle_file):
    response = requests.get(url=f"https://{host}/info", headers=get_auth_headers(token=token), verify=ca_bundle_file)
    assert response.status_code == http.HTTPStatus.OK

    healthy_status = "HEALTHY"
    response_data = response.json()
    mismatches = []
    for service_name, service_info in response_data["services"].items():
        if service_info["status"] != healthy_status:
            mismatches.append(f"Service {service_name} is not healthy: {service_info['status']}")

    assert not mismatches, f"GuardrailsOrchestrator service failures: {mismatches}"


def _send_guardrails_orchestrator_post_request(
    url: str,
    token: str,
    ca_bundle_file: str,
    payload: dict[str, Any],
) -> requests.Response:
    response = requests.post(
        url=url,
        headers=get_auth_headers(token=token),
        json=payload,
        verify=ca_bundle_file,
    )

    if response.status_code != http.HTTPStatus.OK:
        raise TimeoutError(f"Endpoint not available. Status code: {response.status_code}, response: {response.text}")

    return response


def send_chat_detections_request(
    url: str,
    token: str,
    ca_bundle_file: str,
    content: str,
    model: str,
    detectors: dict[str, Any] = None,
) -> requests.Response:
    payload = get_chat_detections_payload(content=content, model=model, detectors=detectors)
    return _send_guardrails_orchestrator_post_request(
        url=url, token=token, ca_bundle_file=ca_bundle_file, payload=payload
    )


@retry(exceptions_dict={TimeoutError: []}, wait_timeout=120, sleep=4)
def send_and_verify_unsuitable_input_detection(
    url: str,
    token: str,
    ca_bundle_file: str,
    prompt: GuardrailsDetectionPrompt,
    model: str,
    detectors: dict[str, Any] = None,
):
    """Send a prompt to the GuardrailsOrchestrator and verify that it triggers an unsuitable input detection"""
    response = send_chat_detections_request(
        url=url, token=token, ca_bundle_file=ca_bundle_file, content=prompt.content, model=model, detectors=detectors
    )

    verify_builtin_detector_unsuitable_input_response(
        response=response,
        detector_id=prompt.detector_id,
        detection_name=prompt.detection_name,
        detection_type=prompt.detection_type,
        detection_text=prompt.detection_text,
    )
    return response


@retry(exceptions_dict={TimeoutError: []}, wait_timeout=120, sleep=1)
def send_and_verify_unsuitable_output_detection(
    url: str,
    token: str,
    ca_bundle_file: str,
    prompt: GuardrailsDetectionPrompt,
    model: str,
    detectors: dict[str, Any] = None,
):
    """Send a prompt to the GuardrailsOrchestrator and verify that it triggers an unsuitable output detection"""

    response = send_chat_detections_request(
        url=url, token=token, ca_bundle_file=ca_bundle_file, content=prompt.content, model=model, detectors=detectors
    )

    verify_builtin_detector_unsuitable_output_response(
        response=response,
        detector_id=prompt.detector_id,
        detection_name=prompt.detection_name,
        detection_type=prompt.detection_type,
    )
    return response


@retry(exceptions_dict={TimeoutError: []}, wait_timeout=10, sleep=1)
def send_and_verify_negative_detection(
    url: str,
    token: str,
    ca_bundle_file: str,
    content: str,
    model: str,
    detectors: dict[str, Any] = None,
):
    """Send a prompt to the GuardrailsOrchestrator and verify that it doesn't trigger any detection"""

    response = send_chat_detections_request(
        url=url, token=token, ca_bundle_file=ca_bundle_file, content=content, model=model, detectors=detectors
    )

    verify_negative_detection_response(response=response)
    return response


@retry(exceptions_dict={TimeoutError: []}, wait_timeout=10, sleep=1)
def send_and_verify_standalone_detection(
    url: str,
    token: str,
    ca_bundle_file: str,
    detector_name: str,
    content: str,
    expected_min_score: float = 0.9,
):
    """Send a prompt to the standalone detections endpoint and verify that it triggers a detection"""

    payload = {"detectors": {detector_name: {}}, "content": content}
    response = _send_guardrails_orchestrator_post_request(
        url=url, token=token, ca_bundle_file=ca_bundle_file, payload=payload
    )

    data = response.json()
    assert "detections" in data, f"Expected 'detections' key in response, got: {data}"

    score = data["detections"][0]["score"]
    assert score > expected_min_score, f"Expected score > {expected_min_score}, got {score}"

    return response
