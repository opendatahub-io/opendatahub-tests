import http

from requests import Response
from simple_logger.logger import get_logger
from typing import Dict, Any, List, Optional

LOGGER = get_logger(name=__name__)


def get_auth_headers(token: str) -> Dict[str, str]:
    return {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}


def get_chat_payload(content: str) -> Dict[str, Any]:
    return {
        "model": "/mnt/models",
        "messages": [
            {"role": "user", "content": content},
        ],
    }


def verify_and_parse_response(response: Response) -> Dict[str, Any]:
    if response.status_code != http.HTTPStatus.OK:
        assert False, f"Expected status code {http.HTTPStatus.OK}, got {response.status_code}"

    try:
        return response.json()
    except ValueError:
        assert False, "Response body is not valid JSON."


def assert_no_errors(errors: List[str], failure_message_prefix: str) -> None:
    if errors:
        error_message = f"{failure_message_prefix}:\n" + "\n".join(f"- {error}" for error in errors)
        assert False, error_message


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

    LOGGER.info(response.text)
    response_data = verify_and_parse_response(response=response)
    errors: List[str] = []

    # Check response warnings
    warnings: List[Dict[str, Any]] = response_data.get("warnings", [])
    unsuitable_input_warning: str = "UNSUITABLE_INPUT"
    if len(warnings) != 1:
        errors.append(f"Expected 1 warning in response, got {len(warnings)}")
    elif warnings[0]["type"] != unsuitable_input_warning:
        errors.append(f"Expected warning type {unsuitable_input_warning}, got {warnings[0]['type']}")

    # Check detections
    detections: Dict[str, Any] = response_data.get("detections", {})
    input_detections: List[Dict[str, Any]] = detections.get("input", [])
    if len(input_detections) != 1:
        errors.append(f"Expected 1 input detection, but got {len(input_detections)}.")
    else:
        # Check first detection (message_index 0)
        results: List[Dict[str, Any]] = input_detections[0].get("results", [])
        if len(results) != 1:
            errors.append(f"Expected 1 detection result, but got {len(results)}")
        else:
            # Check detection details
            detection: Dict[str, Any] = results[0]
            if detection["detector_id"] != detector_id:
                errors.append(f"Expected detector_id {detector_id}, got {detection['detector_id']}")
            if detection["detection"] != detection_name:
                errors.append(f"Expected detection name {detection_name}, got {detection['detection']}")
            if detection["detection_type"] != detection_type:
                errors.append(f"Expected detection_type {detection_type}, got {detection['detection_type']}")
            if detection["text"] != detection_text:
                errors.append(f"Expected text {detection_text}, got {detection['text']}")

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
    LOGGER.info(response.text)

    response_data: Dict[str, Any] = verify_and_parse_response(response)
    errors: List[str] = []

    # Check warning type
    unsuitable_output_warning: str = "UNSUITABLE_OUTPUT"
    warnings: List[Dict[str, Any]] = response_data.get("warnings", [])
    if len(warnings) != 1:
        errors.append(f"Expected 1 warning in response, got {len(warnings)}")
    elif warnings[0]["type"] != unsuitable_output_warning:
        errors.append(f"Expected warning type {unsuitable_output_warning}, got {warnings[0]['type']}")

    # Check detections
    detections: Dict[str, Any] = response_data.get("detections", {})
    output_detections: List[Dict[str, Any]] = detections.get("output", [])
    if len(output_detections) == 0:
        errors.append("Expected output detections")
    else:
        # Check first detection (choice_index 0)
        if output_detections[0].get("choice_index") != 0:
            errors.append(f"Expected choice_index 0, got {output_detections[0].get('choice_index')}")

        results: List[Dict[str, Any]] = output_detections[0].get("results", [])
        if len(results) == 0:
            errors.append("Expected at least one detection result, but got 0.")
        else:
            # Check first detection details
            detection: Dict[str, Any] = results[0]
            if detection["detector_id"] != detector_id:
                errors.append(f"Expected detector_id {detector_id}, got {detection['detector_id']}")
            if detection["detection"] != detection_name:
                errors.append(f"Expected detection name {detection_name}, got {detection['detection']}")
            if detection["detection_type"] != detection_type:
                errors.append(f"Expected detection_type {detection_type}, got {detection['detection_type']}")
            # Check that detection text exists and is non-empty
            detection_text_actual: str = detection.get("text", "")
            if not detection_text_actual or len(detection_text_actual.strip()) == 0:
                errors.append("Expected detection text to be present and non-empty")

    assert_no_errors(errors=errors, failure_message_prefix="Unsuitable output detection verification failed")


def verify_negative_detection_response(response: Response) -> None:
    """
    Verify that a guardrails response indicates no PII detection (negative case).

    Args:
        response: The HTTP response object from the guardrails API
    """
    LOGGER.info(response.text)

    response_data: Dict[str, Any] = verify_and_parse_response(response)
    errors: List[str] = []

    # Check that there are no warnings
    warnings: Optional[List[Any]] = response_data.get("warnings")
    if warnings is not None:
        errors.append(f"Expected no warnings, got {warnings}")

    # Check that there are no detections
    detections: Optional[Dict[str, Any]] = response_data.get("detections")
    if detections is not None:
        errors.append(f"Expected no detections, got {detections}")

    # Check choices array exists and has content
    choices: List[Dict[str, Any]] = response_data.get("choices", [])
    if len(choices) != 1:
        errors.append(f"Expected one choice in response, got {len(choices)}")
    else:
        # Check finish reason is "stop"
        finish_reason: Optional[str] = choices[0].get("finish_reason")
        if finish_reason != "stop":
            errors.append(f"Expected finish_reason 'stop', got '{finish_reason}'")

        # Check message exists and has content
        message: Dict[str, Any] = choices[0].get("message", {})
        content: Optional[str] = message.get("content")
        if content is None:
            errors.append("Expected message content, got none.")

        # Check refusal is null
        refusal: Optional[Any] = message.get("refusal")
        if refusal is not None:
            errors.append(f"Expected refusal to be null, got {refusal}")

    assert_no_errors(errors=errors, failure_message_prefix="Negative detection verification failed")
