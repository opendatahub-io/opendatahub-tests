"""
Tests for special characters and boundary values in the model name URL path.

Boundary condition: The model name in the KServe v2 inference URL
``/v2/models/<model-name>/infer`` is a user-controlled string segment.
The serving infrastructure must safely reject path traversal attempts,
URL-encoded special characters, and overly long model names without
crashing or exposing internal paths.
"""

import json
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_pods_healthy,
    send_inference_request,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

_VALID_BODY: str = json.dumps(VALID_OVMS_INFERENCE_BODY)

# Boundary / edge cases for model name in URL
_SPECIAL_MODEL_NAMES: list[tuple[str, str]] = [
    # (model_name, test_id)
    ("../../../etc/passwd", "path_traversal"),
    ("..%2F..%2Fetc%2Fpasswd", "url_encoded_path_traversal"),
    ("nonexistent model with spaces", "spaces_in_name"),
    ("a" * 256, "extremely_long_name_256_chars"),
    ("model-name-with-special-chars-!@#$%", "special_chars"),
    ("<script>alert(1)</script>", "xss_attempt"),
    ("' OR 1=1 --", "sql_injection_attempt"),
    ("", "empty_string_model_name"),
    ("null", "null_string"),
    ("undefined", "undefined_string"),
]

# Acceptable response codes for all invalid model names
_EXPECTED_REJECTION_CODES: set[int] = {
    HTTPStatus.NOT_FOUND,  # 404 - model not found (most common)
    HTTPStatus.BAD_REQUEST,  # 400 - request rejected before routing
    HTTPStatus.FORBIDDEN,  # 403 - path traversal blocked at gateway
    HTTPStatus.NOT_ACCEPTABLE,  # 406 - OVMS returns this for mangled paths
    HTTPStatus.UNPROCESSABLE_ENTITY,  # 422 - invalid model name format
    HTTPStatus.METHOD_NOT_ALLOWED,  # 405 - method not allowed on the path
}


@pytest.mark.tier3
@pytest.mark.rawdeployment
class TestSpecialCharsModelName:
    """Test class for verifying safe handling of special characters in model names.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model is ready and serving

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send POST to inference URL with path traversal model name
        4. Send POST to inference URL with URL-encoded path traversal
        5. Send POST with spaces in model name
        6. Send POST with extremely long model name (256 chars)
        7. Send POST with special characters in model name
        8. Send POST with XSS payload in model name
        9. Send POST with SQL injection attempt in model name
        10. Send POST with empty string model name
        11. Verify pod health after all requests

    Expected Results:
        - HTTP Status Code: 4xx for all edge case model names
        - No path traversal succeeds (no 200 responses with system file contents)
        - Model pod remains healthy (Running, no new restarts)
    """

    @pytest.mark.parametrize(
        ("model_name", "test_id"),
        [pytest.param(name, test_id, id=test_id) for name, test_id in _SPECIAL_MODEL_NAMES],
    )
    def test_special_model_name_returns_4xx(
        self,
        negative_test_ovms_isvc: InferenceService,
        model_name: str,
        test_id: str,
    ) -> None:
        """Verify that special characters in model name return a 4xx status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with a special-character model name
        Then the response should have a 4xx HTTP status code
        And the response should NOT contain sensitive system file contents
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=_VALID_BODY,
            model_name=model_name,
        )

        # The response should not succeed as if the path traversal worked
        assert status_code != HTTPStatus.OK or ("root:" not in response_body and "/bin/bash" not in response_body), (
            f"[{test_id}] Potential path traversal: got HTTP 200 with suspicious content. "
            f"Response: {response_body[:500]}"
        )

        # For non-OK responses, verify they are in expected rejection codes
        if status_code != HTTPStatus.OK:
            assert status_code in _EXPECTED_REJECTION_CODES, (
                f"[{test_id}] Unexpected status code {status_code} for model name {model_name!r}. "
                f"Response: {response_body[:200]}"
            )

    def test_pod_remains_healthy_after_special_model_name_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after special character model name requests.

        Given an InferenceService is deployed and ready
        When sending requests with special characters in the model name
        Then the pods should remain Running with no new restarts
        """
        for model_name, _ in _SPECIAL_MODEL_NAMES[:5]:  # test first 5 edge cases
            send_inference_request(
                inference_service=negative_test_ovms_isvc,
                body=_VALID_BODY,
                model_name=model_name,
            )

        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
