"""Tests for malformed JSON payload handling in inference requests.

Jira: RHOAIENG-48279
"""

from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.utils import (
    assert_pods_healthy,
    send_inference_request,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")

MALFORMED_JSON_EXPECTED_CODES: set[int] = {
    HTTPStatus.BAD_REQUEST,
    HTTPStatus.PRECONDITION_FAILED,
}
MISSING_BRACE_BODY = '{"inputs": [{"name": "Input3"'
TRAILING_COMMA_BODY = '{"inputs": [{"name": "Input3",}]}'


@pytest.mark.tier1
@pytest.mark.rawdeployment
class TestMalformedJsonPayload:
    """Test class for verifying error handling when receiving malformed JSON payloads.

    Preconditions:
        - InferenceService deployed with OVMS runtime (RawDeployment)
        - Model is ready and serving

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send POST with malformed JSON bodies (missing brace, trailing comma, plain text)
        4. Verify error responses and pod health

    Expected Results:
        - HTTP Status Code: 400 Bad Request or 412 Precondition Failed
          (OVMS returns 412 for JSON parse errors)
        - Response indicates JSON parse failure
        - No pod crash or restart
    """

    @pytest.mark.parametrize(
        "malformed_body",
        [
            pytest.param(MISSING_BRACE_BODY, id="missing_closing_brace"),
            pytest.param(TRAILING_COMMA_BODY, id="trailing_comma"),
            pytest.param("not json at all", id="plain_text"),
        ],
    )
    def test_malformed_json_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        malformed_body: str,
    ) -> None:
        """Verify that malformed JSON payloads return an error status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with a malformed JSON body
        Then the response should have HTTP status code 400 or 412
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=malformed_body,
        )

        assert status_code in MALFORMED_JSON_EXPECTED_CODES, (
            f"Expected 400 or 412 for malformed JSON, got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_malformed_json(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after receiving malformed JSON.

        Given an InferenceService is deployed and ready
        When sending requests with malformed JSON payloads
        Then the same pods should still be running without additional restarts
        """
        send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=MISSING_BRACE_BODY,
        )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
