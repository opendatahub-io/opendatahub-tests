"""Tests for missing required fields in inference requests.

Jira: RHOAIENG-48281
"""

import json
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


@pytest.mark.tier1
@pytest.mark.rawdeployment
class TestMissingRequiredFields:
    """Test class for verifying error handling when required fields are missing.

    Preconditions:
        - InferenceService deployed with OVMS runtime
        - Model is ready and serving

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send POST with empty body {}
        4. Send POST with body missing "inputs" field
        5. Verify error responses and pod health

    Expected Results:
        - HTTP Status Code: 400 Bad Request
        - Error message indicates missing required field
        - No server crash
    """

    @pytest.mark.parametrize(
        "incomplete_body",
        [
            pytest.param("{}", id="empty_body"),
            pytest.param(json.dumps({"id": "test-123"}), id="missing_inputs_field"),
        ],
    )
    def test_missing_required_fields_returns_400(
        self,
        negative_test_ovms_isvc: InferenceService,
        incomplete_body: str,
    ) -> None:
        """Verify that requests missing required fields return 400 status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with missing required fields
        Then the response should have HTTP status code 400 (Bad Request)
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=incomplete_body,
        )

        assert status_code == HTTPStatus.BAD_REQUEST, (
            f"Expected 400 Bad Request for incomplete payload, got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_missing_fields(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after receiving incomplete requests.

        Given an InferenceService is deployed and ready
        When sending requests with missing required fields
        Then the same pods should still be running without additional restarts
        """
        send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body="{}",
        )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
