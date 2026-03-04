"""Tests for invalid model name in inference endpoint.

Jira: RHOAIENG-48282
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

VALID_BODY_RAW = json.dumps(VALID_OVMS_INFERENCE_BODY)


@pytest.mark.tier1
class TestInvalidModelName:
    """Test class for verifying error handling when targeting a non-existent model.

    Preconditions:
        - InferenceService "negative-test-ovms-isvc" deployed and ready
        - No InferenceService with name "nonexistent-model"

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send inference request to /v2/models/nonexistent-model/infer
        4. Verify error response and existing service health

    Expected Results:
        - HTTP Status Code: 404 Not Found
        - Error message indicates model not found
        - No impact on existing model service
    """

    def test_nonexistent_model_returns_404(
        self,
        negative_test_ovms_isvc: InferenceService,
    ) -> None:
        """Verify that inference to a non-existent model returns 404 status code.

        Given an InferenceService is deployed and ready
        When sending a POST request targeting a non-existent model name
        Then the response should have HTTP status code 404 (Not Found)
        """
        status_code, response_body = send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=VALID_BODY_RAW,
            model_name="nonexistent-model",
        )

        assert status_code == HTTPStatus.NOT_FOUND, (
            f"Expected 404 Not Found for nonexistent model, got {status_code}. Response: {response_body}"
        )

    def test_existing_service_unaffected_after_invalid_model_request(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the existing service remains healthy after invalid model requests.

        Given an InferenceService is deployed and ready
        When sending a request targeting a non-existent model name
        Then the existing service pods should remain running without restarts
        """
        send_inference_request(
            inference_service=negative_test_ovms_isvc,
            body=VALID_BODY_RAW,
            model_name="nonexistent-model",
        )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
