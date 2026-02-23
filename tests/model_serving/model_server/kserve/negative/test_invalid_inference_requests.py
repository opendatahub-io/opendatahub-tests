"""Tests for invalid inference requests handling.

This module verifies that KServe properly handles inference requests with
unsupported Content-Type headers, returning appropriate error responses.

Jira: RHOAIENG-48283
"""

from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.utils import (
    send_inference_request_with_content_type,
)
from utilities.infra import get_pods_by_isvc_label

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.jira("RHOAIENG-48283", run=False)
@pytest.mark.tier1
@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, negative_test_ovms_isvc",
    [
        pytest.param(
            {"name": "negative-test-content-type"},
            {"model-dir": "test-dir"},
        )
    ],
    indirect=True,
)
class TestUnsupportedContentType:
    """Test class for verifying error handling when using unsupported Content-Type headers.

    Preconditions:
        - InferenceService deployed and ready
        - Model accepts application/json content type

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send POST to inference endpoint with header Content-Type: text/xml
        4. Send POST with header Content-Type: application/x-www-form-urlencoded
        5. Capture responses for both requests
        6. Verify model pod health status

    Expected Results:
        - HTTP Status Code: 415 Unsupported Media Type for invalid Content-Types
        - Error indicates expected content type is application/json
        - Model pod remains healthy (Running, no restarts)
    """

    VALID_INFERENCE_BODY: dict[str, Any] = {
        "inputs": [
            {
                "name": "Input3",
                "shape": [1, 1, 28, 28],
                "datatype": "FP32",
                "data": [0.0] * 784,
            }
        ]
    }

    @pytest.mark.parametrize(
        "content_type",
        [
            pytest.param("text/xml", id="text_xml"),
            pytest.param("application/x-www-form-urlencoded", id="form_urlencoded"),
        ],
    )
    def test_unsupported_content_type_returns_415(
        self,
        negative_test_ovms_isvc: InferenceService,
        content_type: str,
    ) -> None:
        """Verify that unsupported Content-Type headers return 415 status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with an unsupported Content-Type header
        Then the response should have HTTP status code 415 (Unsupported Media Type)
        """
        status_code, response_body = send_inference_request_with_content_type(
            inference_service=negative_test_ovms_isvc,
            content_type=content_type,
            body=self.VALID_INFERENCE_BODY,
        )

        assert status_code == HTTPStatus.UNSUPPORTED_MEDIA_TYPE, (
            f"Expected 415 Unsupported Media Type for Content-Type '{content_type}', "
            f"got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_invalid_requests(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after receiving invalid requests.

        Given an InferenceService is deployed and ready
        When sending requests with unsupported Content-Type headers
        Then the same pods (by UID) should still be running without additional restarts
        """
        send_inference_request_with_content_type(
            inference_service=negative_test_ovms_isvc,
            content_type="text/xml",
            body=self.VALID_INFERENCE_BODY,
        )

        current_pods = get_pods_by_isvc_label(
            client=admin_client,
            isvc=negative_test_ovms_isvc,
        )

        assert len(current_pods) > 0, "No pods found for the InferenceService"

        current_pod_uids = {pod.instance.metadata.uid for pod in current_pods}
        initial_pod_uids = set(initial_pod_state.keys())

        assert current_pod_uids == initial_pod_uids, (
            f"Pod UIDs changed after invalid requests. "
            f"Initial: {initial_pod_uids}, Current: {current_pod_uids}. "
            f"This indicates pods were recreated."
        )

        for pod in current_pods:
            uid = pod.instance.metadata.uid
            initial_state = initial_pod_state[uid]

            assert pod.instance.status.phase == "Running", (
                f"Pod {pod.name} is not running, status: {pod.instance.status.phase}"
            )

            container_statuses = pod.instance.status.containerStatuses or []
            for container in container_statuses:
                initial_restart_count = initial_state["restart_counts"].get(container.name, 0)
                assert container.restartCount == initial_restart_count, (
                    f"Container {container.name} in pod {pod.name} restarted after invalid requests. "
                    f"Initial count: {initial_restart_count}, Current count: {container.restartCount}"
                )
