"""Tests for wrong data types in input tensor.

Jira: RHOAIENG-48280
"""

import copy
import json
from http import HTTPStatus
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_server.kserve.negative.utils import (
    VALID_OVMS_INFERENCE_BODY,
    assert_pods_healthy,
    send_raw_inference_request,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")


def _make_body_with_input_override(**overrides: Any) -> str:
    """Derive a serialized body from VALID_OVMS_INFERENCE_BODY with input field overrides."""
    body = copy.deepcopy(VALID_OVMS_INFERENCE_BODY)
    body["inputs"][0].update(overrides)
    return json.dumps(body)


STRING_VALUES_AS_FP32_BODY = _make_body_with_input_override(data=["string_value"] * 784)
INVALID_DATATYPE_BODY = _make_body_with_input_override(datatype="INVALID_TYPE")


@pytest.mark.tier1
class TestWrongInputDataType:
    """Test class for verifying error handling when input tensor has wrong data type.

    Preconditions:
        - InferenceService deployed with OVMS runtime expecting FP32 inputs
        - Model is ready and serving

    Test Steps:
        1. Create InferenceService with OVMS runtime
        2. Wait for InferenceService status = Ready
        3. Send inference request with string values where FP32 is expected
        4. Send inference request with mismatched datatype declaration
        5. Verify error responses and pod health

    Expected Results:
        - HTTP Status Code: 400 or 422 indicating data type mismatch
        - Model pod remains healthy (no restart)
    """

    @pytest.mark.parametrize(
        "invalid_input_body",
        [
            pytest.param(STRING_VALUES_AS_FP32_BODY, id="string_values_as_fp32"),
            pytest.param(INVALID_DATATYPE_BODY, id="invalid_datatype_name"),
        ],
    )
    def test_wrong_data_type_returns_error(
        self,
        negative_test_ovms_isvc: InferenceService,
        invalid_input_body: str,
    ) -> None:
        """Verify that wrong input data types return an error status code.

        Given an InferenceService is deployed and ready
        When sending a POST request with mismatched input tensor data types
        Then the response should have HTTP status code 400 or 422
        """
        status_code, response_body = send_raw_inference_request(
            inference_service=negative_test_ovms_isvc,
            raw_body=invalid_input_body,
        )

        assert status_code in (HTTPStatus.BAD_REQUEST, HTTPStatus.UNPROCESSABLE_ENTITY), (
            f"Expected 400 or 422 for wrong data type, got {status_code}. Response: {response_body}"
        )

    def test_model_pod_remains_healthy_after_wrong_dtype(
        self,
        admin_client: DynamicClient,
        negative_test_ovms_isvc: InferenceService,
        initial_pod_state: dict[str, dict[str, Any]],
    ) -> None:
        """Verify that the model pod remains healthy after receiving wrong data type inputs.

        Given an InferenceService is deployed and ready
        When sending requests with wrong input tensor data types
        Then the same pods should still be running without additional restarts
        """
        send_raw_inference_request(
            inference_service=negative_test_ovms_isvc,
            raw_body=STRING_VALUES_AS_FP32_BODY,
        )
        assert_pods_healthy(
            admin_client=admin_client,
            isvc=negative_test_ovms_isvc,
            initial_pod_state=initial_pod_state,
        )
