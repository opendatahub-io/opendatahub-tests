"""TC-NEG-005: MLServer inference with wrong input dimensions.

Validates that MLServer returns a descriptive error when the input tensor
shape does not match the model's expected dimensions.
"""

from typing import Any

import portforward
import pytest
import requests
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.mlserver.constant import BASE_RAW_DEPLOYMENT_CONFIG, LOCALHOST_URL
from tests.model_serving.model_runtime.mlserver.utils import get_model_namespace_dict
from utilities.constants import Ports

WRONG_SHAPE_QUERY: dict[str, Any] = {
    "id": "sklearn-wrong-dim",
    "inputs": [
        {
            "name": "predict",
            "shape": [1, 10],
            "datatype": "FP32",
            "data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]],
        }
    ],
}

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier3
@pytest.mark.negative
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, negative_mlserver_serving_runtime, negative_mlserver_isvc_no_wait",
    [
        pytest.param(
            get_model_namespace_dict(model_format_name="sklearn-neg-dim"),
            {"model-dir": "mlserver/model_repository/sklearn"},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "sklearn-wrong-dim",
                **BASE_RAW_DEPLOYMENT_CONFIG,
            },
            id="mlserver-sklearn-wrong-dimensions",
        ),
    ],
    indirect=[
        "model_namespace",
        "s3_models_storage_uri",
        "negative_mlserver_serving_runtime",
        "negative_mlserver_isvc_no_wait",
    ],
)
class TestWrongDimensions:
    def test_wrong_input_dimensions_returns_error(
        self,
        negative_mlserver_isvc_no_wait: InferenceService,
        mlserver_pod_resource: Pod,
    ) -> None:
        """Given a deployed MLServer sklearn model expecting shape [N, 4],
        When an inference request is sent with shape [1, 10],
        Then the server returns an HTTP 400 with a descriptive error message.
        """
        model_name = negative_mlserver_isvc_no_wait.instance.metadata.name
        endpoint = f"/v2/models/{model_name}/infer"
        port = Ports.REST_PORT

        with portforward.forward(
            pod_or_service=mlserver_pod_resource.name,
            namespace=negative_mlserver_isvc_no_wait.namespace,
            from_port=port,
            to_port=port,
        ):
            response = requests.post(
                url=f"{LOCALHOST_URL}:{port}{endpoint}",
                json=WRONG_SHAPE_QUERY,
                verify=False,
                timeout=30,
            )
            assert response.status_code in (400, 422, 500), (
                f"Expected error status for wrong dimensions, got {response.status_code}: {response.text}"
            )
            error_text = response.text.lower()
            assert any(keyword in error_text for keyword in ("shape", "dimension", "feature", "mismatch", "error")), (
                f"Error response should mention shape/dimension issue: {response.text}"
            )
