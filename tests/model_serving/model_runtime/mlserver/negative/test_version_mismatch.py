"""TC-NEG-006: MLServer inference with model version mismatch.

Validates that MLServer returns a descriptive error when a request targets
a non-existent model version.
"""

import portforward
import pytest
import requests
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.mlserver.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    LOCALHOST_URL,
    SKLEARN_REST_INPUT_QUERY,
)
from tests.model_serving.model_runtime.mlserver.utils import get_model_namespace_dict
from utilities.constants import Ports

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.tier3
@pytest.mark.negative
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, negative_mlserver_serving_runtime, negative_mlserver_isvc_no_wait",
    [
        pytest.param(
            get_model_namespace_dict(model_format_name="sklearn-neg-ver"),
            {"model-dir": "mlserver/model_repository/sklearn"},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "sklearn-ver-mismatch",
                **BASE_RAW_DEPLOYMENT_CONFIG,
            },
            id="mlserver-sklearn-version-mismatch",
        ),
    ],
    indirect=[
        "model_namespace",
        "s3_models_storage_uri",
        "negative_mlserver_serving_runtime",
        "negative_mlserver_isvc_no_wait",
    ],
)
class TestVersionMismatch:
    def test_nonexistent_version_returns_error(
        self,
        negative_mlserver_isvc_no_wait: InferenceService,
        mlserver_pod_resource: Pod,
    ) -> None:
        """Given a deployed MLServer sklearn model at version v1.0.0,
        When an inference request targets version v99.0.0 (non-existent),
        Then the server returns an HTTP 404 or error indicating version not found.
        """
        model_name = negative_mlserver_isvc_no_wait.instance.metadata.name
        endpoint = f"/v2/models/{model_name}/versions/v99.0.0/infer"
        port = Ports.REST_PORT

        with portforward.forward(
            pod_or_service=mlserver_pod_resource.name,
            namespace=negative_mlserver_isvc_no_wait.namespace,
            from_port=port,
            to_port=port,
        ):
            response = requests.post(
                url=f"{LOCALHOST_URL}:{port}{endpoint}",
                json=SKLEARN_REST_INPUT_QUERY,
                verify=False,
                timeout=30,
            )
            assert response.status_code in (400, 404, 500), (
                f"Expected error for non-existent version, got {response.status_code}: {response.text}"
            )
