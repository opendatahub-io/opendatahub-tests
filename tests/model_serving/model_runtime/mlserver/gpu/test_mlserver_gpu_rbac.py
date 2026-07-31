"""RBAC tests for GPU-enabled MLServer CUDA inference endpoint.

Validates that bearer-token authentication is enforced on a GPU MLServer
CUDA runtime InferenceService with an external route.
"""

from http import HTTPStatus
from typing import Any

import pytest
import requests
from ocp_resources.inference_service import InferenceService

from tests.model_serving.model_runtime.mlserver.constant import ONNX_RESNET50_REST_INPUT_QUERY
from tests.model_serving.model_runtime.mlserver.utils import get_model_storage_uri_dict, validate_deterministic_snapshot
from utilities.constants import KServeDeploymentType, ModelFormat, Timeout
from utilities.inference_utils import get_exposed_isvc_url

pytestmark = [
    pytest.mark.usefixtures("skip_if_no_gpu_for_mlserver", "valid_aws_config"),
    pytest.mark.gpu,
    pytest.mark.mlserver_nvidia_gpu,
]

_MINIMAL_PAYLOAD: dict[str, Any] = {"id": "rbac-check", "inputs": []}


@pytest.mark.parametrize(
    ("model_namespace", "s3_models_storage_uri", "mlserver_serving_runtime", "mlserver_inference_service"),
    [
        pytest.param(
            {"name": "mlserver-gpu-rbac"},
            get_model_storage_uri_dict(model_format_name=ModelFormat.ONNX),
            {"deployment_mode": KServeDeploymentType.STANDARD, "gpu": True},
            {
                "name": "resnet-50-onnx",
                "gpu_count": 1,
                "deployment_mode": KServeDeploymentType.STANDARD,
                "enable_external_route": True,
                "enable_auth": True,
                "timeout": Timeout.TIMEOUT_10MIN,
            },
            id="test_mlserver_cuda_rbac_enforcement",
        ),
    ],
    indirect=["model_namespace", "s3_models_storage_uri", "mlserver_serving_runtime", "mlserver_inference_service"],
)
@pytest.mark.usefixtures("authorized_inference_role_binding")
class TestRBACGPUEnabledRuntimes:
    """RBAC enforcement on GPU MLServer CUDA inference endpoints."""

    def test_authorized_inference(
        self,
        mlserver_inference_service: InferenceService,
        authorized_inference_token: str,
    ) -> None:
        """Verify authorized bearer token gets HTTP 200 and valid inference response.

        Given a GPU ISVC with external route and token auth enabled
        When a request is sent with a valid bearer token
        Then the response is HTTP 200 with valid model output.
        """
        base_url = get_exposed_isvc_url(isvc=mlserver_inference_service)
        model_name = mlserver_inference_service.name
        infer_url = f"{base_url}/v2/models/{model_name}/infer"

        response = requests.post(
            url=infer_url,
            json=ONNX_RESNET50_REST_INPUT_QUERY,
            headers={"Authorization": f"Bearer {authorized_inference_token}"},
            verify=False,
            timeout=60,
        )
        assert response.status_code == HTTPStatus.OK, (
            f"Expected HTTP 200 for authorized request, got {response.status_code}: {response.text[:200]}"
        )
        validate_deterministic_snapshot(response=response.json())

    def test_unauthenticated_rejected(
        self,
        mlserver_inference_service: InferenceService,
    ) -> None:
        """Verify request without bearer token is rejected with 401 or 403.

        Given a GPU ISVC with token auth enabled
        When a request is sent without any Authorization header
        Then the response is HTTP 401 or 403.
        """
        base_url = get_exposed_isvc_url(isvc=mlserver_inference_service)
        model_name = mlserver_inference_service.name
        infer_url = f"{base_url}/v2/models/{model_name}/infer"

        response = requests.post(
            url=infer_url,
            json=_MINIMAL_PAYLOAD,
            verify=False,
            timeout=60,
        )
        assert response.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN), (
            f"Expected HTTP 401 or 403 for unauthenticated request, got {response.status_code}: {response.text[:200]}"
        )

    def test_invalid_token_rejected(
        self,
        mlserver_inference_service: InferenceService,
    ) -> None:
        """Verify request with invalid bearer token is rejected with 401 or 403.

        Given a GPU ISVC with token auth enabled
        When a request is sent with an invalid bearer token
        Then the response is HTTP 401 or 403.
        """
        base_url = get_exposed_isvc_url(isvc=mlserver_inference_service)
        model_name = mlserver_inference_service.name
        infer_url = f"{base_url}/v2/models/{model_name}/infer"

        response = requests.post(
            url=infer_url,
            json=_MINIMAL_PAYLOAD,
            headers={"Authorization": "Bearer invalid-token-rbac-check"},
            verify=False,
            timeout=60,
        )
        assert response.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN), (
            f"Expected HTTP 401 or 403 for invalid token, got {response.status_code}: {response.text[:200]}"
        )
