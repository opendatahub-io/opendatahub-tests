from typing import Self

import pytest
import structlog
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_server.kserve.ingress.utils import curl_from_pod
from utilities.constants import RuntimeTemplates

LOGGER = structlog.get_logger(name=__name__)

OVMS_REST_PORT = 8888
OVMS_HEALTH_ENDPOINT = "v2/health/ready"
HTTP_OK = "200"

pytestmark = [pytest.mark.tier1, pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template",
    [
        pytest.param(
            {"name": "endpoint"},
            {
                "name": "ovms-endpoint-runtime",
                "template-name": RuntimeTemplates.OVMS_KSERVE,
                "multi-model": False,
            },
        )
    ],
    indirect=True,
)
class TestKserveInternalEndpoint:
    """
    Tests the internal endpoint of a KServe RawDeployment predictor using OVMS with S3 storage.

    Steps:
        1. Deploy OVMS ServingRuntime and InferenceService with S3 storage in RawDeployment mode.
        2. Verify the model state reaches "Loaded".
        3. Verify the internal endpoint URL is set correctly.
        4. Curl v2/health/ready from a pod in the same namespace — expect HTTP 200.
        5. Curl v2/health/ready from a pod in a different namespace — expect HTTP 200.
    """

    def test_deploy_model_state_loaded(self: Self, endpoint_isvc: InferenceService) -> None:
        """Verifies that the predictor gets to state Loaded."""
        assert endpoint_isvc.instance.status.modelStatus.states.activeModelState == "Loaded"

    def test_deploy_model_url(self: Self, endpoint_isvc: InferenceService) -> None:
        """Verifies that the internal endpoint URL is set."""
        url = endpoint_isvc.instance.status.address.url
        assert url is not None
        assert endpoint_isvc.name in url
        assert endpoint_isvc.namespace in url

    def test_curl_same_namespace(
        self: Self,
        endpoint_isvc: InferenceService,
        same_namespace_pod: Pod,
    ) -> None:
        """
        Verifies the v2 health endpoint is reachable
        from a pod in the same namespace.
        """
        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=same_namespace_pod,
            endpoint=OVMS_HEALTH_ENDPOINT,
            port=OVMS_REST_PORT,
        )
        assert curl_stdout == HTTP_OK

    def test_curl_diff_namespace(
        self: Self,
        endpoint_isvc: InferenceService,
        diff_namespace_pod: Pod,
    ) -> None:
        """
        Verifies the v2 health endpoint is reachable
        from a pod in a different namespace.
        """
        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=diff_namespace_pod,
            endpoint=OVMS_HEALTH_ENDPOINT,
            port=OVMS_REST_PORT,
        )
        assert curl_stdout == HTTP_OK
