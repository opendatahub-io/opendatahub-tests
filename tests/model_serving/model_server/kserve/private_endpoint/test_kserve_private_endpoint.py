from typing import Self

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_serving.model_server.kserve.private_endpoint.utils import curl_from_pod
from utilities.constants import CurlOutput, ModelEndpoint, Protocols, RuntimeTemplates

LOGGER = get_logger(name=__name__)


pytestmark = [pytest.mark.usefixtures("valid_aws_config")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template",
    [
        pytest.param(
            {"name": "endpoint"},
            {
                "name": "flan-example-runtime",
                "template-name": RuntimeTemplates.CAIKIT_TGIS_SERVING,
                "multi-model": False,
            },
        )
    ],
    indirect=True,
)
class TestKserveInternalEndpoint:
    """Tests the internal endpoint of a kserve predictor"""

    def test_deploy_model_state_loaded(self: Self, endpoint_isvc: InferenceService) -> None:
        """Verifies that the predictor gets to state Loaded"""
        assert endpoint_isvc.instance.status.modelStatus.states.activeModelState == "Loaded"

    def test_deploy_model_url(self: Self, endpoint_isvc: InferenceService) -> None:
        """Verifies that the internal endpoint has the expected formatting"""
        assert (
            endpoint_isvc.instance.status.address.url
            == f"https://{endpoint_isvc.name}.{endpoint_isvc.namespace}.svc.cluster.local"
        )

    def test_curl_with_istio_same_ns(
        self: Self,
        endpoint_isvc: InferenceService,
        endpoint_pod_with_istio_sidecar: Pod,
    ) -> None:
        """
        Verifies the response from the health endpoint,
        sending a request from a pod in the same ns and part of the Istio Service Mesh
        """

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=endpoint_pod_with_istio_sidecar,
            endpoint=ModelEndpoint.HEALTH,
        )
        assert curl_stdout == CurlOutput.HEALTH_OK

    def test_curl_with_istio_diff_ns(
        self: Self,
        endpoint_isvc: InferenceService,
        diff_pod_with_istio_sidecar: Pod,
    ) -> None:
        """
        Verifies the response from the health endpoint,
        sending a request from a pod in a different ns and part of the Istio Service Mesh
        """

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=diff_pod_with_istio_sidecar,
            endpoint=ModelEndpoint.HEALTH,
            protocol=Protocols.HTTPS,
        )
        assert curl_stdout == CurlOutput.HEALTH_OK

    def test_curl_outside_istio_same_ns(
        self: Self,
        endpoint_isvc: InferenceService,
        endpoint_pod_without_istio_sidecar: Pod,
    ) -> None:
        """
        Verifies the response from the health endpoint,
        sending a request from a pod in the same ns and not part of the Istio Service Mesh
        """

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=endpoint_pod_without_istio_sidecar,
            endpoint=ModelEndpoint.HEALTH,
            protocol=Protocols.HTTPS,
        )
        assert curl_stdout == CurlOutput.HEALTH_OK

    def test_curl_outside_istio_diff_ns(
        self: Self,
        endpoint_isvc: InferenceService,
        diff_pod_without_istio_sidecar: Pod,
    ) -> None:
        """
        Verifies the response from the health endpoint,
        sending a request from a pod in a different ns and not part of the Istio Service Mesh
        """

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=diff_pod_without_istio_sidecar,
            endpoint=ModelEndpoint.HEALTH,
            protocol=Protocols.HTTPS,
        )
        assert curl_stdout == CurlOutput.HEALTH_OK
