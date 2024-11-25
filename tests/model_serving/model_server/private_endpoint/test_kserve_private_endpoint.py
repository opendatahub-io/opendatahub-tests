from typing import Self

from simple_logger.logger import get_logger
from ocp_resources.namespace import Namespace
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.deployment import Deployment
from tests.model_serving.model_server.private_endpoint.utils import curl_from_pod


LOGGER = get_logger(name=__name__)


class TestKserveInternalEndpoint:
    def test_deploy_model_state_loaded(
        self: Self, endpoint_namespace: Namespace, endpoint_isvc: InferenceService, ready_predictor: Deployment
    ) -> None:
        assert endpoint_isvc.instance.status.modelStatus.states.activeModelState == "Loaded"

    def test_deploy_model_url(
        self: Self, endpoint_namespace: Namespace, endpoint_isvc: InferenceService, ready_predictor: Deployment
    ) -> None:
        assert (
            endpoint_isvc.instance.status.address.url
            == f"https://{endpoint_isvc.name}.{endpoint_namespace.name}.svc.cluster.local"
        )

    def test_curl_with_istio_same_ns(
        self: Self,
        endpoint_isvc: InferenceService,
        endpoint_pod_with_istio_sidecar: Pod,
    ) -> None:
        LOGGER.info("Testing curl from the same namespace with a pod part of the service mesh")

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=endpoint_pod_with_istio_sidecar,
            endpoint="health",
        )
        assert curl_stdout == "OK"

    def test_curl_with_istio_diff_ns(
        self: Self,
        endpoint_isvc: InferenceService,
        diff_pod_with_istio_sidecar: Pod,
    ) -> None:
        LOGGER.info("Testing curl from a different namespace with a pod part of the service mesh")

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=diff_pod_with_istio_sidecar,
            endpoint="health",
            protocol="https",
        )
        assert curl_stdout == "OK"

    def test_curl_outside_istio_same_ns(
        self: Self,
        endpoint_isvc: InferenceService,
        endpoint_pod_without_istio_sidecar: Pod,
    ) -> None:
        LOGGER.info("Testing curl from the same namespace with a pod not part of the service mesh")

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=endpoint_pod_without_istio_sidecar,
            endpoint="health",
            protocol="https",
        )
        assert curl_stdout == "OK"

    def test_curl_outside_istio_diff_ns(
        self: Self,
        endpoint_isvc: InferenceService,
        diff_pod_without_istio_sidecar: Pod,
    ) -> None:
        LOGGER.info("Testing curl from a different namespace with a pod not part of the service mesh")

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=diff_pod_without_istio_sidecar,
            endpoint="health",
            protocol="https",
        )
        assert curl_stdout == "OK"
