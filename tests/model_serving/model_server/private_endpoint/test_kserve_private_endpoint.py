from simple_logger.logger import get_logger
from tests.model_serving.model_server.private_endpoint.utils import curl_from_pod


LOGGER = get_logger(name=__name__)


class TestKserveInternalEndpoint:
    def test_deploy_model(self, endpoint_namespace, endpoint_isvc, running_flan_pod):
        assert endpoint_isvc.instance.status.modelStatus.states.activeModelState == "Loaded"
        assert (
            endpoint_isvc.instance.status.address.url
            == f"https://{endpoint_isvc.name}.{endpoint_namespace.name}.svc.cluster.local"
        )

    def test_curl_with_istio(
        self,
        endpoint_isvc,
        endpoint_pod_with_istio_sidecar,
        diff_pod_with_istio_sidecar,
        service_mesh_member,
    ):
        LOGGER.info("Testing curl from the same namespace with a pod part of the service mesh")

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=endpoint_pod_with_istio_sidecar,
            endpoint="health",
        )
        assert curl_stdout == "OK"

        LOGGER.info("Testing curl from a different namespace with a pod part of the service mesh")

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=diff_pod_with_istio_sidecar,
            endpoint="health",
            protocol="https",
        )
        assert curl_stdout == "OK"

    def test_curl_outside_istio(
        self,
        endpoint_isvc,
        endpoint_pod_without_istio_sidecar,
        diff_pod_without_istio_sidecar,
        service_mesh_member,
    ):
        LOGGER.info("Testing curl from the same namespace with a pod not part of the service mesh")

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=endpoint_pod_without_istio_sidecar,
            endpoint="health",
            protocol="https",
        )
        assert curl_stdout == "OK"

        LOGGER.info("Testing curl from a different namespace with a pod not part of the service mesh")

        curl_stdout = curl_from_pod(
            isvc=endpoint_isvc,
            pod=diff_pod_without_istio_sidecar,
            endpoint="health",
            protocol="https",
        )
        assert curl_stdout == "OK"
