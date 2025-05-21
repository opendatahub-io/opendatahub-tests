import pytest

from tests.model_serving.model_server.components.kserve_dsc_deployment_mode.utils import get_service_cluster_ip
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    Annotations,
    KServeDeploymentType,
    ModelAndFormat,
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
    Protocols,
    DscComponents,
)
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_KSERVE_INFERENCE_CONFIG

pytestmark = [pytest.mark.sanity]

RUNTIME_PARAMS = {
    "runtime-name": ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
    "model-format": {ModelAndFormat.OPENVINO_IR: ModelVersion.OPSET1},
}
INFERENCE_SERVICE_PARAMS = {
    "model-version": ModelVersion.OPSET1,
    "model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL,
}


@pytest.mark.serverless
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_inference_service",
    [
        pytest.param(
            {"name": "dsc-serverless"},
            RUNTIME_PARAMS,
            {
                **{"name": f"{ModelFormat.OPENVINO}-{KServeDeploymentType.SERVERLESS.lower()}"},
                **INFERENCE_SERVICE_PARAMS,
            },
        )
    ],
    indirect=True,
)
class TestKServeDSCServerlessDefaultDeploymentMode:
    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [pytest.param({"updated-deployment-mode": KServeDeploymentType.SERVERLESS})],
        indirect=True,
    )
    def test_isvc_contains_serverless_deployment_mode(
        self,
        patched_deployment_in_dsc,
        ovms_inference_service,
    ):
        """Verify that default deployment mode is set to serverless in inference service."""
        assert ovms_inference_service.instance.status.deploymentMode == KServeDeploymentType.SERVERLESS

    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [pytest.param({"updated-deployment-mode": KServeDeploymentType.SERVERLESS})],
        indirect=True,
    )
    def test_kserve_dsc_serverless_default_deployment_mode(self, patched_deployment_in_dsc, ovms_inference_service):
        """Verify that Serverless model can be deployed without specifying in isvc"""
        verify_inference_response(
            inference_service=ovms_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [pytest.param({"updated-deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT})],
        indirect=True,
    )
    def test_isvc_on_dsc_default_deployment_mode_change_to_raw(
        self,
        patched_deployment_in_dsc,
        ovms_inference_service,
    ):
        """Verify that Serverless isvc not changed after dsc default deployment mode is changed to raw"""
        assert ovms_inference_service.instance.status.deploymentMode == KServeDeploymentType.SERVERLESS

    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [pytest.param({"updated-deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT})],
        indirect=True,
    )
    def test_restarted_pod_is_serverless(
        self,
        patched_deployment_in_dsc,
        restarted_inference_pod,
    ):
        """Verify that serverless pod is not changed after dsc default deployment mode is changed to raw"""
        pod_containers = {container.name for container in restarted_inference_pod.instance.spec.containers}

        assert pod_containers == {"kserve-container", "queue-proxy", "istio-proxy"}


@pytest.mark.rawdeployment
@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_inference_service",
    [
        pytest.param(
            {"name": "dsc-raw"},
            RUNTIME_PARAMS,
            {
                **{"name": f"{ModelFormat.OPENVINO}-{KServeDeploymentType.RAW_DEPLOYMENT.lower()}"},
                **INFERENCE_SERVICE_PARAMS,
            },
        )
    ],
    indirect=True,
)
class TestKServeDSCRawDefaultDeploymentMode:
    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [pytest.param({"updated-deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT})],
        indirect=True,
    )
    def test_isvc_contains_raw_deployment_mode(self, patched_deployment_in_dsc, ovms_inference_service):
        """Verify that default deployment mode is set to raw in inference service."""
        assert (
            ovms_inference_service.instance.metadata.annotations[Annotations.KserveIo.DEPLOYMENT_MODE]
            == KServeDeploymentType.RAW_DEPLOYMENT
        )

    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [pytest.param({"updated-deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT})],
        indirect=True,
    )
    def test_kserve_dsc_raw_default_deployment_mode(self, patched_deployment_in_dsc, ovms_inference_service):
        """Verify that Raw deployment model can be deployed without specifying in isvc"""
        verify_inference_response(
            inference_service=ovms_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [pytest.param({"updated-deployment-mode": KServeDeploymentType.SERVERLESS})],
        indirect=True,
    )
    def test_isvc_on_dsc_default_deployment_mode_change_to_serverless(
        self, patched_deployment_in_dsc, ovms_inference_service
    ):
        """Verify that Raw isvc not changed after dsc default deployment mode is changed to serverless"""
        assert (
            ovms_inference_service.instance.metadata.annotations[Annotations.KserveIo.DEPLOYMENT_MODE]
            == KServeDeploymentType.RAW_DEPLOYMENT
        )

    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [pytest.param({"updated-deployment-mode": KServeDeploymentType.SERVERLESS})],
        indirect=True,
    )
    def test_restarted_pod_is_raw(
        self,
        patched_deployment_in_dsc,
        restarted_inference_pod,
    ):
        """Verify that raw pod is not changed after dsc default deployment mode is changed to serverless"""
        pod_containers = {container.name for container in restarted_inference_pod.instance.spec.containers}

        assert pod_containers == {"kserve-container"}

    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [
            pytest.param(
                {"raw-deployment-service-config": DscComponents.RawDeploymentServiceConfig.HEADLESS},
            )
        ],
        indirect=True,
    )
    def test_default_service_config_is_headless(
        self,
        admin_client,
        patched_deployment_in_dsc,
        ovms_inference_service,
    ):
        """
        Verify that the default service configuration is set to Headless mode
        (serviceClusterIPNone=True).
        """

        assert get_service_cluster_ip(admin_client=admin_client) is True, (
            "Default service config should be Headless (serviceClusterIPNone=True)"
        )

    @pytest.mark.parametrize(
        "patched_deployment_in_dsc",
        [
            pytest.param(
                {"raw-deployment-service-config": DscComponents.RawDeploymentServiceConfig.HEADED},
            )
        ],
        indirect=True,
    )
    def test_service_config_can_be_updated_to_headed(
        self,
        patched_deployment_in_dsc,
        ovms_inference_service,
        admin_client,
    ):
        """
        Verify that the service configuration can be updated from Headless to Headed mode.
        """
        assert get_service_cluster_ip(admin_client=admin_client) is False, (
            "Service config should be Headed (serviceClusterIPNone=False) after update"
        )
