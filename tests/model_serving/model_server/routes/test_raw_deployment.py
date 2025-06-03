import time
import pytest

from ocp_resources.resource import ResourceEditor
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    Annotations,
    KServeDeploymentType,
    Labels,
    ModelFormat,
    ModelStoragePath,
    Protocols,
    ModelInferenceRuntime,
    RuntimeTemplates,
)
from utilities.exceptions import (
    InferenceResponseError,
)
from utilities.inference_utils import Inference
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG

pytestmark = [pytest.mark.usefixtures("valid_aws_config"), pytest.mark.rawdeployment]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-caikit-flan-rest"},
            {
                "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
                "template-name": RuntimeTemplates.CAIKIT_TGIS_SERVING,
                "multi-model": False,
                "enable-http": True,
                "enable-grpc": False,
            },
            {
                "name": f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "model-dir": ModelStoragePath.FLAN_T5_SMALL_CAIKIT,
            },
        )
    ],
    indirect=True,
)
class TestRestRawDeploymentRoutes:
    def test_default_visibility_value(self, s3_models_inference_service):
        """Test default route visibility value"""
        if labels := s3_models_inference_service.labels:
            assert labels.get(Labels.Kserve.NETWORKING_KSERVE_IO) is None

    def test_rest_raw_deployment_internal_route(self, s3_models_inference_service):
        """Test HTTP inference using internal route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.jira("RHOAIENG-17322", run=False)
    @pytest.mark.parametrize(
        "patched_s3_caikit_kserve_isvc_visibility_label",
        [
            pytest.param(
                {"visibility": Labels.Kserve.EXPOSED},
            )
        ],
        indirect=True,
    )
    @pytest.mark.dependency(name="test_rest_raw_deployment_exposed_route")
    def test_rest_raw_deployment_exposed_route(self, patched_s3_caikit_kserve_isvc_visibility_label):
        """Test HTTP inference using exposed (external) route"""
        verify_inference_response(
            inference_service=patched_s3_caikit_kserve_isvc_visibility_label,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_rest_raw_deployment_exposed_route"])
    @pytest.mark.parametrize(
        "patched_s3_caikit_kserve_isvc_visibility_label",
        [
            pytest.param(
                {"visibility": "local-cluster"},
            )
        ],
        indirect=True,
    )
    def test_disabled_rest_raw_deployment_exposed_route(self, patched_s3_caikit_kserve_isvc_visibility_label):
        """Test HTTP inference fails when using external route after it was disabled"""
        verify_inference_response(
            inference_service=patched_s3_caikit_kserve_isvc_visibility_label,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTP,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-caikit-flan-rest-timeout"},
            {
                "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
                "template-name": RuntimeTemplates.CAIKIT_TGIS_SERVING,
                "multi-model": False,
                "enable-http": True,
                "enable-grpc": False,
            },
            {
                "name": f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "model-dir": ModelStoragePath.FLAN_T5_SMALL_CAIKIT,
                "external-route": True,
            },
        )
    ],
    indirect=True,
)
class TestRestRawDeploymentRoutesTimeout:
    def test_rest_raw_deployment_exposed_route(self, s3_models_inference_service):
        """Test HTTP inference using exposed (external) route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.HTTPS,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_rest_raw_deployment_exposed_route"])
    def test_rest_raw_deployment_exposed_route_with_timeout(self, s3_models_inference_service):
        """Test HTTP inference using exposed (external) route fails when timeout is set too low"""
        ResourceEditor(
            patches={
                s3_models_inference_service: {
                    "metadata": {
                        "annotations": {Annotations.HaproxyRouterOpenshiftIo.TIMEOUT: "1us"},
                    }
                }
            }
        ).update()

        # Wait for route to be updated with the annotation and timeout to be applied
        time.sleep(10)

        with pytest.raises(InferenceResponseError) as ire:
            verify_inference_response(
                inference_service=s3_models_inference_service,
                inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
                inference_type=Inference.ALL_TOKENS,
                protocol=Protocols.HTTPS,
                model_name=ModelFormat.CAIKIT,
                use_default_query=True,
            )
        assert "504 Gateway Time-out" in str(ire)


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-caikit-flan-grpc"},
            {
                "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
                "template-name": RuntimeTemplates.CAIKIT_TGIS_SERVING,
                "multi-model": False,
                "enable-grpc": True,
                "enable-http": False,
            },
            {
                "name": f"{Protocols.GRPC}-{ModelFormat.CAIKIT}",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "model-dir": ModelStoragePath.FLAN_T5_SMALL_CAIKIT,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.jira("RHOAIENG-17783", run=False)
class TestGrpcRawDeployment:
    def test_grpc_raw_deployment_internal_route(self, s3_models_inference_service):
        """Test GRPC inference using internal route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.STREAMING,
            protocol=Protocols.GRPC,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.parametrize(
        "patched_s3_caikit_kserve_isvc_visibility_label",
        [
            pytest.param(
                {"visibility": Labels.Kserve.EXPOSED},
            )
        ],
        indirect=True,
    )
    def test_grpc_raw_deployment_exposed_route(self, patched_s3_caikit_kserve_isvc_visibility_label):
        """Test GRPC inference using exposed (external) route"""
        verify_inference_response(
            inference_service=patched_s3_caikit_kserve_isvc_visibility_label,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.STREAMING,
            protocol=Protocols.GRPC,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-caikit-flan-grpc-timeout"},
            {
                "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME}",
                "template-name": RuntimeTemplates.CAIKIT_TGIS_SERVING,
                "multi-model": False,
                "enable-grpc": True,
                "enable-http": False,
            },
            {
                "name": f"{Protocols.GRPC}-{ModelFormat.CAIKIT}",
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "model-dir": ModelStoragePath.FLAN_T5_SMALL_CAIKIT,
                "external-route": True,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.jira("RHOAIENG-17783", run=False)
class TestGrpcRawDeploymentTimeout:
    def test_grpc_raw_deployment_exposed_route(self, s3_models_inference_service):
        """Test GRPC inference using exposed (external) route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
            inference_type=Inference.STREAMING,
            protocol=Protocols.GRPC,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.dependency(depends=["test_rest_raw_deployment_exposed_route"])
    def test_grpc_raw_deployment_exposed_route_with_timeout(self, s3_models_inference_service):
        """Test GRPC inference using exposed (external) route fails when timeout is set too low"""
        ResourceEditor(
            patches={
                s3_models_inference_service: {
                    "metadata": {
                        "annotations": {Annotations.HaproxyRouterOpenshiftIo.TIMEOUT: "1us"},
                    }
                }
            }
        ).update()

        # Wait for route to be updated with the annotation and timeout to be applied
        time.sleep(10)

        with pytest.raises(InferenceResponseError) as ire:
            verify_inference_response(
                inference_service=s3_models_inference_service,
                inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
                inference_type=Inference.STREAMING,
                protocol=Protocols.GRPC,
                model_name=ModelFormat.CAIKIT,
                use_default_query=True,
            )
        assert "504 Gateway Time-out" in str(ire)
