import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelStoragePath,
    Protocols,
    ModelInferenceRuntime,
    RuntimeTemplates,
)
from utilities.inference_utils import Inference
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG

pytestmark = [pytest.mark.usefixtures("valid_aws_config"), pytest.mark.rawdeployment]


@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, s3_models_inference_service",
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
                "model-dir": ModelStoragePath.FLAN_T5_SMALL,
            },
        )
    ],
    indirect=True,
)
class TestRestRawDeploymentRoutes:
    def test_default_visibility_value(self, s3_models_inference_service):
        """Test default route visibility value"""
        if labels := s3_models_inference_service.labels:
            assert labels.get("networking.kserve.io/visibility") is None

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
                {"visibility": "exposed"},
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
    "model_namespace, serving_runtime_from_template, s3_models_inference_service",
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
                "model-dir": ModelStoragePath.FLAN_T5_SMALL,
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
                {"visibility": "exposed"},
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
