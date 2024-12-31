import pytest

from tests.model_serving.model_server.authentication.utils import (
    verify_inference_response,
)
from utilities.constants import (
    ModelFormat,
    ModelStoragePath,
    Protocols,
    RuntimeQueryKeys,
)
from utilities.inference_utils import Inference

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "grpc-raw-deployment"},
            {"model-dir": ModelStoragePath.FLAN_T5_SMALL},
        )
    ],
    indirect=True,
)
class TestGrpcRawDeployment:
    def test_grpc_raw_deployment_internal_route(self, grpc_s3_caikit_raw_inference_service):
        """Test GRPC inference using internal route"""
        verify_inference_response(
            inference_service=grpc_s3_caikit_raw_inference_service,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )

    @pytest.mark.jira("RHOAIENG-17322", run=False)
    @pytest.mark.parametrize(
        "patched_grpc_s3_caikit_raw_isvc_visibility_label",
        [
            pytest.param(
                {"visibility": "exposed"},
            )
        ],
        indirect=True,
    )
    def test_grpc_raw_deployment_exposed_route(self, patched_grpc_s3_caikit_raw_isvc_visibility_label):
        """Test GRPC inference using exposed (external) route"""
        verify_inference_response(
            inference_service=patched_grpc_s3_caikit_raw_isvc_visibility_label,
            runtime=RuntimeQueryKeys.CAIKIT_TGIS_RUNTIME,
            inference_type=Inference.ALL_TOKENS,
            protocol=Protocols.GRPC,
            model_name=ModelFormat.CAIKIT,
            use_default_query=True,
        )
