import pytest

from tests.model_serving.model_server.authentication.utils import (
    verify_inference_response,
)
from utilities.constants import (
    ModelName,
    ModelStoragePath,
    Protocols,
    ModelInferenceRuntime,
)
from utilities.inference_utils import Inference

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.smoke
@pytest.mark.raw_deployment
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri",
    [
        pytest.param(
            {"name": "raw-deployment-caikit-bge"},
            {"model-dir": ModelStoragePath.EMBEDDING_MODEL},
        )
    ],
    indirect=True,
)
@pytest.mark.jira("RHOAIENG-11749")
def test_caikit_bge_large_en_raw_internal_route(http_s3_caikit_raw_inference_service):
    """Test Caikit bge-large-en model inference using internal route"""
    verify_inference_response(
        inference_service=http_s3_caikit_raw_inference_service,
        runtime=ModelInferenceRuntime.CAIKIT_TGIS_RUNTIME,
        inference_type=Inference.ALL_TOKENS,
        protocol=Protocols.HTTP,
        model_name=ModelName.CAIKIT_BGE_LARGE_EN,
        use_default_query=True,
    )
