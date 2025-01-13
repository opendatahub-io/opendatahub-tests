import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelName,
    ModelStoragePath,
    Protocols,
    ModelInferenceRuntime,
    RuntimeTemplates,
)

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.smoke
@pytest.mark.raw_deployment
@pytest.mark.jira("RHOAIENG-11749")
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-caikit-bge"},
            {"model-dir": ModelStoragePath.EMBEDDING_MODEL},
            {
                "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_STANDALONE_RUNTIME}",
                "template-name": RuntimeTemplates.CAIKIT_STANDALONE_SERVING,
                "multi-model": False,
                "enable-http": True,
            },
            {"name": "bge-large-en-caikit", "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT},
        )
    ],
    indirect=True,
)
class TestBgeLargeEnCaikit:
    def test_caikit_bge_large_en_embedding_raw_internal_route(self, s3_models_inference_service):
        """Test Caikit bge-large-en embedding model inference using internal route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            runtime=ModelInferenceRuntime.CAIKIT_STANDALONE_RUNTIME,
            inference_type="embedding",
            protocol=Protocols.HTTP,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    def test_caikit_bge_large_en_rerank_raw_internal_route(self, s3_models_inference_service):
        """Test Caikit bge-large-en rerank model inference using internal route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            runtime=ModelInferenceRuntime.CAIKIT_STANDALONE_RUNTIME,
            inference_type="rerank",
            protocol=Protocols.HTTP,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    def test_caikit_bge_large_en_sentence_similarity_raw_internal_route(self, s3_models_inference_service):
        """Test Caikit bge-large-en sentence-similarity model inference using internal route"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            runtime=ModelInferenceRuntime.CAIKIT_STANDALONE_RUNTIME,
            inference_type="sentence-similarity",
            protocol=Protocols.HTTP,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )
