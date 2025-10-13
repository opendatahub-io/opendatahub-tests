import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    ModelInferenceRuntime,
    ModelName,
    ModelStoragePath,
    Protocols,
    RuntimeTemplates,
)
from utilities.manifests.caikit_standalone import CAIKIT_STANDALONE_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.rawdeployment, pytest.mark.sanity]

RUNTIME_BASE_PARAMS = {
    "template-name": RuntimeTemplates.CAIKIT_STANDALONE_SERVING,
    "multi-model": False,
    "enable-http": True,
}
SERVERLESS_ISVC_PARAMS = {
    "deployment-mode": KServeDeploymentType.SERVERLESS,
    "model-dir": ModelStoragePath.EMBEDDING_MODEL,
}


@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, s3_models_inference_service, s3_models_second_inference_service",
    [
        pytest.param(
            {"name": "serverless-model-priority"},
            {
                **{
                    "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_STANDALONE_RUNTIME}".lower(),
                    "models-priorities": {ModelFormat.CAIKIT: 2},
                },
                **RUNTIME_BASE_PARAMS,
            },
            {
                **{"name": f"{ModelFormat.CAIKIT}-{KServeDeploymentType.SERVERLESS.lower()}-1"},
                **SERVERLESS_ISVC_PARAMS,
            },
            {
                **{"name": f"{ModelFormat.CAIKIT}-{KServeDeploymentType.SERVERLESS.lower()}-2"},
                **SERVERLESS_ISVC_PARAMS,
            },
        )
    ],
    indirect=True,
)
class TestServerlessModelPriority:
    def test_serverless_model_priority_first_model(
        self,
        s3_models_inference_service,
        s3_models_second_inference_service,
    ):
        """Two models with the same runtime and priority set; verify query of first model"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTPS,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    def test_serverless_model_priority_second_model(
        self,
        s3_models_inference_service,
        s3_models_second_inference_service,
    ):
        """Two models with the same runtime and priority set; verify query of second model"""
        verify_inference_response(
            inference_service=s3_models_second_inference_service,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTPS,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "model_namespace, serving_runtime_from_template, s3_models_inference_service, s3_models_second_inference_service",
    [
        pytest.param(
            {"name": "serverless-multi-priorities"},
            {
                **{
                    "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_STANDALONE_RUNTIME}".lower(),
                    "supported-model-formats": [
                        {
                            "name": ModelFormat.CAIKIT,
                            "autoSelect": True,
                            "priority": 2,
                            "version": "1",
                        },
                        {
                            "name": ModelFormat.CAIKIT,
                            "autoSelect": True,
                            "priority": 3,
                            "version": "2",
                        },
                    ],
                },
                **RUNTIME_BASE_PARAMS,
            },
            {
                **{"name": f"{ModelFormat.CAIKIT}-{KServeDeploymentType.SERVERLESS.lower()}-1"},
                **SERVERLESS_ISVC_PARAMS,
            },
            {
                **{"name": f"{ModelFormat.CAIKIT}-{KServeDeploymentType.SERVERLESS.lower()}-2"},
                **SERVERLESS_ISVC_PARAMS,
            },
        )
    ],
    indirect=True,
)
class TestServerlessModelPriorities:
    def test_serverless_model_priorities_first_model(
        self,
        s3_models_inference_service,
        s3_models_second_inference_service,
    ):
        """Two models with the same runtime and priorities are set; verify query of first model"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTPS,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    def test_serverless_model_priorities_second_model(
        self,
        s3_models_inference_service,
        s3_models_second_inference_service,
    ):
        """Two models with the same runtime and priorities are set; verify query of second model"""
        verify_inference_response(
            inference_service=s3_models_second_inference_service,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTPS,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )
