import pytest

from tests.model_serving.model_server.components.constants import KSERVE_RUNTIME_PARAMS, SERVERLESS_ISVC_PARAMS
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
from utilities.inference_utils import Inference
from utilities.manifests.caikit_standalone import CAIKIT_STANDALONE_INFERENCE_CONFIG
from utilities.manifests.openvino import (
    OPENVINO_KSERVE_INFERENCE_CONFIG,
)

pytestmark = [pytest.mark.serverless, pytest.mark.rawdeployment, pytest.mark.sanity]

RAW_RUNTIME_PARAMS = {
    "name": f"{Protocols.HTTP}-{ModelInferenceRuntime.CAIKIT_STANDALONE_RUNTIME}".lower(),
    "template-name": RuntimeTemplates.CAIKIT_STANDALONE_SERVING,
    "multi-model": False,
    "enable-http": True,
}
RAW_ISVC_PARAMS = {
    "name": f"{ModelFormat.CAIKIT}-{KServeDeploymentType.RAW_DEPLOYMENT}".lower(),
    "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
    "model-dir": ModelStoragePath.EMBEDDING_MODEL,
}


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service, "
    "serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "serverless-raw-deployment"},
            KSERVE_RUNTIME_PARAMS,
            SERVERLESS_ISVC_PARAMS,
            RAW_RUNTIME_PARAMS,
            RAW_ISVC_PARAMS,
        )
    ],
    indirect=True,
)
class TestServerlessRawInternalDeploymentInferenceCoExist:
    def test_serverless_openvino_created_before_raw_internal_deployment_caikit_inference(
        self, ovms_kserve_inference_service, s3_models_inference_service
    ):
        """Verify that Serverless model can be queried when running with raw deployment inference service"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_raw_internal_deployment_caikit_created_after_serverless_in_namespace_rest_inference(
        self,
        ovms_kserve_inference_service,
        s3_models_inference_service,
    ):
        """Verify that raw deployment model can be queried when running with kserve inference service"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTP,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ovms_kserve_serving_runtime, ovms_kserve_inference_service, "
    "serving_runtime_from_template, s3_models_inference_service",
    [
        pytest.param(
            {"name": "serverless-raw-ext-deployment"},
            KSERVE_RUNTIME_PARAMS,
            SERVERLESS_ISVC_PARAMS,
            RAW_RUNTIME_PARAMS,
            {**RAW_ISVC_PARAMS, "external-route": True},
        ),
    ],
    indirect=True,
)
class TestServerlessRawExternalDeploymentInferenceCoExist:
    def test_serverless_openvino_created_before_raw_external_deployment_caikit_inference(
        self, ovms_kserve_inference_service, s3_models_inference_service
    ):
        """Verify that Serverless model can be queried when running with raw deployment inference service"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_raw_external_deployment_caikit_created_after_serverless_in_namespace_rest_inference(
        self,
        ovms_kserve_inference_service,
        s3_models_inference_service,
    ):
        """Verify that raw deployment model can be queried when running with kserve inference service"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTPS,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, s3_models_inference_service,"
    "ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "raw-deployment-serverless"},
            RAW_RUNTIME_PARAMS,
            RAW_ISVC_PARAMS,
            KSERVE_RUNTIME_PARAMS,
            SERVERLESS_ISVC_PARAMS,
        )
    ],
    indirect=True,
)
class TestRawInternalDeploymentServerlessInferenceCoExist:
    def test_raw_internal_deployment_caikit_created_before_serverless_openvino_in_namespace_rest_inference(
        self,
        s3_models_inference_service,
        ovms_kserve_inference_service,
    ):
        """Verify that raw deployment model can be queried when running with kserve inference service"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTP,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    def test_serverless_openvino_created_after_raw_internal_deployment_caikit_ns_rest_inference(
        self,
        s3_models_inference_service,
        ovms_kserve_inference_service,
    ):
        """Verify that Serverless model can be queried when running with raw deployment exists"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "unprivileged_model_namespace, serving_runtime_from_template, s3_models_inference_service,"
    "ovms_kserve_serving_runtime, ovms_kserve_inference_service",
    [
        pytest.param(
            {"name": "raw-etx-deployment-serverless"},
            RAW_RUNTIME_PARAMS,
            {**RAW_ISVC_PARAMS, "external-route": True},
            KSERVE_RUNTIME_PARAMS,
            SERVERLESS_ISVC_PARAMS,
        ),
    ],
    indirect=True,
)
class TestRawExternalDeploymentServerlessInferenceCoExist:
    def test_raw_external_deployment_caikit_created_before_serverless_openvino_in_namespace_rest_inference(
        self,
        s3_models_inference_service,
        ovms_kserve_inference_service,
    ):
        """Verify that raw deployment model can be queried when running with kserve inference service"""
        verify_inference_response(
            inference_service=s3_models_inference_service,
            inference_config=CAIKIT_STANDALONE_INFERENCE_CONFIG,
            inference_type="embedding",
            protocol=Protocols.HTTPS,
            model_name=ModelName.CAIKIT_BGE_LARGE_EN,
            use_default_query=True,
        )

    def test_serverless_openvino_created_after_raw_external_deployment_caikit_ns_rest_inference(
        self, s3_models_inference_service, ovms_kserve_inference_service
    ):
        """Verify that Serverless model can be queried when running with raw deployment exists"""
        verify_inference_response(
            inference_service=ovms_kserve_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
