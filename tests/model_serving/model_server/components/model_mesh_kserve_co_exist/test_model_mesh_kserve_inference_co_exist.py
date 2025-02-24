import pytest

from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    ModelAndFormat,
    ModelFormat,
    ModelInferenceRuntime,
    ModelStoragePath,
    ModelVersion,
    Protocols,
)
from utilities.inference_utils import Inference
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG, OPENVINO_KSERVE_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.modelmesh, pytest.mark.sanity]

KSERVE_RUNTIME_PARAMS = {
    "runtime-name": ModelInferenceRuntime.OPENVINO_KSERVE_RUNTIME,
    "model-format": {ModelAndFormat.OPENVINO_IR: ModelVersion.OPSET1},
}
KSERVE_ISVC_PARAMS = {
    "name": ModelFormat.OPENVINO,
    "model-version": ModelVersion.OPSET1,
    "model-dir": ModelStoragePath.KSERVE_OPENVINO_EXAMPLE_MODEL,
}
MODELMESH_ISVC_PARAMS = {
    "model-path": ModelStoragePath.OPENVINO_EXAMPLE_MODEL,
    "modelmesh-enabled": True,
}


@pytest.mark.parametrize(
    "model_namespace, openvino_kserve_serving_runtime, ovms_serverless_inference_service, "
    "http_s3_openvino_model_mesh_inference_service",
    [
        pytest.param(
            {"name": "serverless-model-mesh-openvino", "modelmesh-enabled": True},
            KSERVE_RUNTIME_PARAMS,
            KSERVE_ISVC_PARAMS,
            MODELMESH_ISVC_PARAMS,
        )
    ],
    indirect=True,
)
class TestOpenVINOServerlessModelMesh:
    def test_serverless_openvino_created_before_model_mesh_ns_rest_inference(
        self, ovms_serverless_inference_service, http_s3_openvino_model_mesh_inference_service
    ):
        """Verify that Serverless model can be queried when running with modelmesh inference service"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )

    def test_model_mesh_openvino_created_after_serverless_in_namespace_rest_inference(
        self, ovms_serverless_inference_service, http_s3_openvino_model_mesh_inference_service
    ):
        """Verify that modelmesh model can be queried when running with kserve inference service"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )


@pytest.mark.parametrize(
    "model_namespace, http_s3_openvino_model_mesh_inference_service, openvino_kserve_serving_runtime, "
    "ovms_serverless_inference_service, ",
    [
        pytest.param(
            {"name": "model-mesh-serverless-openvino", "modelmesh-enabled": True},
            MODELMESH_ISVC_PARAMS,
            KSERVE_RUNTIME_PARAMS,
            KSERVE_ISVC_PARAMS,
        )
    ],
    indirect=True,
)
class TestOpenVINOModelMeshServerless:
    def test_model_mesh_openvino_created_before_serverless_in_namespace_rest_inference(
        self, http_s3_openvino_model_mesh_inference_service, ovms_serverless_inference_service
    ):
        """Verify that modelmesh model can be queried when running with kserve inference service"""
        verify_inference_response(
            inference_service=http_s3_openvino_model_mesh_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTP,
            use_default_query=True,
        )

    def test_serverless_openvino_created_after_model_mesh_ns_rest_inference(
        self, http_s3_openvino_model_mesh_inference_service, ovms_serverless_inference_service
    ):
        """Verify that Serverless model can be queried when running with modelmesh inference service"""
        verify_inference_response(
            inference_service=ovms_serverless_inference_service,
            inference_config=OPENVINO_KSERVE_INFERENCE_CONFIG,
            inference_type=Inference.INFER,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
