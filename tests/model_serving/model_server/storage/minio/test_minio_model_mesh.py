import pytest

from tests.model_serving.model_server.storage.minio.constants import (
    MINIO_DATA_CONNECTION_CONFIG,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import (
    KServeDeploymentType,
    MinIo,
    ModelAndFormat,
    ModelFormat,
    ModelName,
    Protocols,
    RunTimeConfigs,
)
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG

pytestmark = [pytest.mark.modelmesh, pytest.mark.minio, pytest.mark.sanity]


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, http_s3_ovms_model_mesh_serving_runtime, "
    "model_mesh_ovms_minio_inference_service",
    [
        pytest.param(
            {
                "name": f"{MinIo.Metadata.NAME}-{KServeDeploymentType.MODEL_MESH.lower()}",
                "modelmesh-enabled": True,
            },
            MinIo.PodConfig.KSERVE_MINIO_CONFIG,
            MINIO_DATA_CONNECTION_CONFIG,
            {"runtime_image": MinIo.PodConfig.KSERVE_MINIO_IMAGE, **RunTimeConfigs.ONNX_OPSET13_RUNTIME_CONFIG},
            {
                "name": f"{ModelName.MNIST}-model",
                "model-format": ModelAndFormat.OPENVINO_IR,
                "model-version": "1",
                "model-dir": f"modelmesh/{ModelFormat.ONNX}",
                "deployment-mode": KServeDeploymentType.MODEL_MESH,
            },
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestMinioModelMesh:
    def test_minio_model_mesh_inference(
        self,
        model_mesh_ovms_minio_inference_service,
    ) -> None:
        """Verify that model mesh minio model can be queried using REST"""
        verify_inference_response(
            inference_service=model_mesh_ovms_minio_inference_service,
            inference_config=ONNX_INFERENCE_CONFIG,
            inference_type=f"infer-{ModelName.MNIST}",
            protocol=Protocols.HTTP,
            use_default_query=True,
        )
