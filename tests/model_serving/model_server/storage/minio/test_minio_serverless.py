import pytest

from tests.model_serving.model_server.storage.minio.constants import (
    AGE_GENDER_INFERENCE_TYPE,
    MINIO_DATA_CONNECTION_CONFIG,
    MINIO_INFERENCE_CONFIG,
    MINIO_RUNTIME_CONFIG,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import KServeDeploymentType, MinIo, Protocols
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG

pytestmark = [pytest.mark.serverless, pytest.mark.minio, pytest.mark.sanity]


@pytest.mark.parametrize(
    "model_namespace, minio_pod, minio_data_connection, ovms_kserve_serving_runtime, "
    "kserve_ovms_minio_inference_service",
    [
        pytest.param(
            {"name": f"{MinIo.Metadata.NAME}-{KServeDeploymentType.SERVERLESS.lower()}"},
            MinIo.PodConfig.KSERVE_MINIO_CONFIG,
            MINIO_DATA_CONNECTION_CONFIG,
            MINIO_RUNTIME_CONFIG,
            {"deployment-mode": KServeDeploymentType.SERVERLESS, **MINIO_INFERENCE_CONFIG},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestMinioServerless:
    def test_minio_serverless_inference(
        self,
        kserve_ovms_minio_inference_service,
    ) -> None:
        """Verify that kserve Serverless minio model can be queried using REST"""
        verify_inference_response(
            inference_service=kserve_ovms_minio_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=AGE_GENDER_INFERENCE_TYPE,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
