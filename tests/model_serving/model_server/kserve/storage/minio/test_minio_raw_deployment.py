import pytest

from tests.model_serving.model_server.kserve.storage.minio.constants import (
    AGE_GENDER_INFERENCE_TYPE,
    MINIO_DATA_CONNECTION_CONFIG,
    MINIO_INFERENCE_CONFIG,
    MINIO_RUNTIME_CONFIG,
)
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import KServeDeploymentType, MinIo, Protocols
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG

pytestmark = [pytest.mark.rawdeployment, pytest.mark.minio]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, minio_pod, unprivileged_minio_data_connection, ovms_kserve_serving_runtime, "
    "kserve_ovms_minio_inference_service",
    [
        pytest.param(
            {"name": f"{MinIo.Metadata.NAME}-{KServeDeploymentType.RAW_DEPLOYMENT.lower()}"},
            MinIo.PodConfig.KSERVE_MINIO_CONFIG,
            MINIO_DATA_CONNECTION_CONFIG,
            MINIO_RUNTIME_CONFIG,
            {"deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT, "external-route": True, **MINIO_INFERENCE_CONFIG},
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestMinioRawDeployment:
    def test_minio_raw_inference(
        self,
        kserve_ovms_minio_inference_service,
    ) -> None:
        """Verify that kserve raw deployment minio model can be queried using REST"""
        verify_inference_response(
            inference_service=kserve_ovms_minio_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=AGE_GENDER_INFERENCE_TYPE,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
