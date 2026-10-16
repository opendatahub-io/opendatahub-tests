import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.model_serving.model_server.kserve.storage.s3.constants import (
    AGE_GENDER_INFERENCE_TYPE,
    MINIO_CONNECTION_DATA_CONNECTION_CONFIG,
    MINIO_DATA_CONNECTION_CONFIG,
    MINIO_INFERENCE_CONFIG,
    MINIO_RUNTIME_CONFIG,
)
from tests.model_serving.model_server.kserve.storage.utils import assert_isvc_s3_fully_injected
from tests.model_serving.model_server.utils import verify_inference_response
from utilities.constants import KServeDeploymentType, MinIo, Protocols
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG

pytestmark = [pytest.mark.tier1, pytest.mark.rawdeployment, pytest.mark.minio]


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
            id="static",
        ),
        pytest.param(
            {"name": f"{MinIo.Metadata.NAME}-{KServeDeploymentType.RAW_DEPLOYMENT.lower()}-connection"},
            MinIo.PodConfig.KSERVE_MINIO_CONFIG,
            MINIO_CONNECTION_DATA_CONNECTION_CONFIG,
            MINIO_RUNTIME_CONFIG,
            {
                "deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "external-route": True,
                "storage-strategy": "connection",
                **MINIO_INFERENCE_CONFIG,
            },
            id="connection",
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestMinioRawDeployment:
    """Validate KServe raw deployment model inference using MinIO as the S3 storage backend.

    Steps:
        1. Deploy a MinIO pod and configure a data connection for model storage (static
           `storage_key`, or a ConnectionsAPI connection Secret for the `connection` param —
           manual 1.1).
        2. Deploy an OVMS inference service as a raw deployment pointing to MinIO.
        3. For the `connection` param, assert the ConnectionsAPI webhook injected the expected
           storage/SA fields.
        4. Send a REST inference request and verify a successful response over HTTPS.
    """

    def test_minio_raw_inference(
        self,
        request: FixtureRequest,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        unprivileged_minio_data_connection: Secret,
        kserve_ovms_minio_inference_service,
    ) -> None:
        """Verify that kserve raw deployment minio model can be queried using REST"""
        isvc_param = request.node.callspec.params["kserve_ovms_minio_inference_service"]
        if isvc_param.get("storage-strategy") == "connection":
            assert_isvc_s3_fully_injected(
                client=unprivileged_client,
                isvc=kserve_ovms_minio_inference_service,
                namespace=unprivileged_model_namespace.name,
                secret_name=unprivileged_minio_data_connection.name,
                expected_path=MINIO_INFERENCE_CONFIG["model-dir"],
            )

        verify_inference_response(
            inference_service=kserve_ovms_minio_inference_service,
            inference_config=OPENVINO_INFERENCE_CONFIG,
            inference_type=AGE_GENDER_INFERENCE_TYPE,
            protocol=Protocols.HTTPS,
            use_default_query=True,
        )
