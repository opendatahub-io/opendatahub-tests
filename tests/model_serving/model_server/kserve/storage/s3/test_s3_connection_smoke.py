import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.storage.s3.constants import (
    MINIO_CONNECTION_DATA_CONNECTION_CONFIG,
    MINIO_INFERENCE_CONFIG,
    MINIO_RUNTIME_CONFIG,
)
from tests.model_serving.model_server.kserve.storage.utils import assert_isvc_s3_fully_injected
from utilities.constants import KServeDeploymentType, MinIo
from utilities.inference_utils import create_isvc

pytestmark = [pytest.mark.smoke]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, minio_pod, unprivileged_minio_data_connection, ovms_kserve_serving_runtime",
    [
        pytest.param(
            {"name": f"{MinIo.Metadata.NAME}-connection-smoke"},
            MinIo.PodConfig.KSERVE_MINIO_CONFIG,
            MINIO_CONNECTION_DATA_CONNECTION_CONFIG,
            MINIO_RUNTIME_CONFIG,
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestS3ConnectionSmoke:
    """Layer B: injection-only smoke check for the S3 ConnectionsAPI CREATE path.

    No readiness wait, no reachable model data — only asserts that the odh-model-controller
    ConnectionsAPI webhook injected the expected fields, giving a fast (<2 min) build-gating signal
    isolated from backend/model-load flakiness. S3 CREATE is the richest injection path (both
    `serviceAccountName`/`storage` spec fields and the `{secret}-sa` ServiceAccount side effect),
    giving the strongest single-test regression signal for RHOAIENG-65587.
    """

    def test_smoke_isvc_s3_create_injects(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        unprivileged_minio_data_connection: Secret,
        ovms_kserve_serving_runtime: ServingRuntime,
    ) -> None:
        """Test steps:

        1. Create an InferenceService with the S3 connection set, without waiting for readiness.
        2. Assert the webhook injected the storage/SA fields.
        """
        with create_isvc(
            client=unprivileged_client,
            name="isvc-s3-connection-smoke",
            namespace=unprivileged_model_namespace.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            model_format=MINIO_INFERENCE_CONFIG["model-format"],
            runtime=ovms_kserve_serving_runtime.name,
            model_version=MINIO_INFERENCE_CONFIG["model-version"],
            connections=unprivileged_minio_data_connection.name,
            connection_path=MINIO_INFERENCE_CONFIG["model-dir"],
            wait=False,
            wait_for_predictor_pods=False,
        ) as isvc:
            assert_isvc_s3_fully_injected(
                client=unprivileged_client,
                isvc=isvc,
                namespace=unprivileged_model_namespace.name,
                secret_name=unprivileged_minio_data_connection.name,
                expected_path=MINIO_INFERENCE_CONFIG["model-dir"],
            )
