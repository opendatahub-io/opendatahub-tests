import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.storage.s3.constants import (
    AGE_GENDER_INFERENCE_TYPE,
    MINIO_CONNECTION_DATA_CONNECTION_CONFIG,
    MINIO_INFERENCE_CONFIG,
    MINIO_RUNTIME_CONFIG,
)
from tests.model_serving.model_server.kserve.storage.utils import (
    assert_isvc_connection_cleared,
    assert_isvc_s3_fully_injected,
    wait_for_isvc_connection_cleared,
)
from tests.model_serving.model_server.utils import (
    add_connection_annotations,
    remove_connection_annotations,
    verify_inference_response,
)
from utilities.constants import KServeDeploymentType, MinIo, Protocols, Timeout
from utilities.inference_utils import create_isvc
from utilities.manifests.openvino import OPENVINO_INFERENCE_CONFIG

pytestmark = [pytest.mark.tier1, pytest.mark.rawdeployment, pytest.mark.minio]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, minio_pod, unprivileged_minio_data_connection, ovms_kserve_serving_runtime",
    [
        pytest.param(
            {"name": f"{MinIo.Metadata.NAME}-connection-update"},
            MinIo.PodConfig.KSERVE_MINIO_CONFIG,
            MINIO_CONNECTION_DATA_CONNECTION_CONFIG,
            MINIO_RUNTIME_CONFIG,
        )
    ],
    indirect=True,
)
@pytest.mark.usefixtures("minio_pod")
class TestS3ConnectionUpdate:
    """Manual 1.4/1.5: UPDATE inject/remove of the S3 ConnectionsAPI annotation on an ISVC."""

    def test_s3_update_add_connection_injects(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        unprivileged_minio_data_connection: Secret,
        ovms_kserve_serving_runtime: ServingRuntime,
    ) -> None:
        """Test steps:

        1. Create a bare InferenceService with no storage configuration.
        2. Assert no ServiceAccount is pre-attached.
        3. Add the `opendatahub.io/connections` (+ `connection-path`) annotations, exercising the
           UPDATE admission path.
        4. Wait for Ready, assert the webhook injected the storage/SA fields, and verify inference.
        """
        with create_isvc(
            client=unprivileged_client,
            name="isvc-s3-update-inject",
            namespace=unprivileged_model_namespace.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            model_format=MINIO_INFERENCE_CONFIG["model-format"],
            runtime=ovms_kserve_serving_runtime.name,
            model_version=MINIO_INFERENCE_CONFIG["model-version"],
            external_route=True,
            wait=False,
            wait_for_predictor_pods=False,
        ) as isvc:
            sa_name = isvc.instance.spec.predictor.get("serviceAccountName")
            assert not sa_name, f"Expected no serviceAccountName before connection is added, got {sa_name!r}"

            add_connection_annotations(
                resource=isvc,
                connections=unprivileged_minio_data_connection.name,
                connection_path=MINIO_INFERENCE_CONFIG["model-dir"],
            )
            isvc.wait_for_condition(condition="Ready", status="True", timeout=Timeout.TIMEOUT_10MIN)

            assert_isvc_s3_fully_injected(
                client=unprivileged_client,
                isvc=isvc,
                namespace=unprivileged_model_namespace.name,
                secret_name=unprivileged_minio_data_connection.name,
                expected_path=MINIO_INFERENCE_CONFIG["model-dir"],
            )
            verify_inference_response(
                inference_service=isvc,
                inference_config=OPENVINO_INFERENCE_CONFIG,
                inference_type=AGE_GENDER_INFERENCE_TYPE,
                protocol=Protocols.HTTPS,
                use_default_query=True,
            )

    def test_s3_remove_connection_clears_fields(
        self,
        unprivileged_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        unprivileged_minio_data_connection: Secret,
        ovms_kserve_serving_runtime: ServingRuntime,
    ) -> None:
        """Test steps:

        1. Create an InferenceService with the S3 connection set from CREATE.
        2. Assert the webhook injected the storage/SA fields.
        3. Remove the ConnectionsAPI annotations, exercising the UPDATE-remove path.
        4. Assert the injected fields are cleared (no inference assertion — manual 1.5 documents
           that the server may remain Ready without a working model).
        """
        with create_isvc(
            client=unprivileged_client,
            name="isvc-s3-update-remove",
            namespace=unprivileged_model_namespace.name,
            deployment_mode=KServeDeploymentType.RAW_DEPLOYMENT,
            model_format=MINIO_INFERENCE_CONFIG["model-format"],
            runtime=ovms_kserve_serving_runtime.name,
            model_version=MINIO_INFERENCE_CONFIG["model-version"],
            external_route=True,
            connections=unprivileged_minio_data_connection.name,
            connection_path=MINIO_INFERENCE_CONFIG["model-dir"],
        ) as isvc:
            assert_isvc_s3_fully_injected(
                client=unprivileged_client,
                isvc=isvc,
                namespace=unprivileged_model_namespace.name,
                secret_name=unprivileged_minio_data_connection.name,
                expected_path=MINIO_INFERENCE_CONFIG["model-dir"],
            )

            remove_connection_annotations(resource=isvc)
            wait_for_isvc_connection_cleared(isvc=isvc)
            assert_isvc_connection_cleared(isvc=isvc)
