"""Layer A (tier1) end-to-end ConnectionsAPI tests for InferenceService (manual Tests 1.1-1.5).

Each CREATE/UPDATE-inject test drives an sklearn/onnx MLServer InferenceService to `Ready` and
then issues a real v2 inference request, proving the injected connection actually wired the
model's data source (not merely that the pod passed its probes). The removal test (1.5) does not
assert inference — an sklearn server with no storage may still pass probes.
"""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.connections_api.constants import (
    CONNECTION_PATH_ANNOTATION,
    CONNECTIONS_ANNOTATION,
    ISVC_NAMESPACE,
    ISVC_OCI_MODEL_FORMAT,
    ISVC_OCI_STORAGE_URI,
    ISVC_S3_CONNECTION_PATH,
    ISVC_S3_MODEL_FORMAT,
    ISVC_URI_MODEL_FORMAT,
    ISVC_URI_MODEL_URI,
    ISVC_URI_ONNX_REST_INPUT_QUERY,
)
from tests.model_serving.model_server.connections_api.utils import (
    assert_isvc_connection_cleared,
    assert_isvc_oci_injected,
    assert_isvc_s3_injected,
    assert_isvc_uri_injected,
    assert_service_account_exists,
    run_isvc_inference,
    wait_for_isvc_connection_cleared,
)
from utilities.constants import KServeDeploymentType, Timeout
from utilities.inference_utils import create_isvc

pytestmark = [pytest.mark.tier1]


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": ISVC_NAMESPACE})],
    indirect=True,
)
@pytest.mark.parametrize(
    "mlserver_serving_runtime",
    [pytest.param({"deployment_mode": KServeDeploymentType.STANDARD})],
    indirect=True,
)
class TestConnectionsApiIsvc:
    """ConnectionsAPI injection for InferenceService, driven to Ready with real inference.

    Steps:
        1. Create (or update) an InferenceService annotated with `opendatahub.io/connections`
           (and `opendatahub.io/connection-path` for S3).
        2. Wait for the InferenceService to reach `Ready`.
        3. Assert the odh-model-controller webhook injected the expected predictor spec fields
           (and, for S3, that the `{secret}-sa` ServiceAccount was created).
        4. Send a real v2 inference request and assert a successful response.
    """

    @pytest.mark.parametrize(
        "s3_connection_secret",
        [pytest.param({"namespace_fixture": "model_namespace", "name": "isvc-s3-connection-secret"})],
        indirect=True,
    )
    def test_isvc_s3_create_injects_storage_and_sa(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        s3_connection_secret: Secret,
    ) -> None:
        """Manual 1.1: S3 CREATE injects SA + storage.key + storage.path, SA exists, Ready, infers."""
        # The ISVC name must equal the model name MLServer registers from the bundled
        # model-settings.json under the S3 path ("sklearn") — KServe's readiness/liveness probes
        # query /v2/models/{isvc-name}/ready, so a mismatched name 404s forever.
        with create_isvc(
            client=admin_client,
            name=ISVC_S3_MODEL_FORMAT,
            namespace=model_namespace.name,
            model_format=ISVC_S3_MODEL_FORMAT,
            runtime=mlserver_serving_runtime.name,
            connections=s3_connection_secret.name,
            connection_path=ISVC_S3_CONNECTION_PATH,
            timeout=Timeout.TIMEOUT_10MIN,
        ) as isvc:
            assert_isvc_s3_injected(
                isvc=isvc, secret_name=s3_connection_secret.name, expected_path=ISVC_S3_CONNECTION_PATH
            )
            assert_service_account_exists(
                client=admin_client, namespace=model_namespace.name, name=f"{s3_connection_secret.name}-sa"
            )
            run_isvc_inference(isvc=isvc, model_format=ISVC_S3_MODEL_FORMAT)

    @pytest.mark.skip_on_disconnected
    @pytest.mark.parametrize(
        "uri_connection_secret_isvc",
        [pytest.param({"namespace_fixture": "model_namespace", "name": "isvc-uri-connection-secret"})],
        indirect=True,
    )
    def test_isvc_uri_create_injects_storage_uri(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        uri_connection_secret_isvc: Secret,
    ) -> None:
        """Manual 1.2: URI CREATE injects storageUri from the connection Secret, Ready, infers."""
        with create_isvc(
            client=admin_client,
            name="isvc-uri-connection",
            namespace=model_namespace.name,
            model_format=ISVC_URI_MODEL_FORMAT,
            runtime=mlserver_serving_runtime.name,
            connections=uri_connection_secret_isvc.name,
            timeout=Timeout.TIMEOUT_10MIN,
        ) as isvc:
            assert_isvc_uri_injected(isvc=isvc, expected_uri=ISVC_URI_MODEL_URI)
            run_isvc_inference(
                isvc=isvc, model_format=ISVC_URI_MODEL_FORMAT, input_query=ISVC_URI_ONNX_REST_INPUT_QUERY
            )

    @pytest.mark.parametrize(
        "oci_connection_secret",
        [pytest.param({"namespace_fixture": "model_namespace", "name": "isvc-oci-connection-secret"})],
        indirect=True,
    )
    def test_isvc_oci_create_injects_image_pull_secrets(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        oci_connection_secret: Secret,
    ) -> None:
        """Manual 1.3: OCI CREATE injects imagePullSecrets from the connection Secret, Ready, infers."""
        with create_isvc(
            client=admin_client,
            name=ISVC_OCI_MODEL_FORMAT,
            namespace=model_namespace.name,
            model_format=ISVC_OCI_MODEL_FORMAT,
            runtime=mlserver_serving_runtime.name,
            storage_uri=ISVC_OCI_STORAGE_URI,
            connections=oci_connection_secret.name,
            timeout=Timeout.TIMEOUT_10MIN,
        ) as isvc:
            assert_isvc_oci_injected(isvc=isvc, secret_name=oci_connection_secret.name)
            run_isvc_inference(isvc=isvc, model_format=ISVC_OCI_MODEL_FORMAT)

    @pytest.mark.parametrize(
        "s3_connection_secret",
        [pytest.param({"namespace_fixture": "model_namespace", "name": "isvc-update-inject-secret"})],
        indirect=True,
    )
    def test_isvc_update_add_connection_injects(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        s3_connection_secret: Secret,
    ) -> None:
        """Manual 1.4: adding a connection annotation on UPDATE injects on the admission call."""
        with create_isvc(
            client=admin_client,
            name=ISVC_S3_MODEL_FORMAT,
            namespace=model_namespace.name,
            model_format=ISVC_S3_MODEL_FORMAT,
            runtime=mlserver_serving_runtime.name,
            wait=False,
            wait_for_predictor_pods=False,
        ) as isvc:
            assert not isvc.instance.spec.predictor.get("serviceAccountName"), (
                "InferenceService should not have a serviceAccountName before any connection is injected"
            )

            resource_dict: dict[str, Any] = {
                "metadata": {
                    "name": isvc.name,
                    "annotations": {
                        CONNECTIONS_ANNOTATION: s3_connection_secret.name,
                        CONNECTION_PATH_ANNOTATION: ISVC_S3_CONNECTION_PATH,
                    },
                }
            }
            isvc.update(resource_dict=resource_dict)

            isvc.wait_for_condition(
                condition=isvc.Condition.READY,
                status=isvc.Condition.Status.TRUE,
                timeout=Timeout.TIMEOUT_10MIN,
            )
            assert_isvc_s3_injected(
                isvc=isvc, secret_name=s3_connection_secret.name, expected_path=ISVC_S3_CONNECTION_PATH
            )
            assert_service_account_exists(
                client=admin_client, namespace=model_namespace.name, name=f"{s3_connection_secret.name}-sa"
            )
            run_isvc_inference(isvc=isvc, model_format=ISVC_S3_MODEL_FORMAT)

    @pytest.mark.parametrize(
        "s3_connection_secret",
        [pytest.param({"namespace_fixture": "model_namespace", "name": "isvc-remove-connection-secret"})],
        indirect=True,
    )
    def test_isvc_remove_connection_clears_fields(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        s3_connection_secret: Secret,
    ) -> None:
        """Manual 1.5: removing the connection annotation clears SA name + storage (no inference check)."""
        with create_isvc(
            client=admin_client,
            name=ISVC_S3_MODEL_FORMAT,
            namespace=model_namespace.name,
            model_format=ISVC_S3_MODEL_FORMAT,
            runtime=mlserver_serving_runtime.name,
            connections=s3_connection_secret.name,
            connection_path=ISVC_S3_CONNECTION_PATH,
            timeout=Timeout.TIMEOUT_10MIN,
        ) as isvc:
            assert_isvc_s3_injected(
                isvc=isvc, secret_name=s3_connection_secret.name, expected_path=ISVC_S3_CONNECTION_PATH
            )

            resource_dict: dict[str, Any] = {
                "metadata": {
                    "name": isvc.name,
                    "annotations": {
                        CONNECTIONS_ANNOTATION: None,
                        CONNECTION_PATH_ANNOTATION: None,
                    },
                }
            }
            isvc.update(resource_dict=resource_dict)

            wait_for_isvc_connection_cleared(isvc=isvc, timeout=Timeout.TIMEOUT_2MIN)
            assert_isvc_connection_cleared(isvc=isvc)
