"""Layer A (tier1, slow) end-to-end ConnectionsAPI tests for LLMInferenceService (manual 1.6-1.9).

All tests run TinyLlama-1.1B on CPU vLLM (`CpuConfig`) — no GPU node is required. Each
CREATE/UPDATE-inject test drives the LLMInferenceService to `Ready` and then sends a real chat
completion request, proving the injected connection actually wired the model's data source. The
removal test (1.9) does not assert inference — the service is expected to become not-Ready.
"""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret

from tests.model_serving.model_server.connections_api.constants import (
    CONNECTION_PATH_ANNOTATION,
    CONNECTIONS_ANNOTATION,
    LLMISVC_NAMESPACE,
    LLMISVC_OCI_MODEL_URI,
    LLMISVC_S3_CONNECTION_PATH,
    LLMISVC_URI_MODEL_URI,
)
from tests.model_serving.model_server.connections_api.utils import (
    assert_llmisvc_connection_cleared,
    assert_llmisvc_oci_injected,
    assert_llmisvc_s3_injected,
    assert_llmisvc_uri_injected,
    assert_service_account_exists,
    create_connection_llmisvc,
    run_llmisvc_inference,
    wait_for_llmisvc_connection_cleared,
)
from utilities.constants import Timeout

pytestmark = [pytest.mark.tier1, pytest.mark.slow]


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param({"name": LLMISVC_NAMESPACE})],
    indirect=True,
)
class TestConnectionsApiLlmisvc:
    """ConnectionsAPI injection for LLMInferenceService, driven to Ready with real inference.

    Steps:
        1. Create (or update) an LLMInferenceService annotated with `opendatahub.io/connections`
           (and `opendatahub.io/connection-path` for S3), on the CPU vLLM TinyLlama template.
        2. Wait for the LLMInferenceService to reach `Ready`.
        3. Assert the odh-model-controller webhook injected the expected template/model spec
           fields (and, for S3, that the `{secret}-sa` ServiceAccount was created).
        4. Send a real chat completion request and assert a successful response.
    """

    @pytest.mark.parametrize(
        "s3_connection_secret",
        [pytest.param({"namespace_fixture": "unprivileged_model_namespace", "name": "llmisvc-s3-connection-secret"})],
        indirect=True,
    )
    def test_llmisvc_s3_create_injects_sa_and_uri(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        models_s3_bucket_name: str,
        s3_connection_secret: Secret,
    ) -> None:
        """Manual 1.6: S3 CREATE injects template.serviceAccountName + model.uri, SA exists, infers."""
        with create_connection_llmisvc(
            client=admin_client,
            name="llmisvc-s3-connection",
            namespace=unprivileged_model_namespace.name,
            connections=s3_connection_secret.name,
            connection_path=LLMISVC_S3_CONNECTION_PATH,
        ) as llmisvc:
            expected_uri = f"s3://{models_s3_bucket_name}/{LLMISVC_S3_CONNECTION_PATH}"
            assert_llmisvc_s3_injected(
                llmisvc=llmisvc, secret_name=s3_connection_secret.name, expected_uri=expected_uri
            )
            assert_service_account_exists(
                client=admin_client,
                namespace=unprivileged_model_namespace.name,
                name=f"{s3_connection_secret.name}-sa",
            )
            run_llmisvc_inference(llmisvc=llmisvc)

    @pytest.mark.skip_on_disconnected
    @pytest.mark.parametrize(
        "uri_connection_secret_llmisvc",
        [pytest.param({"namespace_fixture": "unprivileged_model_namespace", "name": "llmisvc-uri-connection-secret"})],
        indirect=True,
    )
    def test_llmisvc_uri_create_injects_uri(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        uri_connection_secret_llmisvc: Secret,
    ) -> None:
        """Manual 1.7: URI CREATE injects model.uri from the connection Secret, infers."""
        with create_connection_llmisvc(
            client=admin_client,
            name="llmisvc-uri-connection",
            namespace=unprivileged_model_namespace.name,
            connections=uri_connection_secret_llmisvc.name,
        ) as llmisvc:
            assert_llmisvc_uri_injected(llmisvc=llmisvc, expected_uri=LLMISVC_URI_MODEL_URI)
            run_llmisvc_inference(llmisvc=llmisvc)

    @pytest.mark.parametrize(
        "oci_connection_secret",
        [pytest.param({"namespace_fixture": "unprivileged_model_namespace", "name": "llmisvc-oci-connection-secret"})],
        indirect=True,
    )
    def test_llmisvc_oci_create_injects_image_pull_secrets(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        oci_connection_secret: Secret,
    ) -> None:
        """Manual 1.8: OCI CREATE injects template.imagePullSecrets from the connection Secret, infers."""
        with create_connection_llmisvc(
            client=admin_client,
            name="llmisvc-oci-connection",
            namespace=unprivileged_model_namespace.name,
            connections=oci_connection_secret.name,
            model_uri=LLMISVC_OCI_MODEL_URI,
        ) as llmisvc:
            assert_llmisvc_oci_injected(llmisvc=llmisvc, secret_name=oci_connection_secret.name)
            run_llmisvc_inference(llmisvc=llmisvc)

    @pytest.mark.parametrize(
        "s3_connection_secret",
        [
            pytest.param({
                "namespace_fixture": "unprivileged_model_namespace",
                "name": "llmisvc-remove-connection-secret",
            })
        ],
        indirect=True,
    )
    def test_llmisvc_remove_connection_clears_model(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        s3_connection_secret: Secret,
    ) -> None:
        """Manual 1.9: removing the connection annotation clears the SA name and `spec.model` entirely."""
        with create_connection_llmisvc(
            client=admin_client,
            name="llmisvc-remove-connection",
            namespace=unprivileged_model_namespace.name,
            connections=s3_connection_secret.name,
            connection_path=LLMISVC_S3_CONNECTION_PATH,
        ) as llmisvc:
            assert llmisvc.instance.spec.template.get("serviceAccountName") == f"{s3_connection_secret.name}-sa"

            resource_dict: dict[str, Any] = {
                "metadata": {
                    "name": llmisvc.name,
                    "annotations": {
                        CONNECTIONS_ANNOTATION: None,
                        CONNECTION_PATH_ANNOTATION: None,
                    },
                }
            }
            llmisvc.update(resource_dict=resource_dict)

            wait_for_llmisvc_connection_cleared(llmisvc=llmisvc, timeout=Timeout.TIMEOUT_2MIN)
            assert_llmisvc_connection_cleared(llmisvc=llmisvc)
            llmisvc.wait_for_condition(
                condition=llmisvc.Condition.READY,
                status=llmisvc.Condition.Status.FALSE,
                timeout=Timeout.TIMEOUT_2MIN,
            )
