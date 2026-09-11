"""Layer B (smoke) injection-only ConnectionsAPI tests for InferenceService and LLMInferenceService.

Two fast tests, additional to the manual test plan, that assert only the webhook-mutated spec (no
readiness wait, no GPU, no reachable model data). Both use the S3 CREATE case deliberately: it is
the richest injection path in a single admission call — ServiceAccount name, storage/model URI
computation, and the `{secret}-sa` ServiceAccount creation side effect — giving the strongest
single-test regression signal for the defect class of RHOAIENG-65587 (webhook silently not
injecting).
"""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.connections_api.constants import (
    ISVC_S3_CONNECTION_PATH,
    ISVC_S3_MODEL_FORMAT,
    LLMISVC_S3_CONNECTION_PATH,
    SMOKE_ISVC_NAMESPACE,
    SMOKE_LLMISVC_NAMESPACE,
)
from tests.model_serving.model_server.connections_api.utils import (
    assert_isvc_s3_injected,
    assert_llmisvc_s3_injected,
    assert_service_account_exists,
    create_connection_llmisvc,
)
from utilities.constants import KServeDeploymentType
from utilities.inference_utils import create_isvc

pytestmark = [pytest.mark.smoke, pytest.mark.parallel]


@pytest.mark.parametrize(
    "model_namespace",
    [pytest.param({"name": SMOKE_ISVC_NAMESPACE})],
    indirect=True,
)
@pytest.mark.parametrize(
    "mlserver_serving_runtime",
    [pytest.param({"deployment_mode": KServeDeploymentType.STANDARD})],
    indirect=True,
)
@pytest.mark.parametrize(
    "s3_connection_secret",
    [pytest.param({"namespace_fixture": "model_namespace", "name": "smoke-isvc-s3-connection-secret"})],
    indirect=True,
)
class TestConnectionsApiSmokeIsvc:
    """Injection-only S3 CREATE smoke test for InferenceService (no readiness wait)."""

    def test_smoke_isvc_s3_create_injects(
        self,
        admin_client: DynamicClient,
        model_namespace: Namespace,
        mlserver_serving_runtime: ServingRuntime,
        s3_connection_secret: Secret,
    ) -> None:
        """Given an S3 connection Secret, when an ISVC is created, then storage/SA fields are injected immediately."""
        with create_isvc(
            client=admin_client,
            name="smoke-isvc-s3-connection",
            namespace=model_namespace.name,
            model_format=ISVC_S3_MODEL_FORMAT,
            runtime=mlserver_serving_runtime.name,
            connections=s3_connection_secret.name,
            connection_path=ISVC_S3_CONNECTION_PATH,
            wait=False,
            wait_for_predictor_pods=False,
        ) as isvc:
            assert_isvc_s3_injected(
                isvc=isvc, secret_name=s3_connection_secret.name, expected_path=ISVC_S3_CONNECTION_PATH
            )
            assert_service_account_exists(
                client=admin_client, namespace=model_namespace.name, name=f"{s3_connection_secret.name}-sa"
            )


@pytest.mark.parametrize(
    "unprivileged_model_namespace",
    [pytest.param({"name": SMOKE_LLMISVC_NAMESPACE})],
    indirect=True,
)
@pytest.mark.parametrize(
    "s3_connection_secret",
    [
        pytest.param({
            "namespace_fixture": "unprivileged_model_namespace",
            "name": "smoke-llmisvc-s3-connection-secret",
        })
    ],
    indirect=True,
)
class TestConnectionsApiSmokeLlmisvc:
    """Injection-only S3 CREATE smoke test for LLMInferenceService (no readiness wait, no GPU)."""

    def test_smoke_llmisvc_s3_create_injects(
        self,
        admin_client: DynamicClient,
        unprivileged_model_namespace: Namespace,
        models_s3_bucket_name: str,
        s3_connection_secret: Secret,
    ) -> None:
        """Given an S3 connection Secret, when an LLMISVC is created, then template/model fields are injected immediately."""
        with create_connection_llmisvc(
            client=admin_client,
            name="smoke-llmisvc-s3-connection",
            namespace=unprivileged_model_namespace.name,
            connections=s3_connection_secret.name,
            connection_path=LLMISVC_S3_CONNECTION_PATH,
            wait=False,
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
