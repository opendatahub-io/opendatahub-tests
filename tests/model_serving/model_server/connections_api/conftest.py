"""Fixtures for the ConnectionsAPI end-to-end test suite (InferenceService and LLMInferenceService).

Connection-Secret fixtures are parametrized indirectly with `request.param["namespace_fixture"]`
— the name of the namespace fixture to resolve at runtime (`model_namespace` for the ISVC layer,
which the reused `mlserver_serving_runtime` fixture requires; `unprivileged_model_namespace` for
the LLMISVC and smoke layers, matching the llmd suite convention) — so the same Secret fixtures
are reusable across all three test modules despite their differing namespace fixtures.
"""

from collections.abc import Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret

# `mlserver_serving_runtime` is defined under `tests/model_serving/model_runtime/mlserver/`, a
# sibling subtree that pytest does not auto-discover for tests under `model_server/`. Re-exporting
# the imported fixture here (rather than redefining a new ServingRuntime fixture) is how pytest
# shares a fixture across non-overlapping conftest.py trees.
from tests.model_serving.model_runtime.mlserver.conftest import (
    mlserver_serving_runtime,  # noqa: F401
)
from tests.model_serving.model_server.connections_api.constants import (
    OCI_CONNECTION_SECRET_NAME,
    S3_CONNECTION_SECRET_NAME,
    URI_CONNECTION_SECRET_NAME,
)
from tests.model_serving.model_server.connections_api.utils import (
    assert_connections_api_webhooks_configured,
    create_oci_connection_secret,
    create_uri_connection_secret,
)
from utilities.infra import s3_endpoint_secret


@pytest.fixture(scope="session", autouse=True)
def connections_api_webhooks_guard(admin_client: DynamicClient) -> None:
    """Session-wide precondition: fail the whole suite fast if ConnectionsAPI webhooks are wrong.

    A missing/incorrect webhook configuration is a platform-setup defect — exactly the class of
    regression this suite exists to catch (RHOAIENG-65587) — so it must surface as a failure
    rather than a silent skip.
    """
    assert_connections_api_webhooks_configured(client=admin_client)


@pytest.fixture(scope="class")
def s3_connection_secret(
    request: FixtureRequest,
    admin_client: DynamicClient,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret]:
    """S3 connection Secret, annotated for both the ConnectionsAPI webhook and secret_controller.

    Reuses `utilities.infra.s3_endpoint_secret`, which already sets the
    `opendatahub.io/managed`/`opendatahub.io/dashboard` labels and the
    `opendatahub.io/connection-type(-protocol)` annotations needed for the odh-model-controller
    `secret_controller` to auto-create `storage-config` (required so KServe's pod-mutator accepts
    the predictor pod).
    """
    namespace = request.getfixturevalue(argname=request.param["namespace_fixture"])
    with s3_endpoint_secret(
        client=admin_client,
        name=request.param.get("name", S3_CONNECTION_SECRET_NAME),
        namespace=namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
        aws_s3_region=models_s3_bucket_region,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def uri_connection_secret(request: FixtureRequest, admin_client: DynamicClient) -> Generator[Secret]:
    """`uri`-typed connection Secret pointing at the `request.param["uri"]` model reference.

    Shared by the ISVC suite (tiny ONNX model-catalog test model) and the LLMISVC suite
    (TinyLlama-1.1B on HuggingFace); only the referenced model URI differs between them.
    """
    namespace = request.getfixturevalue(argname=request.param["namespace_fixture"])
    with create_uri_connection_secret(
        client=admin_client,
        name=request.param.get("name", URI_CONNECTION_SECRET_NAME),
        namespace=namespace.name,
        uri=request.param["uri"],
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def oci_connection_secret(request: FixtureRequest, admin_client: DynamicClient) -> Generator[Secret]:
    """`oci`-typed connection Secret (dockerconfigjson) for OCI modelcar pulls.

    The reused images (`SharedImages.MLSERVER_SKLEARN`, `SharedImages.OCI_TINYLLAMA`) are public
    quay.io modelcars, so an empty docker config is sufficient — the webhook still injects
    `imagePullSecrets` referencing this Secret regardless of its credential content.
    """
    namespace = request.getfixturevalue(argname=request.param["namespace_fixture"])
    with create_oci_connection_secret(
        client=admin_client,
        name=request.param.get("name", OCI_CONNECTION_SECRET_NAME),
        namespace=namespace.name,
    ) as secret:
        yield secret
