from collections.abc import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.kserve.storage.constants import ISVC_URI_MODEL_URI
from tests.model_serving.model_server.utils import create_uri_connection_secret
from utilities.constants import KServeDeploymentType, ModelInferenceRuntime, RuntimeTemplates
from utilities.serving_runtime import ServingRuntimeFromTemplate


@pytest.fixture(scope="class")
def kserve_storage_mlserver_runtime(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    mlserver_runtime_image: str | None,
) -> Generator[ServingRuntime]:
    """CPU-only MLServer ServingRuntime for the standalone URI ConnectionsAPI test.

    A local equivalent of `tests.model_serving.model_runtime.mlserver.conftest`'s
    `mlserver_serving_runtime` fixture, defined here under a distinct name since pytest does not
    auto-discover fixtures across that sibling subtree for tests under `model_server/`.
    """
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name=ModelInferenceRuntime.MLSERVER_RUNTIME,
        namespace=unprivileged_model_namespace.name,
        template_name=RuntimeTemplates.MLSERVER,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        runtime_image=mlserver_runtime_image,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def uri_connection_secret(
    admin_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
) -> Generator[Secret]:
    """`uri`-typed connection Secret pointing at the tiny ONNX model-catalog test model."""
    with create_uri_connection_secret(
        client=admin_client,
        name="kserve-storage-uri-connection-secret",
        namespace=unprivileged_model_namespace.name,
        uri=ISVC_URI_MODEL_URI,
    ) as secret:
        yield secret
