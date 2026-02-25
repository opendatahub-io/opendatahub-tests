from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger

from tests.model_serving.model_server.kserve.private_endpoint.utils import create_sidecar_pod
from utilities.constants import KServeDeploymentType, ModelFormat, ModelStoragePath
from utilities.inference_utils import create_isvc
from utilities.infra import create_ns

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def diff_namespace(admin_client: DynamicClient, unprivileged_client: DynamicClient) -> Generator[Namespace, Any, Any]:
    with create_ns(admin_client=admin_client, unprivileged_client=unprivileged_client, name="diff-namespace") as ns:
        yield ns


@pytest.fixture(scope="class")
def endpoint_isvc(
    unprivileged_client: DynamicClient,
    serving_runtime_from_template: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name="endpoint-isvc",
        namespace=serving_runtime_from_template.namespace,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.FLAN_T5_SMALL_CAIKIT,
        model_format=ModelFormat.CAIKIT,
        runtime=serving_runtime_from_template.name,
        wait_for_predictor_pods=True,
    ) as isvc:
        yield isvc


@pytest.fixture()
def endpoint_pod_with_istio_sidecar(
    unprivileged_client: DynamicClient, unprivileged_model_namespace: Namespace
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        use_istio=True,
        pod_name="test-with-istio",
    ) as pod:
        yield pod


@pytest.fixture()
def endpoint_pod_without_istio_sidecar(
    unprivileged_client: DynamicClient, unprivileged_model_namespace: Namespace
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        client=unprivileged_client,
        namespace=unprivileged_model_namespace.name,
        use_istio=False,
        pod_name="test",
    ) as pod:
        yield pod


@pytest.fixture()
def diff_pod_with_istio_sidecar(
    unprivileged_client: DynamicClient,
    diff_namespace: Namespace,
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        client=unprivileged_client,
        namespace=diff_namespace.name,
        use_istio=True,
        pod_name="test-with-istio",
    ) as pod:
        yield pod


@pytest.fixture()
def diff_pod_without_istio_sidecar(
    unprivileged_client: DynamicClient,
    diff_namespace: Namespace,
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        client=unprivileged_client,
        namespace=diff_namespace.name,
        use_istio=False,
        pod_name="test",
    ) as pod:
        yield pod
