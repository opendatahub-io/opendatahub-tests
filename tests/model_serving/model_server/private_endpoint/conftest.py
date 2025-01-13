from typing import Any, Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger

from tests.model_serving.model_server.private_endpoint.utils import create_sidecar_pod
from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import KServeDeploymentType, ModelFormat, ModelStoragePath
from utilities.infra import create_ns, create_storage_config_secret

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def diff_namespace(admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    with create_ns(admin_client=admin_client, name="diff-namespace") as ns:
        yield ns


@pytest.fixture(scope="class")
def endpoint_isvc(
    admin_client: DynamicClient,
    serving_runtime_from_template: ServingRuntime,
    models_endpoint_s3_secret: Secret,
    storage_config_secret: Secret,
) -> Generator[InferenceService, None, None]:
    with create_isvc(
        client=admin_client,
        name="endpoint-isvc",
        namespace=serving_runtime_from_template.namespace,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.FLAN_T5_SMALL,
        model_format=ModelFormat.CAIKIT,
        runtime=serving_runtime_from_template.name,
        wait_for_predictor_pods=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def storage_config_secret(
    admin_client: DynamicClient,
    models_endpoint_s3_secret: Secret,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    with create_storage_config_secret(
        admin_client=admin_client,
        endpoint_secret_name=models_endpoint_s3_secret.name,
        namespace=models_endpoint_s3_secret.namespace,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as storage_config:
        yield storage_config


@pytest.fixture()
def endpoint_pod_with_istio_sidecar(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        admin_client=admin_client,
        namespace=model_namespace.name,
        use_istio=True,
        pod_name="test-with-istio",
    ) as pod:
        yield pod


@pytest.fixture()
def endpoint_pod_without_istio_sidecar(
    admin_client: DynamicClient, model_namespace: Namespace
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        admin_client=admin_client,
        namespace=model_namespace.name,
        use_istio=False,
        pod_name="test",
    ) as pod:
        yield pod


@pytest.fixture()
def diff_pod_with_istio_sidecar(
    admin_client: DynamicClient,
    diff_namespace: Namespace,
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        admin_client=admin_client,
        namespace=diff_namespace.name,
        use_istio=True,
        pod_name="test-with-istio",
    ) as pod:
        yield pod


@pytest.fixture()
def diff_pod_without_istio_sidecar(
    admin_client: DynamicClient,
    diff_namespace: Namespace,
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        admin_client=admin_client,
        namespace=diff_namespace.name,
        use_istio=False,
        pod_name="test",
    ) as pod:
        yield pod
