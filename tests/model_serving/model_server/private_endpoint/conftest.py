import json
import pytest
from typing import Generator, Any
from ocp_resources.inference_service import InferenceService
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from ocp_resources.serving_runtime import ServingRuntime
from kubernetes.dynamic import DynamicClient

from utilities.serving_runtime import ServingRuntimeFromTemplate
from tests.model_serving.model_server.utils import b64_encoded_string, create_isvc
from tests.model_serving.model_server.private_endpoint.utils import (
    create_sidecar_pod,
)
from utilities.infra import create_ns, s3_endpoint_secret, wait_for_kserve_predictor_deployment_replicas
from utilities.constants import KServeDeploymentType, ModelStoragePath, ModelFormat


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="class")
def endpoint_namespace(admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    with create_ns(admin_client=admin_client, name="endpoint-namespace") as ns:
        yield ns


@pytest.fixture(scope="class")
def diff_namespace(admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    with create_ns(admin_client=admin_client, name="diff-namespace") as ns:
        yield ns


@pytest.fixture(scope="class")
def endpoint_sr(
    admin_client: DynamicClient,
    endpoint_namespace: Namespace,
) -> Generator[ServingRuntime, None, None]:
    with ServingRuntimeFromTemplate(
        client=admin_client,
        name="flan-example-sr",
        namespace=endpoint_namespace.name,
        template_name="caikit-tgis-serving-template",
    ) as model_runtime:
        yield model_runtime


@pytest.fixture(scope="class")
def endpoint_s3_secret(
    admin_client: DynamicClient,
    endpoint_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    with s3_endpoint_secret(
        admin_client=admin_client,
        name="endpoint-s3-secret",
        namespace=endpoint_namespace.name,
        aws_access_key=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_s3_region=models_s3_bucket_region,
        aws_s3_bucket=models_s3_bucket_name,
        aws_s3_endpoint=models_s3_bucket_endpoint,
    ) as secret:
        yield secret


@pytest.fixture(scope="class")
def endpoint_isvc(
    admin_client: DynamicClient,
    endpoint_sr: ServingRuntime,
    endpoint_s3_secret: Secret,
    storage_config_secret: Secret,
    endpoint_namespace: Namespace,
) -> Generator[InferenceService, None, None]:
    with create_isvc(
        client=admin_client,
        name="test",
        namespace=endpoint_namespace.name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        storage_key="endpoint-s3-secret",
        storage_path=ModelStoragePath.FLAN_T5_SMALL,
        model_format=ModelFormat.CAIKIT,
        runtime=endpoint_sr.name,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def storage_config_secret(
    admin_client: DynamicClient,
    endpoint_namespace: Namespace,
    endpoint_s3_secret: Secret,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    secret = {
        "access_key_id": aws_access_key_id,
        "bucket": models_s3_bucket_name,
        "default_bucket": models_s3_bucket_name,
        "endpoint_url": models_s3_bucket_endpoint,
        "region": models_s3_bucket_region,
        "secret_access_key": aws_secret_access_key,
        "type": "s3",
    }
    data = {"endpoint-s3-secret": b64_encoded_string(string_to_encode=json.dumps(secret))}
    with Secret(
        client=admin_client,
        namespace=endpoint_namespace.name,
        data_dict=data,
        wait_for_resource=True,
        name="storage-config",
    ) as storage_config:
        yield storage_config


@pytest.fixture()
def endpoint_pod_with_istio_sidecar(
    admin_client: DynamicClient, endpoint_namespace: Namespace
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        admin_client=admin_client,
        namespace=endpoint_namespace.name,
        use_istio=True,
        pod_name="test-with-istio",
    ) as pod:
        yield pod


@pytest.fixture()
def endpoint_pod_without_istio_sidecar(
    admin_client: DynamicClient, endpoint_namespace: Namespace
) -> Generator[Pod, Any, Any]:
    with create_sidecar_pod(
        admin_client=admin_client,
        namespace=endpoint_namespace.name,
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


@pytest.fixture()
def ready_predictor(admin_client: DynamicClient, endpoint_isvc: InferenceService) -> None:
    wait_for_kserve_predictor_deployment_replicas(
        client=admin_client,
        isvc=endpoint_isvc,
    )
