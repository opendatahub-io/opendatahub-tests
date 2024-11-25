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
from tests.model_serving.model_server.storage.pvc.utils import create_isvc
from tests.model_serving.model_server.private_endpoint.utils import (
    create_sidecar_pod,
    get_flan_deployment,
    b64_encoded_string,
)
from tests.model_serving.model_server.private_endpoint.infra import create_ns
from tests.model_serving.constants import KSERVE_SERVERLESS


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session")
def endpoint_namespace(admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    yield from create_ns(admin_client=admin_client, name="endpoint-namespace")


@pytest.fixture(scope="session")
def diff_namespace(admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    yield from create_ns(admin_client=admin_client, name="diff-namespace")


@pytest.fixture(scope="session")
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


@pytest.fixture(scope="session")
def endpoint_s3_secret(
    admin_client: DynamicClient,
    endpoint_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    models_s3_bucket_name: str,
    models_s3_bucket_region: str,
    models_s3_bucket_endpoint: str,
) -> Generator[Secret, None, None]:
    data = {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(aws_access_key_id),
        "AWS_DEFAULT_REGION": b64_encoded_string(models_s3_bucket_region),
        "AWS_S3_BUCKET": b64_encoded_string(models_s3_bucket_name),
        "AWS_S3_ENDPOINT": b64_encoded_string(models_s3_bucket_endpoint),
        "AWS_SECRET_ACCESS_KEY": b64_encoded_string(aws_secret_access_key),
    }
    with Secret(
        client=admin_client,
        namespace=endpoint_namespace.name,
        name="endpoint-s3-secret",
        data_dict=data,
        wait_for_resource=True,
    ) as secret:
        yield secret


@pytest.fixture(scope="session")
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
        deployment_mode=KSERVE_SERVERLESS,
        storage_key="endpoint-s3-secret",
        storage_path="flan-t5-small/flan-t5-small-caikit",
        model_format="caikit",
        runtime=endpoint_sr.name,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="session")
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
    data = {"endpoint-s3-secret": b64_encoded_string(json.dumps(secret))}
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
        istio=True,
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
        istio=False,
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
        istio=True,
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
        istio=False,
        pod_name="test",
    ) as pod:
        yield pod


@pytest.fixture()
def ready_predictor(admin_client: DynamicClient, endpoint_isvc: InferenceService) -> None:
    predictor_deployment = get_flan_deployment(
        namespace=endpoint_isvc.namespace,
        client=admin_client,
        name_prefix=endpoint_isvc.name,
    )
    predictor_deployment.wait_for_replicas()
