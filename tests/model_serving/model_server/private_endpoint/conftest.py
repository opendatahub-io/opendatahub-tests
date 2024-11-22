import json
import pytest
from typing import Generator
from ocp_resources.inference_service import InferenceService
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from ocp_resources.service_mesh_member import ServiceMeshMember
from ocp_resources.serving_runtime import ServingRuntime
from kubernetes.dynamic import DynamicClient

from utilities.serving_runtime import ServingRuntimeFromTemplate
from tests.model_serving.model_server.storage.pvc.utils import create_isvc
from tests.model_serving.model_server.private_endpoint.utils import create_sidecar_pod, get_flan_pod, b64_encoded_string
from tests.model_serving.model_server.private_endpoint.infra import create_ns
from tests.model_serving.model_server.private_endpoint.constants import (
    AWS_REGION_EAST_2,
    AWS_ENDPOINT_EAST_2,
)


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="module")
def endpoint_namespace(admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    yield from create_ns(admin_client=admin_client, name="endpoint-namespace")


@pytest.fixture(scope="module")
def diff_namespace(admin_client: DynamicClient) -> Generator[Namespace, None, None]:
    yield from create_ns(admin_client=admin_client, name="diff-namespace")


@pytest.fixture()
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


@pytest.fixture()
def endpoint_s3_secret(
    admin_client: DynamicClient,
    endpoint_namespace: Namespace,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket_name_wisdom: str,
) -> Generator[Secret, None, None]:
    data = {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(aws_access_key_id),
        "AWS_DEFAULT_REGION": b64_encoded_string(AWS_REGION_EAST_2),
        "AWS_S3_BUCKET": b64_encoded_string(s3_bucket_name_wisdom),
        "AWS_S3_ENDPOINT": b64_encoded_string(AWS_ENDPOINT_EAST_2),
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


@pytest.fixture()
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
        deployment_mode="Serverless",
        storage_key="endpoint-s3-secret",
        storage_path="flan-t5-small/flan-t5-small-caikit",
        model_format="caikit",
        runtime=endpoint_sr.name,
    ) as isvc:
        yield isvc


@pytest.fixture()
def storage_config_secret(
    admin_client: DynamicClient,
    endpoint_namespace: Namespace,
    endpoint_s3_secret: Secret,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    s3_bucket_name_wisdom: str,
) -> Generator[Secret, None, None]:
    secret = {
        "access_key_id": aws_access_key_id,
        "bucket": s3_bucket_name_wisdom,
        "default_bucket": s3_bucket_name_wisdom,
        "endpoint_url": AWS_ENDPOINT_EAST_2,
        "region": AWS_REGION_EAST_2,
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
def service_mesh_member(
    admin_client: DynamicClient, diff_namespace: Namespace
) -> Generator[ServiceMeshMember, None, None]:
    with ServiceMeshMember(
        client=admin_client,
        namespace=diff_namespace.name,
        name="default",
        control_plane_ref={"name": "data-science-smcp", "namespace": "istio-system"},
        wait_for_resource=True,
    ) as smm:
        yield smm


@pytest.fixture()
def endpoint_pod_with_istio_sidecar(admin_client: DynamicClient, endpoint_namespace: Namespace) -> Pod:
    pod = create_sidecar_pod(
        admin_client=admin_client,
        namespace=endpoint_namespace.name,
        istio=True,
        pod_name="test-with-istio",
    )
    yield pod
    pod.clean_up()


@pytest.fixture()
def endpoint_pod_without_istio_sidecar(admin_client: DynamicClient, endpoint_namespace: Namespace) -> Pod:
    pod = create_sidecar_pod(
        admin_client=admin_client,
        namespace=endpoint_namespace.name,
        istio=False,
        pod_name="test",
    )
    yield pod
    pod.clean_up()


@pytest.fixture()
def diff_pod_with_istio_sidecar(admin_client: DynamicClient, diff_namespace: Namespace) -> Pod:
    pod = create_sidecar_pod(
        admin_client=admin_client,
        namespace=diff_namespace.name,
        istio=True,
        pod_name="test-with-istio",
    )
    yield pod
    pod.clean_up()


@pytest.fixture()
def diff_pod_without_istio_sidecar(admin_client: DynamicClient, diff_namespace: Namespace) -> Pod:
    pod = create_sidecar_pod(
        admin_client=admin_client,
        namespace=diff_namespace.name,
        istio=False,
        pod_name="test",
    )
    yield pod
    pod.clean_up()


@pytest.fixture()
def running_flan_pod(admin_client: DynamicClient, endpoint_isvc: InferenceService) -> None:
    predictor_pod = get_flan_pod(
        namespace=endpoint_isvc.namespace,
        client=admin_client,
        name_prefix=endpoint_isvc.name,
    )
    predictor_pod.wait_for_status(status="Running")
    predictor_pod.wait_for_condition(condition="Ready", status="True")
