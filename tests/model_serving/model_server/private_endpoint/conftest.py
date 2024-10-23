import json
import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.secret import Secret
from ocp_resources.namespace import Namespace
from ocp_resources.serving_runtime import ServingRuntime
from simple_logger.logger import get_logger
from ocp_resources.service_mesh_member import ServiceMeshMember
from kubernetes.dynamic import DynamicClient

from tests.model_serving.model_server.private_endpoint.utils import create_sidecar_pod, get_flan_pod, b64_encoded_string
from tests.model_serving.model_server.private_endpoint.infra import create_ns
from tests.model_serving.model_server.private_endpoint.constants import (
    AWS_REGION,
    AWS_BUCKET,
    AWS_ENDPOINT,
    SR_ANNOTATIONS,
    SR_CONTAINERS_KSERVE_CAIKIT,
    SR_SUPPORTED_FORMATS_CAIKIT,
    SR_VOLUMES,
)


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="module")
def endpoint_namespace(admin_client):
    yield from create_ns(admin_client=admin_client, name="endpoint-namespace")


@pytest.fixture(scope="module")
def diff_namespace(admin_client):
    yield from create_ns(admin_client=admin_client, name="diff-namespace")


@pytest.fixture()
def endpoint_sr(
    admin_client: DynamicClient,
    endpoint_namespace: Namespace,
) -> ServingRuntime:
    with ServingRuntime(
        client=admin_client,
        name="flan-example-sr",
        namespace=endpoint_namespace.name,
        containers=SR_CONTAINERS_KSERVE_CAIKIT,
        multi_model=False,
        supported_model_formats=SR_SUPPORTED_FORMATS_CAIKIT,
        volumes=SR_VOLUMES,
        spec_annotations=SR_ANNOTATIONS,
    ) as model_runtime:
        yield model_runtime


@pytest.fixture()
def endpoint_s3_secret(admin_client, endpoint_namespace, aws_access_key, aws_secret_access_key):
    data = {
        "AWS_ACCESS_KEY_ID": b64_encoded_string(aws_access_key),
        "AWS_DEFAULT_REGION": b64_encoded_string(AWS_REGION),
        "AWS_S3_BUCKET": b64_encoded_string(AWS_BUCKET),
        "AWS_S3_ENDPOINT": b64_encoded_string(AWS_REGION),
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
def endpoint_isvc(admin_client, endpoint_sr, endpoint_s3_secret, storage_config_secret, endpoint_namespace):
    predictor = {
        "model": {
            "modelFormat": {
                "name": "caikit",
            },
            "name": "kserve-container",
            "resources": {
                "limits": {"cpu": "2", "memory": "8Gi"},
                "requests": {"cpu": "1", "memory": "4Gi"},
            },
            "runtime": endpoint_sr.name,
            "storage": {
                "key": "endpoint-s3-secret",
                "path": "flan-t5-small/flan-t5-small-caikit",
            },
        },
    }

    with InferenceService(
        client=admin_client, namespace=endpoint_namespace.name, predictor=predictor, name="test"
    ) as isvc:
        isvc.wait_for_condition(condition="Ready", status="True")
        yield isvc


@pytest.fixture()
def storage_config_secret(admin_client, endpoint_namespace, endpoint_s3_secret, aws_access_key, aws_secret_access_key):
    secret = {
        "access_key_id": aws_access_key,
        "bucket": AWS_BUCKET,
        "default_bucket": AWS_BUCKET,
        "endpoint_url": AWS_ENDPOINT,
        "region": AWS_REGION,
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
def service_mesh_member(admin_client, diff_namespace):
    with ServiceMeshMember(
        client=admin_client,
        namespace=diff_namespace.name,
        name="default",
        control_plane_ref={"name": "data-science-smcp", "namespace": "istio-system"},
        wait_for_resource=True,
    ) as smm:
        yield smm


@pytest.fixture()
def endpoint_pod_with_istio_sidecar(admin_client, endpoint_namespace):
    pod = create_sidecar_pod(
        admin_client=admin_client,
        namespace=endpoint_namespace.name,
        istio=True,
        pod_name="test-with-istio",
    )
    yield pod
    pod.clean_up()


@pytest.fixture()
def endpoint_pod_without_istio_sidecar(admin_client, endpoint_namespace):
    pod = create_sidecar_pod(
        admin_client=admin_client,
        namespace=endpoint_namespace.name,
        istio=False,
        pod_name="test",
    )
    yield pod
    pod.clean_up()


@pytest.fixture()
def diff_pod_with_istio_sidecar(admin_client, diff_namespace):
    pod = create_sidecar_pod(
        admin_client=admin_client,
        namespace=diff_namespace.name,
        istio=True,
        pod_name="test-with-istio",
    )
    yield pod
    pod.clean_up()


@pytest.fixture()
def diff_pod_without_istio_sidecar(admin_client, diff_namespace):
    pod = create_sidecar_pod(
        admin_client=admin_client,
        namespace=diff_namespace.name,
        istio=False,
        pod_name="test",
    )
    yield pod
    pod.clean_up()


@pytest.fixture()
def running_flan_pod(admin_client, endpoint_isvc):
    predictor_pod = get_flan_pod(
        namespace=endpoint_isvc.namespace,
        client=admin_client,
        name_prefix=endpoint_isvc.name,
    )
    predictor_pod.wait_for_status(status="Running")
    predictor_pod.wait_for_condition(condition="Ready", status="True")
