from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.pod import Pod
from ocp_resources.resource import get_client
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.trustyai.utils.minio import MinioPod, MinioService, MinioSecret


MINIO: str = "minio"
OPENDATAHUB_IO: str = "opendatahub.io"


@pytest.fixture(scope="session")
def admin_client() -> DynamicClient:
    return get_client()


@pytest.fixture(scope="class")
def model_namespace(request: Any, admin_client: DynamicClient) -> Namespace:
    with Namespace(
        client=admin_client,
        name=request.param["name"],
    ) as ns:
        ns.wait_for_status(status=Namespace.Status.ACTIVE, timeout=120)
        yield ns


@pytest.fixture(scope="class")
def minio_pod(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[MinioPod]:
    with Pod(
        name=MINIO,
        namespace=model_namespace.name,
        containers=[
            {
                "args": [
                    "server",
                    "/data1",
                ],
                "env": [
                    {
                        "name": "MINIO_ACCESS_KEY",
                        "value": "THEACCESSKEY",
                    },
                    {
                        "name": "MINIO_SECRET_KEY",
                        "value": "THESECRETKEY",
                    },
                ],
                "image": "quay.io/trustyai/modelmesh-minio-examples@"
                "sha256:e8360ec33837b347c76d2ea45cd4fea0b40209f77520181b15e534b101b1f323",
                "name": MINIO,
            }
        ],
        label={"app": "minio", "maistra.io/expose-route": "true"},
        annotations={"sidecar.istio.io/inject": "true"},
    ) as minio_pod:
        yield minio_pod


@pytest.fixture(scope="class")
def minio_service(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[MinioService]:
    with Service(
        name=MINIO,
        namespace=model_namespace.name,
        ports=[
            {
                "name": "minio-client-port",
                "port": 9000,
                "protocol": "TCP",
                "targetPort": 9000,
            }
        ],
        selector={
            "app": "minio",
        },
    ) as minio_service:
        yield minio_service


@pytest.fixture(scope="class")
def minio_secret(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[MinioSecret]:
    with Secret(
        name="aws-connection-minio-data-connection",
        namespace=model_namespace.name,
        data_dict={
            "AWS_ACCESS_KEY_ID": "VEhFQUNDRVNTS0VZ",
            "AWS_DEFAULT_REGION": "dXMtc291dGg=",
            "AWS_S3_BUCKET": "bW9kZWxtZXNoLWV4YW1wbGUtbW9kZWxz",
            "AWS_S3_ENDPOINT": "aHR0cDovL21pbmlvOjkwMDA=",
            "AWS_SECRET_ACCESS_KEY": "VEhFU0VDUkVUS0VZ",
        },
        label={
            f"{OPENDATAHUB_IO}/dashboard": "true",
            f"{OPENDATAHUB_IO}/managed": "true",
        },
        annotations={
            f"{OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    ) as minio_secret:
        yield minio_secret


@pytest.fixture(scope="class")
def minio_data_connection(
    minio_service: Service, minio_pod: Pod, minio_secret: Secret
) -> Generator[Secret]:
    yield minio_secret
