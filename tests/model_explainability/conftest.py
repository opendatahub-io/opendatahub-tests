from typing import Generator, Any

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service import Service

from tests.model_explainability.constants import MINIO, MINIO_PORT
from utilities.constants import ApiGroups, Labels


@pytest.fixture(scope="class")
def minio_service(admin_client: DynamicClient, model_namespace: Namespace) -> Generator[Service, Any, Any]:
    with Service(
        client=admin_client,
        name=MINIO,
        namespace=model_namespace.name,
        ports=[
            {
                "name": "minio-client-port",
                "port": MINIO_PORT,
                "protocol": "TCP",
                "targetPort": MINIO_PORT,
            }
        ],
        selector={
            "app": MINIO,
        },
    ) as minio_service:
        yield minio_service


@pytest.fixture(scope="class")
def minio_data_connection(
    request: FixtureRequest, admin_client: DynamicClient, model_namespace: Namespace, minio_service: Service
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name="aws-connection-minio-data-connection",
        namespace=model_namespace.name,
        data_dict=request.param["data-dict"],
        label={
            Labels.OpenDataHub.DASHBOARD: "true",
            Labels.OpenDataHubIo.MANAGED: "true",
        },
        annotations={
            f"{ApiGroups.OPENDATAHUB_IO}/connection-type": "s3",
            "openshift.io/display-name": "Minio Data Connection",
        },
    ) as minio_secret:
        yield minio_secret
