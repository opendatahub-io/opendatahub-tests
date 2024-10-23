from typing import Optional, Tuple

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.service_mesh_member import ServiceMeshMember
from pytest_testconfig import py_config


@pytest.fixture(scope="session")
def aws_access_key() -> Optional[str]:
    access_key = py_config.get("aws_access_key_id")
    if not access_key:
        raise ValueError("AWS access key is not set")

    return access_key


@pytest.fixture(scope="session")
def aws_secret_access_key() -> Optional[str]:
    secret_access_key = py_config.get("aws_secret_access_key")
    if not secret_access_key:
        raise ValueError("AWS secret key is not set")

    return secret_access_key


@pytest.fixture(scope="session")
def valid_aws_config(aws_access_key: str, aws_secret_access_key: str) -> Tuple[str, str]:
    return aws_access_key, aws_secret_access_key


@pytest.fixture(scope="class")
def service_mesh_member(admin_client: DynamicClient, model_namespace: Namespace) -> ServiceMeshMember:
    with ServiceMeshMember(
        client=admin_client,
        name="default",
        namespace=model_namespace.name,
        control_plane_ref={"name": "data-science-smcp", "namespace": "istio-system"},
    ) as smm:
        yield smm
