from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from ocp_resources.storage_class import StorageClass
from ocp_utilities.operators import install_operator, uninstall_operator

from utilities.constants import StorageType
from utilities.infra import create_ns


@pytest.fixture(scope="class")
def s3_models_storage_uri(request, models_s3_bucket_name) -> str:
    return f"s3://{models_s3_bucket_name}/{request.param['model-dir']}/"


@pytest.fixture(scope="class")
def model_pvc(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
) -> Generator[PersistentVolumeClaim, Any, Any]:
    access_mode = "ReadWriteOnce"
    size = "15Gi"
    pvc_kwargs = {
        "name": "model-pvc",
        "namespace": model_namespace.name,
        "client": admin_client,
    }
    if hasattr(request, "param"):
        access_mode = request.param.get("access-modes")
        size = request.param.get("size")

        if storage_class_name := request.param.get("storage-class-name"):
            pvc_kwargs["storage_class"] = storage_class_name

    pvc_kwargs["accessmodes"] = access_mode
    pvc_kwargs["size"] = size

    with PersistentVolumeClaim(**pvc_kwargs) as pvc:
        yield pvc


@pytest.fixture(scope="module")
def skip_if_no_nfs_storage_class(admin_client: DynamicClient) -> None:
    if not StorageClass(client=admin_client, name=StorageType.NFS).exists:
        pytest.skip(f"StorageClass {StorageType.NFS} is missing from the cluster")


@pytest.fixture(scope="session")
def nfs_provisioner_namespace(admin_client: DynamicClient) -> Namespace:
    with create_ns(
        client=admin_client,
        name="nfs-provisioner",
    ) as ns:
        yield ns


@pytest.fixture(scope="session")
def nfs_storage_operator(admin_client: DynamicClient, nfs_provisioner_namespace: Namespace) -> None:
    name = "nfs-provisioner-operator"
    install_operator(
        admin_client=admin_client,
        name=name,
        operator_namespace=nfs_provisioner_namespace.name,
        channel="alpha",
        target_namespaces=[],
        source="community-operators",
    )
    yield
    uninstall_operator(admin_client=admin_client, name=name, operator_namespace=nfs_provisioner_namespace.name)


@pytest.fixture(scope="session")
def nfs_provisioner(admin_client: DynamicClient, nfs_storage_operator: None) -> None:
    pass
