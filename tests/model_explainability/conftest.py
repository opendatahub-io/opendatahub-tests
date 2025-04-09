from typing import Generator, Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim


@pytest.fixture(scope="class")
def pvc_minio_namespace(
    admin_client: DynamicClient, minio_namespace: Namespace
) -> Generator[PersistentVolumeClaim, Any, Any]:
    with PersistentVolumeClaim(
        client=admin_client,
        name="minio-pvc",
        namespace=minio_namespace.name,
        accessmodes=PersistentVolumeClaim.AccessMode.RWO,
        volume_mode=PersistentVolumeClaim.VolumeMode.FILE,
        size="10Gi",
    ) as pvc:
        yield pvc
