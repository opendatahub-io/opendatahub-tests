from collections.abc import Generator
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.persistent_volume_claim import PersistentVolumeClaim
from pytest_testconfig import config as py_config

from utilities.constants import TRUSTYAI_SERVICE_NAME


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


@pytest.fixture(scope="session")
def trustyai_operator_configmap(
    admin_client: DynamicClient,
) -> ConfigMap:
    return ConfigMap(
        client=admin_client,
        namespace=py_config["applications_namespace"],
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-config",
        ensure_exists=True,
    )
