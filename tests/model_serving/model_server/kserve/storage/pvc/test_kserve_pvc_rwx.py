import shlex

import pytest

from tests.model_serving.model_server.kserve.storage.constants import (
    INFERENCE_SERVICE_PARAMS,
    KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
)
from utilities.constants import Containers, KServeDeploymentType, StorageClassName

POD_LS_SPLIT_COMMAND: list[str] = shlex.split("ls /mnt/models")


pytestmark = [pytest.mark.tier1, pytest.mark.usefixtures("skip_if_no_nfs_storage_class")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ci_bucket_downloaded_model_data, model_pvc, serving_runtime_from_template, "
    "pvc_inference_service",
    [
        pytest.param(
            {"name": "pvc-rwx-access"},
            {"model-dir": "test-dir"},
            {"access-modes": "ReadWriteMany", "storage-class-name": StorageClassName.NFS, "pvc-size": "4Gi"},
            KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
            INFERENCE_SERVICE_PARAMS | {"deployment-mode": KServeDeploymentType.SERVERLESS, "min-replicas": 2},
        )
    ],
    indirect=True,
)
class TestKservePVCReadWriteManyAccess:
    """Validate ReadWriteMany PVC access across multiple KServe predictor pods.

    Steps:
        1. Deploy a serverless ISVC with 2 replicas backed by an NFS ReadWriteMany PVC.
        2. Verify the first predictor pod can read from the mounted PVC path.
        3. Verify the second predictor pod can also read from the same PVC path.
    """

    def test_first_isvc_pvc_read_access(self, predictor_pods_scope_class):
        """Test that the first predictor pod has read access to the PVC"""
        predictor_pods_scope_class[0].execute(
            container=Containers.KSERVE_CONTAINER_NAME,
            command=POD_LS_SPLIT_COMMAND,
        )

    def test_second_isvc_pvc_read_access(self, predictor_pods_scope_class):
        """Test that the second predictor pod has read access to the PVC"""
        predictor_pods_scope_class[1].execute(
            container=Containers.KSERVE_CONTAINER_NAME,
            command=POD_LS_SPLIT_COMMAND,
        )
