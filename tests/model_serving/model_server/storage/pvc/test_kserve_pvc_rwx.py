import shlex
from typing import List

import pytest

from tests.model_serving.model_server.storage.constants import (
    INFERENCE_SERVICE_PARAMS,
    KSERVE_CONTAINER_NAME,
    KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
    STANDARD_CSI_STR,
)

POD_LS_SPLIT_COMMAND: List[str] = shlex.split("ls /mnt/models")


pytestmark = pytest.mark.usefixtures("skip_if_no_standard_csi_storage_class")


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, model_pvc, pvc_serving_runtime, pvc_inference_service",
    [
        pytest.param(
            {"name": "pvc-rxw-access"},
            {"model-dir": "test-dir"},
            {"access-modes": "ReadWriteMany", "storage-class-name": STANDARD_CSI_STR},
            KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
            INFERENCE_SERVICE_PARAMS | {"deployment-mode": "Serverless", "min-replicas": 2},
        )
    ],
    indirect=True,
)
class TestKservePVCReadWriteManyAccess:
    def test_first_isvc_pvc_read_access(self, predictor_pods_scope_class):
        predictor_pods_scope_class[0].execute(
            container=KSERVE_CONTAINER_NAME,
            command=POD_LS_SPLIT_COMMAND,
        )

    def test_second_isvc_pvc_read_access(self, predictor_pods_scope_class):
        predictor_pods_scope_class[1].execute(
            container=KSERVE_CONTAINER_NAME,
            command=POD_LS_SPLIT_COMMAND,
        )
