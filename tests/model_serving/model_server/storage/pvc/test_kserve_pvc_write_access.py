import shlex
from typing import List

from ocp_resources.pod import ExecOnPodError
import pytest

from tests.model_serving.model_server.storage.constants import (
    INFERENCE_SERVICE_PARAMS,
    KSERVE_CONTAINER_NAME,
    KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
    NFS_STR,
)
from tests.model_serving.model_server.utils import get_pods_by_isvc_label

pytestmark = pytest.mark.usefixtures("skip_if_no_nfs_storage_class", "valid_aws_config")


POD_TOUCH_SPLIT_COMMAND: List[str] = shlex.split("touch /mnt/models/test")


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, model_pvc, pvc_serving_runtime, pvc_inference_service",
    [
        pytest.param(
            {"name": "pvc-write-access"},
            {"model-dir": "test-dir"},
            {"access-modes": "ReadWriteMany", "storage-class-name": NFS_STR},
            KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
            INFERENCE_SERVICE_PARAMS,
        )
    ],
    indirect=True,
)
class TestKservePVCWriteAccess:
    def test_pod_containers_not_restarted(self, first_predictor_pod):
        restarted_containers = [
            container.name
            for container in first_predictor_pod.instance.status.containerStatuses
            if container.restartCount > 0
        ]
        assert not restarted_containers, f"Containers {restarted_containers} restarted"

    def test_isvc_read_only_annotation_not_set_by_default(self, pvc_inference_service):
        assert not pvc_inference_service.instance.metadata.annotations.get("storage.kserve.io/readonly"), (
            "Read only annotation is set"
        )

    def test_isvc_read_only_annotation_default_value(self, first_predictor_pod):
        with pytest.raises(ExecOnPodError):
            first_predictor_pod.execute(
                container=KSERVE_CONTAINER_NAME,
                command=POD_TOUCH_SPLIT_COMMAND,
            )

    @pytest.mark.parametrize(
        "patched_read_only_isvc",
        [
            pytest.param(
                {"readonly": "false"},
            ),
        ],
        indirect=True,
    )
    def test_isvc_read_only_annotation_false(self, admin_client, patched_read_only_isvc):
        new_pod = get_pods_by_isvc_label(
            client=admin_client,
            isvc=patched_read_only_isvc,
        )[0]
        new_pod.execute(
            container=KSERVE_CONTAINER_NAME,
            command=POD_TOUCH_SPLIT_COMMAND,
        )

    @pytest.mark.parametrize(
        "patched_read_only_isvc",
        [
            pytest.param(
                {"readonly": "true"},
            ),
        ],
        indirect=True,
    )
    def test_isvc_read_only_annotation_true(self, admin_client, patched_read_only_isvc):
        new_pod = get_pods_by_isvc_label(
            client=admin_client,
            isvc=patched_read_only_isvc,
        )[0]
        with pytest.raises(ExecOnPodError):
            new_pod.execute(
                container=KSERVE_CONTAINER_NAME,
                command=POD_TOUCH_SPLIT_COMMAND,
            )
