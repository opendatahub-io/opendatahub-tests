import shlex

import pytest
from ocp_resources.pod import ExecOnPodError

from tests.model_serving.model_server.kserve.storage.constants import (
    INFERENCE_SERVICE_PARAMS,
    KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
)
from utilities.constants import Containers, StorageClassName
from utilities.infra import get_pods_by_isvc_label

pytestmark = [pytest.mark.tier1, pytest.mark.usefixtures("skip_if_no_nfs_storage_class", "valid_aws_config")]


POD_TOUCH_SPLIT_COMMAND: list[str] = shlex.split("touch /mnt/models/test")


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ci_bucket_downloaded_model_data, model_pvc, serving_runtime_from_template,"
    "pvc_inference_service",
    [
        pytest.param(
            {"name": "pvc-write-access"},
            {"model-dir": "test-dir"},
            {"access-modes": "ReadWriteMany", "storage-class-name": StorageClassName.NFS, "pvc-size": "4Gi"},
            KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
            INFERENCE_SERVICE_PARAMS,
        )
    ],
    indirect=True,
)
class TestKservePVCWriteAccess:
    """Validate PVC write access control via the KServe read-only storage annotation.

    Steps:
        1. Deploy an ISVC with a ReadWriteMany NFS PVC and no explicit read-only annotation.
        2. Verify no pod containers have restarted.
        3. Verify the read-only annotation is not set by default.
        4. Verify write access is denied by default (touch command fails).
        5. Patch the ISVC with readonly=false and verify write access is allowed.
        6. Patch the ISVC with readonly=true and verify write access is denied again.
    """

    def test_pod_containers_not_restarted(self, first_predictor_pod):
        """Test that the containers are not restarted"""
        restarted_containers = [
            container.name
            for container in first_predictor_pod.instance.status.containerStatuses
            if container.restartCount > 0
        ]
        assert not restarted_containers, f"Containers {restarted_containers} restarted"

    def test_isvc_read_only_annotation_not_set_by_default(self, pvc_inference_service):
        """Test that the read only annotation is not set by default"""
        assert not pvc_inference_service.instance.metadata.annotations.get("storage.kserve.io/readonly"), (
            "Read only annotation is set"
        )

    def test_isvc_read_only_annotation_default_value(self, first_predictor_pod):
        """Test that write access is denied by default"""
        with pytest.raises(ExecOnPodError):
            first_predictor_pod.execute(
                container=Containers.KSERVE_CONTAINER_NAME,
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
    def test_isvc_read_only_annotation_false(self, unprivileged_client, patched_read_only_isvc):
        """Test that write access is allowed when the read only annotation is set to false"""
        new_pod = get_pods_by_isvc_label(
            client=unprivileged_client,
            isvc=patched_read_only_isvc,
        )[0]
        new_pod.execute(
            container=Containers.KSERVE_CONTAINER_NAME,
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
    def test_isvc_read_only_annotation_true(self, unprivileged_client, patched_read_only_isvc):
        """Verify that write access is denied when the read-only annotation is set to true."""
        new_pod = get_pods_by_isvc_label(
            client=unprivileged_client,
            isvc=patched_read_only_isvc,
        )[0]
        with pytest.raises(ExecOnPodError):
            new_pod.execute(
                container=Containers.KSERVE_CONTAINER_NAME,
                command=POD_TOUCH_SPLIT_COMMAND,
            )
