import logging

import pytest

from tests.model_serving.model_server.kserve.storage.constants import (
    INFERENCE_SERVICE_PARAMS,
    KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
)
from tests.model_serving.model_server.kserve.storage.pvc.utils import (
    get_mount_mode,
    get_running_predictor_pod,
    get_volume_mount_readonly,
    wait_for_rollout_complete,
)
from utilities.constants import KServeDeploymentType, StorageClassName

LOGGER = logging.getLogger(__name__)

pytestmark = [pytest.mark.tier1, pytest.mark.usefixtures("skip_if_no_nfs_storage_class", "valid_aws_config")]


@pytest.mark.parametrize(
    "unprivileged_model_namespace, ci_bucket_downloaded_model_data, model_pvc, serving_runtime_from_template,"
    "pvc_inference_service",
    [
        pytest.param(
            {"name": "pvc-write-access"},
            {"model-dir": "test-dir"},
            {"access-modes": "ReadWriteMany", "storage-class-name": StorageClassName.NFS, "pvc-size": "4Gi"},
            KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
            INFERENCE_SERVICE_PARAMS | {"deployment-mode": KServeDeploymentType.RAW_DEPLOYMENT},
        )
    ],
    indirect=True,
)
class TestKservePVCWriteAccess:
    """Validate PVC write access control via the storage.kserve.io/readonly annotation.

    Deploys a raw deployment ISVC with a ReadWriteMany NFS PVC and verifies:
        1. Default state: no annotation, pod spec readOnly=true, /mnt/models mounted ro.
        2. None→false→true: patch to false (rw), then toggle to true (ro).
        3. None→true→false: patch to true (ro), then toggle to false (rw).

    Each step checks the ISVC annotation, pod spec volumeMount.readOnly, and
    /proc/mounts inside the container.
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

    def test_isvc_read_only_pod_spec_default(self, first_predictor_pod):
        """Test that the pod spec has readOnly=true by default (webhook contract)"""
        LOGGER.info(
            f"pod={first_predictor_pod.name} "
            f"created={first_predictor_pod.instance.metadata.creationTimestamp} "
            f"uid={first_predictor_pod.instance.metadata.uid}"
        )
        assert get_volume_mount_readonly(pod=first_predictor_pod), (
            "Expected volumeMount.readOnly=true on /mnt/models by default"
        )

    def test_isvc_read_only_mount_default(self, first_predictor_pod):
        """Test that /mnt/models is mounted read-only by default (runtime effect)"""
        LOGGER.info(
            f"pod={first_predictor_pod.name} "
            f"created={first_predictor_pod.instance.metadata.creationTimestamp} "
            f"uid={first_predictor_pod.instance.metadata.uid}"
        )
        assert get_mount_mode(pod=first_predictor_pod) == "ro", (
            "Expected /mnt/models to be mounted read-only by default"
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
    def test_isvc_read_only_false(self, unprivileged_client, patched_read_only_isvc):
        """Test None→false→true transition.

        1. Fixture patches annotation to readonly=false and waits for rollout.
        2. Verify annotation is "false".
        3. Verify pod spec has volumeMount.readOnly=false.
        4. Verify /mnt/models is mounted rw via /proc/mounts.
        5. Patch annotation to readonly=true and wait for rollout.
        6. Verify annotation is "true".
        7. Verify pod spec has volumeMount.readOnly=true.
        8. Verify /mnt/models is mounted ro via /proc/mounts.
        """
        annotation = patched_read_only_isvc.instance.metadata.annotations.get("storage.kserve.io/readonly")
        assert annotation == "false", f"Expected annotation readonly=false after patch, got {annotation}"
        pod = get_running_predictor_pod(client=unprivileged_client, isvc=patched_read_only_isvc)
        LOGGER.info(f"pod={pod.name} created={pod.instance.metadata.creationTimestamp} uid={pod.instance.metadata.uid}")
        assert not get_volume_mount_readonly(pod=pod), "Expected volumeMount.readOnly=false with readonly=false"
        assert get_mount_mode(pod=pod) == "rw", "Expected /mnt/models mounted rw with readonly=false"

        patched_read_only_isvc.update(
            resource_dict={
                "metadata": {
                    "name": patched_read_only_isvc.name,
                    "annotations": {"storage.kserve.io/readonly": "true"},
                }
            }
        )
        wait_for_rollout_complete(client=unprivileged_client, isvc=patched_read_only_isvc)
        annotation = patched_read_only_isvc.instance.metadata.annotations.get("storage.kserve.io/readonly")
        assert annotation == "true", f"Expected annotation readonly=true after toggle, got {annotation}"
        pod = get_running_predictor_pod(client=unprivileged_client, isvc=patched_read_only_isvc)
        LOGGER.info(f"pod={pod.name} created={pod.instance.metadata.creationTimestamp} uid={pod.instance.metadata.uid}")
        assert get_volume_mount_readonly(pod=pod), "Expected volumeMount.readOnly=true after toggle false→true"
        assert get_mount_mode(pod=pod) == "ro", "Expected /mnt/models mounted ro after toggle false→true"

    @pytest.mark.parametrize(
        "patched_read_only_isvc",
        [
            pytest.param(
                {"readonly": "true"},
            ),
        ],
        indirect=True,
    )
    def test_isvc_read_only_true(self, unprivileged_client, patched_read_only_isvc):
        """Test None→true→false transition.

        1. Fixture patches annotation to readonly=true and waits for rollout.
        2. Verify annotation is "true".
        3. Verify pod spec has volumeMount.readOnly=true.
        4. Verify /mnt/models is mounted ro via /proc/mounts.
        5. Patch annotation to readonly=false and wait for rollout.
        6. Verify annotation is "false".
        7. Verify pod spec has volumeMount.readOnly=false.
        8. Verify /mnt/models is mounted rw via /proc/mounts.
        """
        annotation = patched_read_only_isvc.instance.metadata.annotations.get("storage.kserve.io/readonly")
        assert annotation == "true", f"Expected annotation readonly=true after patch, got {annotation}"
        pod = get_running_predictor_pod(client=unprivileged_client, isvc=patched_read_only_isvc)
        LOGGER.info(f"pod={pod.name} created={pod.instance.metadata.creationTimestamp} uid={pod.instance.metadata.uid}")
        assert get_volume_mount_readonly(pod=pod), "Expected volumeMount.readOnly=true with readonly=true"
        assert get_mount_mode(pod=pod) == "ro", "Expected /mnt/models mounted ro with readonly=true"

        patched_read_only_isvc.update(
            resource_dict={
                "metadata": {
                    "name": patched_read_only_isvc.name,
                    "annotations": {"storage.kserve.io/readonly": "false"},
                }
            }
        )
        wait_for_rollout_complete(client=unprivileged_client, isvc=patched_read_only_isvc)
        annotation = patched_read_only_isvc.instance.metadata.annotations.get("storage.kserve.io/readonly")
        assert annotation == "false", f"Expected annotation readonly=false after toggle, got {annotation}"
        pod = get_running_predictor_pod(client=unprivileged_client, isvc=patched_read_only_isvc)
        LOGGER.info(f"pod={pod.name} created={pod.instance.metadata.creationTimestamp} uid={pod.instance.metadata.uid}")
        assert not get_volume_mount_readonly(pod=pod), "Expected volumeMount.readOnly=false after toggle true→false"
        assert get_mount_mode(pod=pod) == "rw", "Expected /mnt/models mounted rw after toggle true→false"
