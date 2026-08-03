import logging
import shlex

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_server.kserve.storage.constants import (
    INFERENCE_SERVICE_PARAMS,
    KSERVE_OVMS_SERVING_RUNTIME_PARAMS,
)
from utilities.constants import Containers, KServeDeploymentType, StorageClassName
from utilities.infra import get_pods_by_isvc_label

LOGGER = logging.getLogger(__name__)

pytestmark = [pytest.mark.tier1, pytest.mark.usefixtures("skip_if_no_nfs_storage_class", "valid_aws_config")]

MOUNT_CHECK_COMMAND: list[str] = shlex.split("cat /proc/mounts")
MODELS_MOUNT_PATH = "/mnt/models"


def get_volume_mount_readonly(pod: Pod, container: str = Containers.KSERVE_CONTAINER_NAME) -> bool:
    """Return the readOnly field from the pod spec for the /mnt/models volumeMount.

    Inspects the Kubernetes pod spec (the webhook's output) to check whether the
    admission webhook set volumeMount.readOnly on the /mnt/models mount. Returns
    True if readOnly is set, False otherwise (including when the field is absent,
    which Kubernetes treats as read-write).

    Raises AssertionError if /mnt/models is not found in the container's volumeMounts.
    """
    for container_spec in pod.instance.spec.containers:
        if container_spec.name == container:
            for vm in container_spec.volumeMounts:
                if vm.mountPath == MODELS_MOUNT_PATH:
                    return bool(vm.readOnly)
    raise AssertionError(f"volumeMount for {MODELS_MOUNT_PATH} not found in container {container}")


def get_mount_mode(pod: Pod, container: str = Containers.KSERVE_CONTAINER_NAME) -> str:
    """Return 'ro' or 'rw' for the /mnt/models mount by inspecting /proc/mounts.

    Reads the container's /proc/mounts and parses each line in fstab(5) format
    (device, mountpoint, fstype, options, dump, pass). Looks for entries where
    the mountpoint matches /mnt/models and extracts 'ro' or 'rw' from the
    comma-separated mount options field. Returns the mode from the last matching
    entry, since the kernel honours the last mount for a given path.

    Raises AssertionError if /mnt/models is not found in /proc/mounts.
    """
    output = pod.execute(container=container, command=MOUNT_CHECK_COMMAND)
    mode = None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[1] == MODELS_MOUNT_PATH:
            LOGGER.info(f"Pod {pod.name} /mnt/models mount: {line}")
            mount_opts = parts[3].split(",")
            if "ro" in mount_opts:
                mode = "ro"
            elif "rw" in mount_opts:
                mode = "rw"
    if mode:
        return mode
    raise AssertionError(f"Mount point {MODELS_MOUNT_PATH} not found in /proc/mounts")


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
    """Validate PVC write access control via the KServe read-only storage annotation.

    Steps:
        1. Deploy a raw deployment ISVC with a ReadWriteMany NFS PVC and no explicit read-only annotation.
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

    def test_isvc_read_only_pod_spec_default(self, first_predictor_pod):
        """Test that the pod spec has readOnly=true by default (webhook contract)"""
        assert get_volume_mount_readonly(pod=first_predictor_pod), (
            "Expected volumeMount.readOnly=true on /mnt/models by default"
        )

    def test_isvc_read_only_mount_default(self, first_predictor_pod):
        """Test that /mnt/models is mounted read-only by default (runtime effect)"""
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
    def test_isvc_read_only_pod_spec_false(self, unprivileged_client, patched_read_only_isvc):
        """Test that the pod spec has readOnly=false when annotation is false (webhook contract)"""
        new_pod = get_pods_by_isvc_label(
            client=unprivileged_client,
            isvc=patched_read_only_isvc,
        )[0]
        assert not get_volume_mount_readonly(pod=new_pod), (
            "Expected volumeMount.readOnly=false on /mnt/models with readonly=false"
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
    def test_isvc_read_only_mount_false(self, unprivileged_client, patched_read_only_isvc):
        """Test that /mnt/models is mounted read-write when annotation is false (runtime effect)"""
        new_pod = get_pods_by_isvc_label(
            client=unprivileged_client,
            isvc=patched_read_only_isvc,
        )[0]
        assert get_mount_mode(pod=new_pod) == "rw", "Expected /mnt/models to be mounted read-write with readonly=false"

    @pytest.mark.parametrize(
        "patched_read_only_isvc",
        [
            pytest.param(
                {"readonly": "true", "rollout": False},
            ),
        ],
        indirect=True,
    )
    def test_isvc_read_only_pod_spec_true(self, unprivileged_client, patched_read_only_isvc):
        """Verify that the pod spec has readOnly=true when annotation is true (webhook contract)."""
        new_pod = get_pods_by_isvc_label(
            client=unprivileged_client,
            isvc=patched_read_only_isvc,
        )[0]
        assert get_volume_mount_readonly(pod=new_pod), (
            "Expected volumeMount.readOnly=true on /mnt/models with readonly=true"
        )

    @pytest.mark.parametrize(
        "patched_read_only_isvc",
        [
            pytest.param(
                {"readonly": "true", "rollout": False},
            ),
        ],
        indirect=True,
    )
    def test_isvc_read_only_mount_true(self, unprivileged_client, patched_read_only_isvc):
        """Verify that /mnt/models is mounted read-only when annotation is true (runtime effect)."""
        new_pod = get_pods_by_isvc_label(
            client=unprivileged_client,
            isvc=patched_read_only_isvc,
        )[0]
        assert get_mount_mode(pod=new_pod) == "ro", "Expected /mnt/models to be mounted read-only with readonly=true"

    @pytest.mark.parametrize(
        "patched_read_only_isvc, expected_mode",
        [
            pytest.param({"readonly": "false"}, "rw", id="test_readonly_annotation_false"),
            pytest.param({"readonly": "true"}, "ro", id="test_readonly_annotation_true"),
        ],
        indirect=["patched_read_only_isvc"],
    )
    def test_isvc_readonly_toggle_pod_spec(
        self,
        unprivileged_client: DynamicClient,
        patched_read_only_isvc: InferenceService,
        expected_mode: str,
    ) -> None:
        """Verify pod spec readOnly reflects each annotation toggle on a live ISVC (webhook contract).

        Regression coverage: RHOAIENG-8288 — annotation toggle on a live ISVC did not update
        the effective mount access mode across transitions.
        """
        expected_annotation = "true" if expected_mode == "ro" else "false"
        assert (
            patched_read_only_isvc.instance.metadata.annotations.get("storage.kserve.io/readonly")
            == expected_annotation
        ), f"Expected annotation readonly={expected_annotation} was not applied"

        new_pod = get_pods_by_isvc_label(
            client=unprivileged_client,
            isvc=patched_read_only_isvc,
        )[0]
        expected_readonly = expected_mode == "ro"
        assert get_volume_mount_readonly(pod=new_pod) == expected_readonly, (
            f"Expected volumeMount.readOnly={expected_readonly} with readonly={expected_annotation}"
        )

    @pytest.mark.parametrize(
        "patched_read_only_isvc, expected_mode",
        [
            pytest.param({"readonly": "false"}, "rw", id="test_readonly_annotation_false"),
            pytest.param({"readonly": "true"}, "ro", id="test_readonly_annotation_true"),
        ],
        indirect=["patched_read_only_isvc"],
    )
    def test_isvc_readonly_toggle_mount(
        self,
        unprivileged_client: DynamicClient,
        patched_read_only_isvc: InferenceService,
        expected_mode: str,
    ) -> None:
        """Verify /mnt/models mount mode reflects each annotation toggle on a live ISVC (runtime effect).

        Regression coverage: RHOAIENG-8288 — annotation toggle on a live ISVC did not update
        the effective mount access mode across transitions.
        """
        new_pod = get_pods_by_isvc_label(
            client=unprivileged_client,
            isvc=patched_read_only_isvc,
        )[0]
        actual_mode = get_mount_mode(pod=new_pod)
        assert actual_mode == expected_mode, f"Expected /mnt/models to be mounted {expected_mode}, got {actual_mode}"
