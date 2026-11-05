import uuid
from typing import Any

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.exceptions import ExecOnPodError
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

from utilities.general import namespace_fs_group
from utilities.image_constants import SharedImages

LOGGER = structlog.get_logger(name=__name__)


def get_running_predictor_pods_raw(client: DynamicClient, isvc: InferenceService) -> list:
    """Return the currently Running predictor pods (raw objects) for the ISVC.

    Uses field_selector to filter Running pods at the API level (raw=True
    avoids extra API calls).
    """
    label_selector = f"serving.kserve.io/inferenceservice={isvc.name}"
    return list(
        Pod.get(
            client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
            field_selector="status.phase=Running",
            raw=True,
        )
    )


def get_predictor_pod_uids(client: DynamicClient, isvc: InferenceService) -> set:
    """Return UIDs of the currently Running predictor pods for the ISVC."""
    return {p.metadata.uid for p in get_running_predictor_pods_raw(client=client, isvc=isvc)}


def get_isvc_readonly_flag(isvc: InferenceService) -> bool:
    """Return the readOnly flag the controller derives from the ISVC annotation.

    Mirrors KServe: only an explicit "false" makes the storage read-write; any
    other value (including a missing annotation) defaults to read-only.
    """
    return isvc.instance.metadata.annotations.get("storage.kserve.io/readonly") != "false"


def wait_for_rollout_complete(
    client: DynamicClient,
    isvc: InferenceService,
    timeout: int = 120,
    previous_pod_uids: set | None = None,
) -> None:
    """Wait until the ISVC deployment rollout is complete.

    A rollout is complete when replicas == readyReplicas == updatedReplicas,
    meaning all pods are running the latest template with no stragglers from
    previous ReplicaSets.

    When `previous_pod_uids` is provided (snapshot of pod UIDs taken before
    the change that triggers the rollout), the wait additionally requires a
    Running pod with a UID not in that snapshot whose `/mnt/models`
    volumeMount matches the readOnly value the ISVC annotation currently
    requests. Without this, the convergence check can pass immediately
    against the still-converged old deployment before the controller has
    observed the change, or accept a newly-created pod from an old
    ReplicaSet (e.g. recreated after an eviction), causing the caller to
    inspect a stale pod.
    """
    expected = isvc.instance.spec.predictor.get("minReplicas", 1)
    label_selector = f"serving.kserve.io/inferenceservice={isvc.name}"
    expected_readonly = get_isvc_readonly_flag(isvc=isvc)

    def _rollout_complete() -> bool:
        deployments = list(Deployment.get(client=client, namespace=isvc.namespace, label_selector=label_selector))
        if len(deployments) != 1:
            return False
        status = deployments[0].instance.status
        ready = status.get("readyReplicas", 0)
        updated = status.get("updatedReplicas", 0)
        total = status.get("replicas", 0)
        LOGGER.info(f"rollout: replicas={total} ready={ready} updated={updated} expected={expected}")
        if not (ready == expected and updated == expected and total == expected):
            return False
        if previous_pod_uids is None:
            return True
        for raw_pod in get_running_predictor_pods_raw(client=client, isvc=isvc):
            if raw_pod.metadata.uid in previous_pod_uids:
                continue
            pod = Pod(client=client, name=raw_pod.metadata.name, namespace=raw_pod.metadata.namespace)
            if get_volume_mount_readonly(pod=pod) == expected_readonly:
                LOGGER.info(
                    f"rollout: new pod {raw_pod.metadata.name} runs the requested template "
                    f"(volumeMount.readOnly={expected_readonly})"
                )
                return True
        return False

    for sample in TimeoutSampler(wait_timeout=timeout, sleep=2, func=_rollout_complete):
        if sample:
            return


def get_running_predictor_pod(client: DynamicClient, isvc: InferenceService) -> Pod:
    """Return the most recently created Running predictor pod for the ISVC.

    Uses field_selector to filter Running pods at the API level, then picks
    the newest by creationTimestamp (raw=True avoids extra API calls).
    """
    label_selector = f"serving.kserve.io/inferenceservice={isvc.name}"
    raw_pods = list(
        Pod.get(
            client=client,
            namespace=isvc.namespace,
            label_selector=label_selector,
            field_selector="status.phase=Running",
            raw=True,
        )
    )
    assert raw_pods, f"No Running pods found for {isvc.name}"
    for p in raw_pods:
        LOGGER.info(f"Running pod: {p.metadata.name} created={p.metadata.creationTimestamp} uid={p.metadata.uid}")
    newest = max(raw_pods, key=lambda p: p.metadata.creationTimestamp)
    return Pod(client=client, name=newest.metadata.name, namespace=newest.metadata.namespace)


MOUNT_CHECK_COMMAND = ["cat", "/proc/mounts"]
MODELS_MOUNT_PATH = "/mnt/models"


def get_volume_mount_readonly(pod: Pod, container: str = "kserve-container") -> bool:
    """Return the readOnly field from the pod spec for the /mnt/models volumeMount."""
    for container_spec in pod.instance.spec.containers:
        if container_spec.name == container:
            for vm in container_spec.volumeMounts:
                if vm.mountPath == MODELS_MOUNT_PATH:
                    return bool(vm.readOnly)
    raise AssertionError(f"volumeMount for {MODELS_MOUNT_PATH} not found in container {container}")


def get_mount_mode(pod: Pod, container: str = "kserve-container") -> str:
    """Return 'ro' or 'rw' for the /mnt/models mount by inspecting /proc/mounts."""
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


def log_write_debug_info(pod: Pod, container: str = "kserve-container") -> None:
    """Log uid/gid, directory listing, and volumeMount spec for write-permission diagnostics."""
    try:
        uid_output = pod.execute(container=container, command=["id"])
        LOGGER.info(f"[write-debug] pod={pod.name} id: {uid_output.strip()}")
    except ExecOnPodError as exc:
        LOGGER.info(f"[write-debug] pod={pod.name} id FAILED: {exc}")

    try:
        ls_output = pod.execute(container=container, command=["ls", "-la", MODELS_MOUNT_PATH])
        LOGGER.info(f"[write-debug] pod={pod.name} ls -la {MODELS_MOUNT_PATH}:\n{ls_output}")
    except ExecOnPodError as exc:
        LOGGER.info(f"[write-debug] pod={pod.name} ls FAILED: {exc}")

    for cs in pod.instance.spec.containers:
        if cs.name == container:
            for vm in cs.volumeMounts:
                if vm.mountPath == MODELS_MOUNT_PATH:
                    LOGGER.info(
                        f"[write-debug] pod={pod.name} volumeMount: "
                        f"subPath={vm.get('subPath', '<none>')} readOnly={vm.get('readOnly', '<unset>')}"
                    )


def try_write_file(pod: Pod, container: str = "kserve-container") -> bool:
    """Attempt to create a test file on /mnt/models. Returns True on success."""
    test_file = f"{MODELS_MOUNT_PATH}/.write-test-{uuid.uuid4().hex[:8]}"

    log_write_debug_info(pod=pod, container=container)
    try:
        pod.execute(container=container, command=["touch", test_file])
        LOGGER.info(f"[write-debug] pod={pod.name} touch {test_file} SUCCEEDED")
        return True
    except ExecOnPodError as exc:
        LOGGER.info(f"[write-debug] pod={pod.name} touch {test_file} FAILED: {type(exc).__name__}: {exc}")
        return False


def chmod_model_directory(
    client: DynamicClient,
    namespace: str,
    pvc_name: str,
    model_path: str,
    mode: str = "777",
) -> None:
    """Run a pod to chmod the model subdirectory on the PVC.

    The download pod creates the model directory as UID 1000 (busybox image
    USER) with 755 permissions. The serving pod runs under the namespace SCC
    UID range, so it cannot write to a 755 directory it doesn't own. A
    customer preparing a PVC for read-write serving would chmod the directory
    themselves — this helper simulates that step.
    """
    target_path = f"/mnt/models/{model_path}"
    fs_group = namespace_fs_group(client=client, namespace=namespace)

    pod_kwargs: dict[str, Any] = {
        "client": client,
        "namespace": namespace,
        "name": "chmod-model-dir",
        "containers": [
            {
                "name": "chmod",
                "image": SharedImages.BUSYBOX,
                "command": ["chmod", mode, target_path],
                "volumeMounts": [{"mountPath": "/mnt/models/", "name": pvc_name}],
                "securityContext": {
                    "allowPrivilegeEscalation": False,
                    "capabilities": {"drop": ["ALL"]},
                    "runAsNonRoot": True,
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
            }
        ],
        "volumes": [{"name": pvc_name, "persistentVolumeClaim": {"claimName": pvc_name}}],
        "restart_policy": "Never",
    }
    if fs_group is not None:
        pod_kwargs["security_context"] = {
            "fsGroup": fs_group,
            "seccompProfile": {"type": "RuntimeDefault"},
        }

    with Pod(**pod_kwargs) as pod:
        pod.wait_for_status(status=Pod.Status.SUCCEEDED, timeout=120)
        LOGGER.info(f"chmod {mode} {target_path} completed")
