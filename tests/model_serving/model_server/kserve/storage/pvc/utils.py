import logging

from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from timeout_sampler import TimeoutSampler

LOGGER = logging.getLogger(__name__)


def wait_for_rollout_complete(
    client: DynamicClient,
    isvc: InferenceService,
    timeout: int = 120,
) -> None:
    """Wait until the ISVC deployment rollout is complete.

    A rollout is complete when replicas == readyReplicas == updatedReplicas,
    meaning all pods are running the latest template with no stragglers from
    previous ReplicaSets.
    """
    expected = isvc.instance.spec.predictor.get("minReplicas", 1)
    label_selector = f"serving.kserve.io/inferenceservice={isvc.name}"

    def _rollout_complete() -> bool:
        deployments = list(Deployment.get(client=client, namespace=isvc.namespace, label_selector=label_selector))
        if len(deployments) != 1:
            return False
        status = deployments[0].instance.status
        ready = status.get("readyReplicas", 0)
        updated = status.get("updatedReplicas", 0)
        total = status.get("replicas", 0)
        LOGGER.info(f"rollout: replicas={total} ready={ready} updated={updated} expected={expected}")
        return ready == expected and updated == expected and total == expected

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
