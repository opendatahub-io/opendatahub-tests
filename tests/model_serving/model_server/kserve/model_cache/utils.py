"""Helpers and Kubernetes resource wrappers for KServe local model cache tests."""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from ocp_resources.resource import Resource
from pyhelper_utils.shell import run_command
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.llmd.utils import get_llmd_vllm_pods
from utilities.constants import ApiGroups
from utilities.infra import get_pods_by_isvc_label
from utilities.resources.llm_inference_service import LLMInferenceService
from utilities.resources.local_model_namespace_cache import LocalModelNamespaceCache

KSERVE_LOCALMODEL_LABEL: str = f"internal.{ApiGroups.KSERVE}/localmodel"
KSERVE_LOCALMODEL_PVC_ANNOTATION: str = f"internal.{ApiGroups.KSERVE}/localmodel-pvc-name"
KSERVE_PVC_SOURCE_VOLUME_NAME: str = "kserve-pvc-source"
LOCAL_MODEL_NODE_GROUP_NAME: str = "workers"
MODEL_CACHE_AGENT_DAEMONSET: str = "kserve-localmodelnode-agent"
MODEL_CACHE_NODE_PVC_NAME: str = "kserve-localmodelnode-pvc"
MODEL_CACHE_HOST_PATH: str = "/var/lib/kserve/models"
MODEL_CACHE_STORAGE_CLASS: str = "local-storage"
MODEL_CACHE_SIZE: str = "1Gi"
MODEL_CACHE_NODE_COUNT: int = 2
MINT_ONNX_STORAGE_PATH: str = "test-dir"


class LocalModelNodeGroup(Resource):
    """`LocalModelNodeGroup` CR provisioned by the operator for model-cache worker nodes."""

    api_group: str = Resource.ApiGroup.SERVING_KSERVE_IO


class LocalModelNode(Resource):
    """`LocalModelNode` CR representing a node participating in model caching."""

    api_group: str = Resource.ApiGroup.SERVING_KSERVE_IO


def resource_instance_to_dict(*, resource: Resource) -> dict[str, Any]:
    """Return the wrapper's live object as a plain ``dict``."""
    resource.get()
    inst = resource.instance
    if hasattr(inst, "to_dict"):
        return inst.to_dict()
    if isinstance(inst, dict):
        return inst
    raise TypeError(f"Unsupported kubernetes instance type: {type(inst)!r}")


def cache_status_dict(*, cache: LocalModelNamespaceCache) -> dict[str, Any]:
    """Read ``status`` from a ``LocalModelNamespaceCache`` after refreshing from the API."""
    body = resource_instance_to_dict(resource=cache)
    status = body.get("status")
    return status if isinstance(status, dict) else {}


def wait_for_local_model_cache_nodes_downloaded(*, cache: LocalModelNamespaceCache, timeout: int) -> dict[str, Any]:
    """Poll until every reported node reaches ``NodeDownloaded`` and copies are consistent.

    Args:
        cache: Active ``LocalModelNamespaceCache`` resource handle.
        timeout: Maximum seconds to wait for downloads.

    Returns:
        The cache ``status`` dict when successful.

    Raises:
        AssertionError: If the cache does not become ready within *timeout* seconds.
    """
    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=15,
            func=lambda: _cache_download_state_sample(cache=cache),
        ):
            if sample["ready"]:
                return sample["status"]
    except TimeoutExpiredError:
        last = cache_status_dict(cache=cache)
        pytest.fail(
            f"LocalModelNamespaceCache {cache.name} did not reach NodeDownloaded on all nodes "
            f"in {timeout}s; last status={last!r}"
        )

    pytest.fail(f"LocalModelNamespaceCache {cache.name}: polling stopped before NodeDownloaded status")


def _cache_download_state_sample(*, cache: LocalModelNamespaceCache) -> dict[str, Any]:
    status = cache_status_dict(cache=cache)
    node_status = status.get("nodeStatus") or {}
    copies = status.get("copies") or {}
    failed = copies.get("failed", 0)
    available = copies.get("available")
    total = copies.get("total")

    all_downloaded = bool(node_status) and all(state == "NodeDownloaded" for state in node_status.values())
    copies_ok = failed == 0 and available is not None and total is not None and available == total and available >= 1
    return {"ready": bool(all_downloaded and copies_ok), "status": status}


def wait_for_shared_pvc_cache_condition(
    *,
    cache: LocalModelNamespaceCache,
    expected_status: str,
    expected_reasons: set[str],
    timeout: int,
) -> dict[str, Any]:
    """Wait for a shared-PVC cache Ready condition to reach an expected state."""
    try:
        for status in TimeoutSampler(
            wait_timeout=timeout,
            sleep=15,
            func=lambda: cache_status_dict(cache=cache),
        ):
            ready = next((condition for condition in status.get("conditions", []) if condition["type"] == "Ready"), {})
            if ready.get("status") == expected_status and ready.get("reason") in expected_reasons:
                return status
    except TimeoutExpiredError:
        pytest.fail(
            f"LocalModelNamespaceCache {cache.namespace}/{cache.name} did not reach "
            f"Ready={expected_status} with reason in {sorted(expected_reasons)}; "
            f"last status={cache_status_dict(cache=cache)!r}"
        )

    pytest.fail(f"LocalModelNamespaceCache {cache.namespace}/{cache.name}: condition polling stopped unexpectedly")


def _assert_pod_has_pvc_source_volume(*, pod: Any) -> None:
    """Assert a Pod mounts the ``kserve-pvc-source`` volume and it is PVC-backed."""
    spec = pod.instance.spec
    volume_names = [vol.name for vol in (spec.volumes or [])]
    pvc_volumes = [vol for vol in (spec.volumes or []) if vol.name == KSERVE_PVC_SOURCE_VOLUME_NAME]
    assert pvc_volumes, (
        f"Expected '{KSERVE_PVC_SOURCE_VOLUME_NAME}' PVC volume on pod {pod.name}; volumes={volume_names!r}"
    )
    assert any(
        getattr(vol, "persistent_volume_claim", None) or getattr(vol, "persistentVolumeClaim", None)
        for vol in pvc_volumes
    ), f"Volume '{KSERVE_PVC_SOURCE_VOLUME_NAME}' on pod {pod.name} exists but is not PVC-backed"


def assert_predictor_uses_cached_pvc(
    *,
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str,
) -> None:
    """Assert the local model cache PVC is mounted directly on the predictor Pod.

    KServe rewrites ``pvc://`` URIs to a direct volume mount (no storage-initializer
    init container), so we verify:
    1. The ``kserve-pvc-source`` volume exists and is PVC-backed.
    2. The pod carries the ``internal.kserve.io/localmodel-pvc-name`` annotation.
    """
    pods = get_pods_by_isvc_label(client=client, isvc=isvc, runtime_name=runtime_name)
    assert pods, f"No predictor pods found for InferenceService {isvc.namespace}/{isvc.name}"
    pod = pods[0]
    _assert_pod_has_pvc_source_volume(pod=pod)

    meta = pod.instance.metadata
    annotations = (meta.annotations or {}) if meta else {}
    assert annotations.get(KSERVE_LOCALMODEL_PVC_ANNOTATION), (
        f"Missing {KSERVE_LOCALMODEL_PVC_ANNOTATION} annotation on predictor pod {pod.name}"
    )


def assert_predictors_share_read_only_pvc(
    *,
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str,
    pvc_name: str,
) -> list[Pod]:
    """Assert two cross-node predictors mount one PVC read-only without local model transfer."""
    pods = get_pods_by_isvc_label(client=client, isvc=isvc, runtime_name=runtime_name)
    assert len(pods) == 2, f"Expected exactly two predictor pods, got {[pod.name for pod in pods]}"
    assert len({pod.instance.spec.nodeName for pod in pods}) == 2, "Predictor replicas did not span two nodes"

    for pod in pods:
        spec = pod.instance.spec
        init_names = {container.name for container in (spec.initContainers or [])}
        assert "storage-initializer" not in init_names, f"storage-initializer unexpectedly present on {pod.name}"

        volumes = {volume.name: volume for volume in (spec.volumes or [])}
        pvc_volume = volumes.get(KSERVE_PVC_SOURCE_VOLUME_NAME)
        claim = getattr(pvc_volume, "persistentVolumeClaim", None) if pvc_volume else None
        assert claim and claim.claimName == pvc_name, (
            f"Pod {pod.name} expected PVC {pvc_name!r}, got {getattr(claim, 'claimName', None)!r}"
        )

        model_mounts = [
            mount
            for container in spec.containers
            for mount in (container.volumeMounts or [])
            if mount.mountPath.startswith("/mnt/models")
        ]
        assert model_mounts and all(mount.readOnly for mount in model_mounts), (
            f"Pod {pod.name} must mount model data read-only"
        )
        empty_dirs = {name for name, volume in volumes.items() if getattr(volume, "emptyDir", None) is not None}
        assert not empty_dirs.intersection(mount.name for mount in model_mounts), (
            f"Pod {pod.name} stores model data in emptyDir volumes {empty_dirs}"
        )

    return pods


def assert_modelcar_absent_from_node_image_cache(*, pods: list[Pod], source_uri: str) -> None:
    """Assert CRI-O did not pull the source ModelCar as a container image."""
    image_reference = source_uri.removeprefix("oci://").split("@", maxsplit=1)[0]
    last_slash = image_reference.rfind("/")
    last_colon = image_reference.rfind(":")
    repository = image_reference[:last_colon] if last_colon > last_slash else image_reference

    for node_name in {pod.instance.spec.nodeName for pod in pods}:
        _, output, _ = run_command(
            command=["oc", "debug", f"node/{node_name}", "--", "chroot", "/host", "crictl", "images"],
            verify_stderr=False,
            check=True,
            timeout=180,
        )
        assert repository not in output, f"ModelCar {repository} unexpectedly exists in CRI-O on {node_name}"


def assert_llmisvc_uses_cached_pvc(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> None:
    """Assert the local model cache PVC rewrite is active for an ``LLMInferenceService``.

    Unlike ``InferenceService`` (where the pod admission webhook injects the PVC
    volume), the LLMISVC controller rewrites ``spec.model.uri`` to a ``pvc://`` URI
    directly in the reconciler, keyed off the ``internal.kserve.io/localmodel*``
    label/annotation that the LLMISVC defaulting webhook sets on the CR itself. Verify
    both the CR-level markers and the resulting workload Pod volume mount.
    """
    llmisvc.get()
    meta = llmisvc.instance.metadata
    labels = (meta.labels or {}) if meta else {}
    annotations = (meta.annotations or {}) if meta else {}

    assert labels.get(KSERVE_LOCALMODEL_LABEL), (
        f"Missing {KSERVE_LOCALMODEL_LABEL} label on LLMInferenceService {llmisvc.namespace}/{llmisvc.name}"
    )
    assert annotations.get(KSERVE_LOCALMODEL_PVC_ANNOTATION), (
        f"Missing {KSERVE_LOCALMODEL_PVC_ANNOTATION} annotation on LLMInferenceService "
        f"{llmisvc.namespace}/{llmisvc.name}"
    )

    pods = get_llmd_vllm_pods(client=client, llmisvc=llmisvc)
    assert pods, f"No workload pods found for LLMInferenceService {llmisvc.namespace}/{llmisvc.name}"
    for pod in pods:
        _assert_pod_has_pvc_source_volume(pod=pod)
