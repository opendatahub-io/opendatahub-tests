"""Helpers and Kubernetes resource wrappers for KServe local model cache tests."""

from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import Resource
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from utilities.constants import ApiGroups
from utilities.infra import get_pods_by_isvc_label

KSERVE_LOCALMODEL_LABEL: str = f"{ApiGroups.KSERVE}/localmodel"
KSERVE_LOCALMODEL_PVC_ANNOTATION: str = f"{ApiGroups.KSERVE}/localmodel-pvc-name"
MODEL_CACHE_JOBS_NAMESPACE: str = "kserve-localmodel-jobs"
MODEL_CACHE_DOWNLOAD_SA: str = "kserve-localmodel-sa"
LOCAL_MODEL_NODE_GROUP_NAME: str = "workers"
MODEL_CACHE_AGENT_DAEMONSET: str = "kserve-localmodelnode-agent"
MNIST_ONNX_S3_PATH: str = "test-dir"


class LocalModelCache(Resource):
    """Cluster-scoped `LocalModelCache` CR (KServe `serving.kserve.io/v1alpha1`)."""

    api_group: str = Resource.ApiGroup.SERVING_KSERVE_IO
    api_version: str = "v1alpha1"

    def __init__(
        self,
        source_model_uri: str,
        model_size: str,
        node_groups: list[str],
        service_account_name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Build a `LocalModelCache` with model download source and target node groups."""
        super().__init__(**kwargs)
        self.source_model_uri = source_model_uri
        self.model_size = model_size
        self.node_groups = node_groups
        self.service_account_name = service_account_name

    def to_dict(self) -> None:
        """Populate the Kubernetes manifest for this cache."""
        super().to_dict()

        if not self.kind_dict and not self.yaml_file:
            spec: dict[str, Any] = {
                "sourceModelUri": self.source_model_uri,
                "modelSize": self.model_size,
                "nodeGroups": self.node_groups,
            }
            if self.service_account_name:
                spec["serviceAccountName"] = self.service_account_name
            self.res["spec"] = spec


class LocalModelNodeGroup(Resource):
    """`LocalModelNodeGroup` CR provisioned by the operator for model-cache workers."""

    api_group: str = Resource.ApiGroup.SERVING_KSERVE_IO
    api_version: str = "v1alpha1"


def resource_instance_to_dict(*, resource: Resource) -> dict[str, Any]:
    """Return the wrapper's live object as a plain `dict`."""
    resource.get()
    inst = resource.instance
    if hasattr(inst, "to_dict"):
        return inst.to_dict()
    if isinstance(inst, dict):
        return inst
    raise TypeError(f"Unsupported kubernetes instance type: {type(inst)!r}")


def cache_status_dict(*, cache: LocalModelCache) -> dict[str, Any]:
    """Read `status` from a `LocalModelCache` after refreshing from the API."""
    body = resource_instance_to_dict(resource=cache)
    status = body.get("status")
    return status if isinstance(status, dict) else {}


def wait_for_local_model_cache_nodes_downloaded(*, cache: LocalModelCache, timeout: int) -> dict[str, Any]:
    """Poll until every reported node reaches `NodeDownloaded` and copies are consistent.

    Args:
        cache: Active `LocalModelCache` resource handle.
        timeout: Maximum seconds to wait for downloads.

    Returns:
        The cache `status` dict when successful.

    Raises:
        AssertionError: If the cache does not become ready in time.
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
            f"LocalModelCache {cache.name} did not reach NodeDownloaded on all nodes in {timeout}s; "
            f"last status={last!r}"
        )

    pytest.fail(f"LocalModelCache {cache.name}: polling stopped before NodeDownloaded status")


def _cache_download_state_sample(*, cache: LocalModelCache) -> dict[str, Any]:
    status = cache_status_dict(cache=cache)
    node_status = status.get("nodeStatus") or {}
    copies = status.get("copies") or {}
    failed = copies.get("failed", 0)
    available = copies.get("available")
    total = copies.get("total")

    all_downloaded = bool(node_status) and all(state == "NodeDownloaded" for state in node_status.values())
    copies_ok = failed == 0 and available is not None and total is not None and available == total and available >= 1
    return {"ready": bool(all_downloaded and copies_ok), "status": status}


def assert_predictor_storage_initializer_uses_pvc(
    *,
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str,
) -> None:
    """Assert the storage initializer rewrote storage to a `pvc://` URI for the predictor Pod."""
    pods = get_pods_by_isvc_label(client=client, isvc=isvc, runtime_name=runtime_name)
    pod = pods[0]
    spec = pod.instance.spec
    init_list = spec.initContainers or []
    args_blob = " ".join(str(a) for c in init_list for a in (c.args or []))
    assert "pvc://" in args_blob, f"Expected pvc:// rewrite in init container args; pod spec init args={args_blob!r}"

    meta = pod.instance.metadata
    annotations = (meta.annotations or {}) if meta else {}
    assert annotations.get(KSERVE_LOCALMODEL_PVC_ANNOTATION), (
        f"Missing {KSERVE_LOCALMODEL_PVC_ANNOTATION} annotation on predictor pod {pod.name}"
    )
