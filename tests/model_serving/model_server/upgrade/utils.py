import json
from typing import Any, Generator, TypedDict

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ApiException
from simple_logger.logger import get_logger
from ocp_resources.config_map import ConfigMap
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config

from tests.model_serving.model_server.upgrade.kserve_kueue_upgrade_config import (
    KSERVE_KUEUE_POD_CPU_LIMIT,
    KSERVE_KUEUE_POD_CPU_REQUEST,
    KSERVE_KUEUE_POD_MEMORY_LIMIT,
    KSERVE_KUEUE_POD_MEMORY_REQUEST,
)
from utilities.constants import (
    Labels,
    ModelFormat,
    RuntimeTemplates,
    Protocols,
    Timeout,
)
from utilities.exceptions import PodContainersRestartError, ResourceMismatchError
from utilities.general import create_isvc_label_selector_str
from utilities.infra import get_inference_serving_runtime, get_pods_by_isvc_label
from utilities.kueue_utils import (
    ClusterQueue,
    LocalQueue,
    ResourceFlavor,
    check_gated_pods_and_running_pods,
    create_cluster_queue,
    create_local_queue,
    create_resource_flavor,
)
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.model_serving.model_server.utils import verify_inference_response

UPGRADE_BASELINE_CM_NAME = "upgrade-test-baseline"

LOGGER = get_logger(name=__name__)


def verify_pod_containers_not_restarted(client: DynamicClient, component_name: str) -> None:
    """Verify pod containers not restarted."""
    restarted_containers = {}

    for pod in Pod.get(
        dyn_client=client,
        namespace=py_config["applications_namespace"],
        label_selector=f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of={component_name}",
    ):
        if _restarted_containers := [
            container.name for container in pod.instance.status.containerStatuses if container.restartCount > 0
        ]:
            restarted_containers[pod.name] = _restarted_containers

    if restarted_containers:
        raise PodContainersRestartError(f"Containers {restarted_containers} restarted")


def _kserve_kueue_upgrade_runtime_template_kwargs(
    runtime_kwargs: dict[str, Any],
    teardown_resources: bool,
) -> dict[str, Any]:
    """Build ServingRuntimeFromTemplate kwargs for KServe Kueue upgrade tests."""
    return {
        **runtime_kwargs,
        "template_name": RuntimeTemplates.OVMS_KSERVE,
        "multi_model": False,
        "enable_http": True,
        "teardown": teardown_resources,
        "resources": {
            ModelFormat.OVMS: {
                "requests": {"cpu": KSERVE_KUEUE_POD_CPU_REQUEST, "memory": KSERVE_KUEUE_POD_MEMORY_REQUEST},
                "limits": {"cpu": KSERVE_KUEUE_POD_CPU_LIMIT, "memory": KSERVE_KUEUE_POD_MEMORY_LIMIT},
            }
        },
    }


class ISVCBaseline(TypedDict):
    isvc_observed_generation: int
    runtime_name: str
    runtime_generation: int
    pod_restart_counts: dict[str, dict[str, int]]


class ISVCKueueBaseline(ISVCBaseline):
    kueue_integration_stats: dict[str, int]
    total_copies: int
    min_replicas: int


def verify_inference_generation(isvc: InferenceService, expected_generation: int) -> None:
    """
    Verify that inference generation is equal to expected generation.

    Args:
        isvc (InferenceService): InferenceService instance
        expected_generation (int): Expected generation

    Raises:
        ResourceMismatch: If inference generation is not equal to expected generation
    """
    if isvc.instance.status.observedGeneration != expected_generation:
        raise ResourceMismatchError(f"Inference service {isvc.name} was modified")


def verify_serving_runtime_generation(isvc: InferenceService, expected_generation: int) -> None:
    """
    Verify that serving runtime generation is equal to expected generation.
    Args:
        isvc (InferenceService): InferenceService instance
        expected_generation (int): Expected generation

    Raises:
        ResourceMismatch: If serving runtime generation is not equal to expected generation
    """
    runtime = get_inference_serving_runtime(isvc=isvc)
    if runtime.instance.metadata.generation != expected_generation:
        raise ResourceMismatchError(f"Serving runtime {runtime.name} was modified")


def read_isvc_total_copies(isvc: InferenceService) -> int:
    """Return totalCopies from ISVC status without requiring a fully Loaded model state.

    Use for Kueue gating scenarios where additional replicas are pending admission and
    ``targetModelState`` may remain ``Pending`` while ``totalCopies`` reflects running
    loaded copies (see ``test_kueue_isvc_raw``).
    """
    # Refresh before reading: callers often patch the ISVC (e.g. scale minReplicas) and
    # wait for pods, so the cached instance can still have stale/missing modelStatus.copies.
    isvc.get()
    model_status = isvc.instance.status.modelStatus
    if not model_status or model_status.copies is None:
        raise AssertionError(f"modelStatus.copies not populated for InferenceService {isvc.name}")
    return model_status.copies.totalCopies


def capture_isvc_baseline(client: DynamicClient, isvc: InferenceService) -> ISVCBaseline:
    """
    Capture baseline values for an InferenceService before upgrade.

    Captures observedGeneration, runtime generation, and per-container restart
    counts so post-upgrade assertions can compare against actual pre-upgrade
    state rather than hardcoded values.

    Args:
        client: DynamicClient instance
        isvc: InferenceService instance

    Returns:
        ISVCBaseline with isvc_observed_generation, runtime_generation, and pod_restart_counts
    """
    baseline: ISVCBaseline = {
        "isvc_observed_generation": isvc.instance.status.observedGeneration,
        "runtime_name": "",
        "runtime_generation": 0,
        "pod_restart_counts": {},
    }

    runtime = get_inference_serving_runtime(isvc=isvc)
    baseline["runtime_name"] = runtime.name
    baseline["runtime_generation"] = runtime.instance.metadata.generation

    pod_restart_counts: dict[str, dict[str, int]] = {}
    pods = get_pods_by_isvc_label(client=client, isvc=isvc)
    for pod in pods:
        if pod.instance.status.containerStatuses:
            pod_restart_counts[pod.name] = {
                container.name: container.restartCount for container in pod.instance.status.containerStatuses
            }

    baseline["pod_restart_counts"] = pod_restart_counts
    LOGGER.info(f"Captured baseline for {isvc.name}: {baseline}")
    return baseline


def get_isvc_kueue_integration_stats(
    client: DynamicClient,
    isvc: InferenceService,
    runtime_name: str,
) -> dict[str, int]:
    """Get Kueue integration stats (running and gated pod counts) for a raw ISVC.

    Args:
        client: Kubernetes dynamic client.
        isvc: The InferenceService to inspect.
        runtime_name: ServingRuntime name used for pod label selection.

    Returns:
        Dict with ``running`` and ``gated`` pod counts.
    """
    pod_labels = [
        create_isvc_label_selector_str(
            isvc=isvc,
            resource_type="pod",
            runtime_name=runtime_name,
        )
    ]
    running, gated = check_gated_pods_and_running_pods(
        labels=pod_labels,
        namespace=isvc.namespace,
        admin_client=client,
    )
    return {"running": running, "gated": gated}


def capture_isvc_kueue_baseline(client: DynamicClient, isvc: InferenceService) -> ISVCKueueBaseline:
    """Capture pre-upgrade baseline for a Kueue-integrated raw InferenceService."""
    baseline = capture_isvc_baseline(client=client, isvc=isvc)
    runtime_name = baseline["runtime_name"]
    # No .get() needed - capture_isvc_baseline already refreshed the ISVC
    total_copies = read_isvc_total_copies(isvc=isvc)
    min_replicas = isvc.instance.spec.predictor.get("minReplicas", 1)
    kueue_baseline: ISVCKueueBaseline = {
        **baseline,
        "kueue_integration_stats": get_isvc_kueue_integration_stats(
            client=client,
            isvc=isvc,
            runtime_name=runtime_name,
        ),
        "total_copies": total_copies,
        "min_replicas": min_replicas,
    }
    LOGGER.info(f"Captured Kueue baseline for {isvc.name}: {kueue_baseline}")
    return kueue_baseline


def save_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baselines: dict[str, ISVCBaseline],
    cm_name: str = UPGRADE_BASELINE_CM_NAME,
) -> ConfigMap:
    """
    Save captured baselines to a ConfigMap on the cluster.

    Args:
        client: DynamicClient instance
        namespace: Namespace where the ConfigMap will be created
        baselines: Dict mapping ISVC names to their baseline dicts

    Returns:
        The created ConfigMap
    """
    cm = ConfigMap(client=client, name=cm_name, namespace=namespace)
    if not cm.exists:
        cm = ConfigMap(
            client=client,
            name=cm_name,
            namespace=namespace,
            data={"baseline": json.dumps(baselines)},
        )
        cm.deploy()
        return cm

    # Optimistic retry loop to avoid dropping entries if concurrent writers update
    # the same baseline ConfigMap around the same time.
    last_conflict: Exception | None = None
    for _ in range(5):
        try:
            cm = ConfigMap(client=client, name=cm_name, namespace=namespace)
            if not cm.exists:
                cm = ConfigMap(
                    client=client,
                    name=cm_name,
                    namespace=namespace,
                    data={"baseline": json.dumps(baselines)},
                )
                cm.deploy()
                return cm

            cm_data = cm.instance.data or {}
            existing_data = json.loads(cm_data.get("baseline", "{}"))
            existing_data.update(baselines)
            resource_dict = cm.instance.to_dict()
            resource_dict.setdefault("data", {})
            resource_dict["data"]["baseline"] = json.dumps(existing_data)
            cm.update(resource_dict=resource_dict)
            return cm
        except ApiException as exc:
            if exc.status == 409:
                last_conflict = exc
                continue
            raise
        except Exception:
            # Re-raise any other exceptions
            raise

    raise AssertionError(
        f"Failed to update baseline ConfigMap '{cm_name}' due to repeated update conflicts."
    ) from last_conflict


def load_baseline_from_configmap(
    client: DynamicClient,
    namespace: str,
    cm_name: str = UPGRADE_BASELINE_CM_NAME,
) -> dict[str, ISVCBaseline]:
    """
    Load baselines from the ConfigMap on the cluster.

    Args:
        client: DynamicClient instance
        namespace: Namespace where the ConfigMap was created
        cm_name: Name of the ConfigMap to load from

    Returns:
        Dict mapping ISVC names to their baseline dicts

    Raises:
        AssertionError: If ConfigMap does not exist or has no baseline data
    """
    cm = ConfigMap(
        client=client,
        name=cm_name,
        namespace=namespace,
    )

    if not cm.exists:
        raise AssertionError(
            f"Baseline ConfigMap '{cm_name}' not found in namespace '{namespace}'. "
            f"Ensure pre-upgrade tests ran successfully."
        )

    cm_data = cm.instance.data or {}
    raw = cm_data.get("baseline")
    if not raw:
        raise AssertionError(f"Baseline ConfigMap '{cm_name}' has no 'baseline' key in data.")

    return json.loads(raw)


def verify_isvc_pods_not_restarted_against_baseline(
    client: DynamicClient,
    isvc: InferenceService,
    baseline_restart_counts: dict[str, dict[str, int]],
    *,
    allow_new_pods: bool = False,
) -> None:
    """
    Verify that pod restart counts have not increased since the pre-upgrade baseline.

    Args:
        client: DynamicClient instance
        isvc: InferenceService instance
        baseline_restart_counts: Pre-upgrade restart counts per pod per container
        allow_new_pods: When True, additional pods are permitted (e.g. Kueue-gated pods
            that were not in the baseline because they had no containerStatuses yet).

    Raises:
        PodContainersRestartError: If any baseline pod is missing or any container's
            restart count increased
    """
    pods = get_pods_by_isvc_label(client=client, isvc=isvc)
    increased_containers: dict[str, list[str]] = {}

    current_pod_names = {pod.name for pod in pods}
    baseline_pod_names = set(baseline_restart_counts.keys())
    missing_pods = baseline_pod_names - current_pod_names
    new_pods = current_pod_names - baseline_pod_names
    if missing_pods:
        raise PodContainersRestartError(f"Baseline pods missing after upgrade for {isvc.name}: {sorted(missing_pods)}")
    if new_pods and not allow_new_pods:
        raise PodContainersRestartError(
            f"Unexpected new pods found for {isvc.name} (new pods not allowed): {sorted(new_pods)}"
        )

    for pod in pods:
        if pod.name not in baseline_restart_counts:
            if allow_new_pods:
                statuses = pod.instance.status.containerStatuses or []
                for container in statuses:
                    if container.restartCount > 0:
                        increased_containers.setdefault(pod.name, []).append(
                            f"{container.name} (pre=0, post={container.restartCount})"
                        )
            continue
        statuses = pod.instance.status.containerStatuses or []
        pod_baseline = baseline_restart_counts[pod.name]
        if not statuses and pod_baseline:
            raise PodContainersRestartError(
                f"Container statuses missing after upgrade for pod {pod.name}; "
                f"baseline expected {sorted(pod_baseline.keys())}"
            )

        current_container_names = {container.name for container in statuses}
        missing_containers = set(pod_baseline.keys()) - current_container_names
        if missing_containers:
            raise PodContainersRestartError(
                f"Container set changed after upgrade for pod {pod.name}: "
                f"missing containers {sorted(missing_containers)}"
            )

        for container in statuses:
            if container.name not in pod_baseline:
                raise PodContainersRestartError(
                    f"Container set changed after upgrade for pod {pod.name}: new container '{container.name}'"
                )
            pre_count = pod_baseline[container.name]
            if container.restartCount > pre_count:
                increased_containers.setdefault(pod.name, []).append(
                    f"{container.name} (pre={pre_count}, post={container.restartCount})"
                )

    if increased_containers:
        raise PodContainersRestartError(f"Container restart counts increased after upgrade: {increased_containers}")


def _read_isvc_status_url(isvc: InferenceService) -> str:
    """Return status.url when present."""
    return isvc.instance.status.url or ""


def wait_for_isvc_inference_url(isvc: InferenceService, timeout: int = Timeout.TIMEOUT_2MIN) -> str:
    """Wait until the ISVC status reports an external inference URL.

    Args:
        isvc: InferenceService to poll.
        timeout: Maximum wait time in seconds.

    Returns:
        The ISVC status URL string.

    Raises:
        pytest.fail: If the URL is not populated within the timeout.
    """
    last_url = ""
    try:
        for last_url in TimeoutSampler(
            wait_timeout=timeout,
            sleep=5,
            func=lambda: _read_isvc_status_url(isvc),
        ):
            if last_url:
                LOGGER.info(f"ISVC '{isvc.name}' inference URL ready: {last_url}")
                return last_url
    except TimeoutExpiredError:
        pytest.fail(
            f"Timeout waiting for inference URL on ISVC '{isvc.name}' in namespace '{isvc.namespace}'. "
            f"Last status.url={last_url!r}. Ensure external_route=True and the OpenShift Route is reconciled."
        )
    raise AssertionError(f"ISVC '{isvc.name}' has no status.url")


def verify_kserve_kueue_upgrade_inference(
    inference_service: InferenceService,
    inference_config: dict[str, Any],
    inference_type: str,
    inference_timeout: int | None = None,
) -> None:
    """Verify post-upgrade inference via the external route (no port-forward).

    The Python ``portforward`` library uses a Rust kube client that fails TLS verification
    on some ROSA kubeconfigs (``unable to get local issuer certificate``). External routes
    use curl with ``--insecure`` on managed clusters instead, avoiding API-server port-forward.

    Args:
        inference_service: InferenceService to query.
        inference_config: Inference request/response configuration.
        inference_type: Inference type key in ``inference_config``.
        inference_timeout: Retry timeout in seconds for the inference request.
    """
    visibility = (inference_service.labels or {}).get(Labels.Kserve.NETWORKING_KSERVE_IO)
    if visibility != Labels.Kserve.EXPOSED:
        pytest.fail(
            f"ISVC '{inference_service.name}' is not exposed (networking.kserve.io/visibility={visibility!r}). "
            "Set external_route=True when creating the ISVC and re-run pre-upgrade."
        )

    wait_for_isvc_inference_url(isvc=inference_service)

    verify_inference_response(
        inference_service=inference_service,
        inference_config=inference_config,
        inference_type=inference_type,
        protocol=Protocols.HTTPS,
        use_default_query=True,
        # Raw OpenShift Routes are not covered by the istio/knative CA that get_ca_bundle()
        # returns by default; skip TLS verify (curl --insecure) for managed-cluster routes.
        insecure=True,
        inference_timeout=inference_timeout,
    )


def _create_kueue_upgrade_resources(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    namespace: str,
    local_queue_name: str,
    cluster_queue_name: str,
    resource_flavor_name: str,
    cpu_quota: int,
    memory_quota: str,
    teardown_resources: bool,
) -> Generator[LocalQueue, Any, Any]:
    """Create or look up Kueue resources for upgrade tests."""
    if pytestconfig.option.post_upgrade:
        local_queue = LocalQueue(
            client=admin_client,
            name=local_queue_name,
            cluster_queue=cluster_queue_name,
            namespace=namespace,
        )
        cluster_queue = ClusterQueue(client=admin_client, name=cluster_queue_name)
        resource_flavor = ResourceFlavor(client=admin_client, name=resource_flavor_name)
        missing_resources = [
            resource_label
            for resource_label, resource in (
                (f"LocalQueue '{local_queue_name}' in namespace '{namespace}'", local_queue),
                (f"ClusterQueue '{cluster_queue_name}'", cluster_queue),
                (f"ResourceFlavor '{resource_flavor_name}'", resource_flavor),
            )
            if not resource.exists
        ]
        if missing_resources:
            pytest.fail(
                "[POST-UPGRADE] Kueue resources missing after upgrade: "
                f"{'; '.join(missing_resources)}. "
                "Ensure pre-upgrade KServe+Kueue tests completed successfully."
            )
        yield local_queue
        if teardown_resources:
            local_queue.clean_up()
            ClusterQueue(client=admin_client, name=cluster_queue_name).clean_up()
            ResourceFlavor(client=admin_client, name=resource_flavor_name).clean_up()
    else:
        local_queue = LocalQueue(
            client=admin_client,
            name=local_queue_name,
            cluster_queue=cluster_queue_name,
            namespace=namespace,
        )
        cluster_queue = ClusterQueue(client=admin_client, name=cluster_queue_name)
        resource_flavor = ResourceFlavor(client=admin_client, name=resource_flavor_name)
        stale_resources = [
            resource_label
            for resource_label, resource in (
                (f"LocalQueue '{local_queue_name}' in namespace '{namespace}'", local_queue),
                (f"ClusterQueue '{cluster_queue_name}'", cluster_queue),
                (f"ResourceFlavor '{resource_flavor_name}'", resource_flavor),
            )
            if resource.exists
        ]
        if stale_resources:
            pytest.fail(
                f"Stale upgrade resources found: {', '.join(stale_resources)}. "
                "Clear them before pre-upgrade or use --delete-pre-upgrade-resources."
            )
        with (
            create_resource_flavor(
                client=admin_client,
                name=resource_flavor_name,
                teardown=teardown_resources,
            ),
            create_cluster_queue(
                client=admin_client,
                name=cluster_queue_name,
                resource_groups=kueue_resource_groups(
                    flavor_name=resource_flavor_name,
                    cpu_quota=cpu_quota,
                    memory_quota=memory_quota,
                ),
                teardown=teardown_resources,
            ),
            create_local_queue(
                client=admin_client,
                name=local_queue_name,
                cluster_queue=cluster_queue_name,
                namespace=namespace,
                teardown=teardown_resources,
            ) as local_queue,
        ):
            yield local_queue


def _capture_and_save_isvc_kueue_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    isvc: InferenceService,
) -> None:
    """Capture Kueue ISVC baseline and save ConfigMap in the ISVC namespace. No-op during post-upgrade."""
    if pytestconfig.option.post_upgrade:
        return

    baselines = {
        isvc.name: capture_isvc_kueue_baseline(client=admin_client, isvc=isvc),
    }
    save_baseline_to_configmap(
        client=admin_client,
        namespace=isvc.namespace,
        baselines=baselines,
    )


def kueue_resource_groups(
    flavor_name: str,
    cpu_quota: int,
    memory_quota: str,
) -> list[dict[str, Any]]:
    """Return Kueue ClusterQueue resource group spec for upgrade tests."""
    return [
        {
            "coveredResources": ["cpu", "memory"],
            "flavors": [
                {
                    "name": flavor_name,
                    "resources": [
                        {"name": "cpu", "nominalQuota": cpu_quota},
                        {"name": "memory", "nominalQuota": memory_quota},
                    ],
                }
            ],
        }
    ]
