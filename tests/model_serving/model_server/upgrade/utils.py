import json

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.gateway import Gateway
from ocp_resources.inference_service import InferenceService
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.prometheus import Prometheus
from ocp_resources.route import Route

from utilities.constants import Annotations
from utilities.exceptions import PodContainersRestartError, ResourceMismatchError
from utilities.infra import get_inference_serving_runtime, get_pods_by_isvc_label

UPGRADE_BASELINE_CM_NAME = "upgrade-test-baseline"


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


def verify_auth_enabled(isvc: InferenceService) -> None:
    """
    Verify that authentication is enabled on the InferenceService.

    Args:
        isvc: InferenceService instance to verify

    Raises:
        AssertionError: If authentication annotation is not set to 'true'
    """
    annotations = isvc.instance.metadata.annotations or {}
    auth_value = annotations.get(Annotations.KserveAuth.SECURITY)

    if auth_value != "true":
        raise AssertionError(
            f"Authentication not enabled on InferenceService {isvc.name}. "
            f"Expected annotation '{Annotations.KserveAuth.SECURITY}' to be 'true', got '{auth_value}'"
        )


def verify_model_status_loaded(isvc: InferenceService) -> None:
    """
    Verify that the model status is in Loaded state.

    Args:
        isvc: InferenceService instance to verify

    Raises:
        AssertionError: If model is not in Loaded state or not UpToDate
    """
    model_status = isvc.instance.status.modelStatus

    if not model_status:
        raise AssertionError(f"Model status not available for InferenceService {isvc.name}")

    active_state = model_status.states.activeModelState
    target_state = model_status.states.targetModelState
    transition_status = model_status.transitionStatus

    if active_state != "Loaded":
        raise AssertionError(
            f"Model not loaded for InferenceService {isvc.name}. "
            f"Expected activeModelState 'Loaded', got '{active_state}'"
        )

    if target_state != "Loaded":
        raise AssertionError(
            f"Model target state incorrect for InferenceService {isvc.name}. "
            f"Expected targetModelState 'Loaded', got '{target_state}'"
        )

    if transition_status != "UpToDate":
        raise AssertionError(
            f"Model not up to date for InferenceService {isvc.name}. "
            f"Expected transitionStatus 'UpToDate', got '{transition_status}'"
        )


def verify_isvc_pods_not_restarted(client: DynamicClient, isvc: InferenceService, max_restarts: int = 0) -> None:
    """
    Verify that pods associated with the InferenceService have not restarted.

    Args:
        client: DynamicClient instance
        isvc: InferenceService instance
        max_restarts: Maximum allowed restart count (default 0)

    Raises:
        PodContainersRestartError: If any container has restarted more than max_restarts times
    """
    pods = get_pods_by_isvc_label(client=client, isvc=isvc)
    restarted_containers: dict[str, list[str]] = {}

    for pod in pods:
        if pod.instance.status.containerStatuses:
            for container in pod.instance.status.containerStatuses:
                if container.restartCount > max_restarts:
                    if pod.name not in restarted_containers:
                        restarted_containers[pod.name] = []
                    restarted_containers[pod.name].append(f"{container.name} (restarts: {container.restartCount})")

    if restarted_containers:
        raise PodContainersRestartError(f"Containers restarted: {restarted_containers}")


def verify_storage_uri_unchanged(isvc: InferenceService, expected_uri: str) -> None:
    """
    Verify that the storage URI has not changed.

    Args:
        isvc: InferenceService instance
        expected_uri: Expected storage URI

    Raises:
        ResourceMismatchError: If storage URI has changed
    """
    actual_uri = isvc.instance.spec.predictor.model.storageUri

    if actual_uri != expected_uri:
        raise ResourceMismatchError(
            f"Storage URI changed for InferenceService {isvc.name}. Expected '{expected_uri}', got '{actual_uri}'"
        )


def verify_metrics_configmap_exists(isvc: InferenceService) -> ConfigMap:
    """
    Verify that the metrics dashboard ConfigMap exists.

    Args:
        isvc: InferenceService instance

    Returns:
        ConfigMap: The metrics dashboard ConfigMap

    Raises:
        AssertionError: If ConfigMap does not exist or is not properly configured
    """
    metrics_cm_name = f"{isvc.name}-metrics-dashboard"
    metrics_cm = ConfigMap(
        client=isvc.client,
        name=metrics_cm_name,
        namespace=isvc.namespace,
    )

    if not metrics_cm.exists:
        raise AssertionError(
            f"Metrics dashboard ConfigMap '{metrics_cm_name}' not found in namespace '{isvc.namespace}'"
        )

    supported_value = metrics_cm.instance.data.get("supported")
    if supported_value != "true":
        raise AssertionError(
            f"Metrics dashboard ConfigMap '{metrics_cm_name}' has 'supported: {supported_value}'. "
            f"Expected 'supported: true' for metrics to be available."
        )

    return metrics_cm


def verify_metrics_retained(
    prometheus: Prometheus,
    query: str,
    min_value: int,
    timeout: int = 240,
) -> None:
    """
    Verify that metrics are retained and meet minimum threshold.

    Args:
        prometheus: Prometheus instance
        query: Prometheus query string
        min_value: Minimum expected value
        timeout: Timeout in seconds to wait for metrics (default 240)

    Raises:
        AssertionError: If metrics are not retained or below threshold
    """
    from timeout_sampler import TimeoutExpiredError, TimeoutSampler

    logger = structlog.get_logger(name=__name__)

    try:
        for sample in TimeoutSampler(
            wait_timeout=timeout,
            sleep=15,
            func=lambda: prometheus.query_sampler(query=query),
        ):
            if sample:
                metric_values = [value for metric_val in sample for value in metric_val.get("value", [])]
                if metric_values and len(metric_values) >= 2:
                    value = int(float(metric_values[1]))
                    if value >= min_value:
                        logger.info(f"Metrics value {value} meets minimum threshold {min_value}")
                        return
                    logger.info(f"Current metrics value: {value}, waiting for: {min_value}")
            else:
                logger.info(f"No metrics found yet for query: {query}")
    except TimeoutExpiredError:
        raise AssertionError(f"Timed out waiting for metrics. Query: {query}, minimum expected: {min_value}") from None


def get_metrics_value(prometheus: Prometheus, query: str) -> int | None:
    """
    Get the current value of a metrics query.

    Args:
        prometheus: Prometheus instance
        query: Prometheus query string

    Returns:
        int | None: The metrics value or None if not found
    """
    result = prometheus.query_sampler(query=query)

    if not result:
        return None

    metric_values = [value for metric_val in result for value in metric_val.get("value", [])]
    if not metric_values or len(metric_values) < 2:
        return None

    return int(float(metric_values[1]))


def verify_private_endpoint_url(isvc: InferenceService) -> None:
    """
    Verify that the InferenceService has an internal cluster URL (private endpoint).

    Args:
        isvc: InferenceService instance to verify

    Raises:
        AssertionError: If URL is not in internal cluster format
    """
    if not isvc.instance.status or not isvc.instance.status.address:
        raise AssertionError(f"InferenceService {isvc.name} does not have an address in status")

    url = isvc.instance.status.address.url
    namespace_suffix = f".{isvc.namespace}.svc.cluster.local"

    if not url or namespace_suffix not in url or not url.startswith(f"http://{isvc.name}"):
        raise AssertionError(
            f"InferenceService {isvc.name} does not have internal cluster URL. "
            f"Expected URL starting with 'http://{isvc.name}' and containing '{namespace_suffix}', got '{url}'"
        )


def verify_no_external_route(client: DynamicClient, isvc: InferenceService) -> None:
    """
    Verify that no external Route exists for the InferenceService.

    Args:
        client: DynamicClient instance
        isvc: InferenceService instance

    Raises:
        AssertionError: If an external Route exists for this InferenceService
    """
    routes = list(
        Route.get(
            client=client,
            namespace=isvc.namespace,
            label_selector=f"serving.kserve.io/inferenceservice={isvc.name}",
        )
    )

    if routes:
        route_names = [route.name for route in routes]
        raise AssertionError(
            f"External Route(s) found for private InferenceService {isvc.name}: {route_names}. "
            f"Private endpoints should not have external routes."
        )


def verify_isvc_internal_access(isvc: InferenceService) -> str:
    """
    Get the internal service URL for an InferenceService.

    Args:
        isvc: InferenceService instance

    Returns:
        str: The internal service URL

    Raises:
        AssertionError: If internal URL is not available
    """
    if not isvc.instance.status or not isvc.instance.status.address:
        raise AssertionError(f"InferenceService {isvc.name} does not have status.address")

    url = isvc.instance.status.address.url
    if not url:
        raise AssertionError(f"InferenceService {isvc.name} has empty URL in status.address")

    return url


def verify_llmd_pods_not_restarted(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    max_restarts: int = 0,
) -> None:
    """
    Verify that workload pods for an LLMInferenceService have not restarted.

    Args:
        client: DynamicClient instance
        llmisvc: LLMInferenceService instance
        max_restarts: Maximum allowed restart count (default 0)

    Raises:
        PodContainersRestartError: If any container has restarted more than max_restarts times
    """
    from tests.model_serving.model_server.llmd.utils import get_llmd_workload_pods

    pods = get_llmd_workload_pods(client=client, llmisvc=llmisvc)
    restarted_containers: dict[str, list[str]] = {}

    for pod in pods:
        if pod.instance.status.containerStatuses:
            for container in pod.instance.status.containerStatuses:
                if container.restartCount > max_restarts:
                    restarted_containers.setdefault(pod.name, []).append(
                        f"{container.name} (restarts: {container.restartCount})"
                    )

    if restarted_containers:
        raise PodContainersRestartError(f"LLMD workload containers restarted: {restarted_containers}")


def verify_llmd_router_not_restarted(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    max_restarts: int = 0,
) -> None:
    """
    Verify that the router-scheduler pod for an LLMInferenceService has not restarted.

    Args:
        client: DynamicClient instance
        llmisvc: LLMInferenceService instance
        max_restarts: Maximum allowed restart count (default 0)

    Raises:
        PodContainersRestartError: If any container has restarted more than max_restarts times
    """
    from tests.model_serving.model_server.llmd.utils import get_llmd_router_scheduler_pod

    router_pod = get_llmd_router_scheduler_pod(client=client, llmisvc=llmisvc)
    if not router_pod:
        raise PodContainersRestartError(f"Router-scheduler pod not found for {llmisvc.name}")

    restarted_containers: dict[str, list[str]] = {}
    if router_pod.instance.status.containerStatuses:
        for container in router_pod.instance.status.containerStatuses:
            if container.restartCount > max_restarts:
                restarted_containers.setdefault(router_pod.name, []).append(
                    f"{container.name} (restarts: {container.restartCount})"
                )

    if restarted_containers:
        raise PodContainersRestartError(f"LLMD router-scheduler containers restarted: {restarted_containers}")


def verify_gateway_accepted(gateway: Gateway) -> None:
    """
    Verify that a Gateway resource exists and has an Accepted condition.

    Args:
        gateway: Gateway instance

    Raises:
        AssertionError: If gateway does not exist or is not accepted
    """
    if not gateway.exists:
        raise AssertionError(f"Gateway {gateway.name} does not exist in namespace {gateway.namespace}")

    conditions = gateway.instance.status.get("conditions", [])
    is_accepted = any(
        condition.get("type") == "Accepted" and condition.get("status") == "True" for condition in conditions
    )
    if not is_accepted:
        raise AssertionError(f"Gateway {gateway.name} is not Accepted. Conditions: {conditions}")


def capture_isvc_baseline(client: DynamicClient, isvc: InferenceService) -> dict:
    """
    Capture baseline values for an InferenceService before upgrade.

    Captures observedGeneration, runtime generation, and per-container restart
    counts so post-upgrade assertions can compare against actual pre-upgrade
    state rather than hardcoded values.

    Args:
        client: DynamicClient instance
        isvc: InferenceService instance

    Returns:
        dict with isvc_observed_generation, runtime_generation, and pod_restart_counts
    """
    logger = structlog.get_logger(name=__name__)

    baseline: dict = {
        "isvc_observed_generation": isvc.instance.status.observedGeneration,
    }

    runtime = get_inference_serving_runtime(isvc=isvc)
    baseline["runtime_name"] = runtime.name
    baseline["runtime_generation"] = runtime.instance.metadata.generation

    pod_restart_counts: dict[str, dict[str, int]] = {}
    pods = get_pods_by_isvc_label(client=client, isvc=isvc)
    for pod in pods:
        if pod.instance.status.containerStatuses:
            pod_restart_counts[pod.name] = {
                container.name: container.restartCount
                for container in pod.instance.status.containerStatuses
            }

    baseline["pod_restart_counts"] = pod_restart_counts
    logger.info(f"Captured baseline for {isvc.name}: {baseline}")
    return baseline


def save_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baselines: dict[str, dict],
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
    cm = ConfigMap(
        client=client,
        name=UPGRADE_BASELINE_CM_NAME,
        namespace=namespace,
    )

    if cm.exists:
        existing_data = json.loads(cm.instance.data.get("baseline", "{}"))
        existing_data.update(baselines)
        resource_dict = cm.instance.to_dict()
        resource_dict["data"]["baseline"] = json.dumps(existing_data)
        cm.update(resource_dict=resource_dict)
    else:
        cm = ConfigMap(
            client=client,
            name=UPGRADE_BASELINE_CM_NAME,
            namespace=namespace,
            data={"baseline": json.dumps(baselines)},
        )
        cm.deploy()

    return cm


def load_baseline_from_configmap(
    client: DynamicClient,
    namespace: str,
) -> dict[str, dict]:
    """
    Load baselines from the ConfigMap on the cluster.

    Args:
        client: DynamicClient instance
        namespace: Namespace where the ConfigMap was created

    Returns:
        Dict mapping ISVC names to their baseline dicts

    Raises:
        AssertionError: If ConfigMap does not exist or has no baseline data
    """
    cm = ConfigMap(
        client=client,
        name=UPGRADE_BASELINE_CM_NAME,
        namespace=namespace,
    )

    if not cm.exists:
        raise AssertionError(
            f"Baseline ConfigMap '{UPGRADE_BASELINE_CM_NAME}' not found in namespace '{namespace}'. "
            f"Ensure pre-upgrade tests ran successfully."
        )

    raw = cm.instance.data.get("baseline")
    if not raw:
        raise AssertionError(
            f"Baseline ConfigMap '{UPGRADE_BASELINE_CM_NAME}' has no 'baseline' key in data."
        )

    return json.loads(raw)


def verify_isvc_pods_not_restarted_against_baseline(
    client: DynamicClient,
    isvc: InferenceService,
    baseline_restart_counts: dict[str, dict[str, int]],
) -> None:
    """
    Verify that pod restart counts have not increased since the pre-upgrade baseline.

    Args:
        client: DynamicClient instance
        isvc: InferenceService instance
        baseline_restart_counts: Pre-upgrade restart counts per pod per container

    Raises:
        PodContainersRestartError: If any container's restart count increased
    """
    pods = get_pods_by_isvc_label(client=client, isvc=isvc)
    increased_containers: dict[str, list[str]] = {}

    for pod in pods:
        if not pod.instance.status.containerStatuses:
            continue
        pod_baseline = baseline_restart_counts.get(pod.name, {})
        for container in pod.instance.status.containerStatuses:
            pre_count = pod_baseline.get(container.name, 0)
            if container.restartCount > pre_count:
                increased_containers.setdefault(pod.name, []).append(
                    f"{container.name} (pre={pre_count}, post={container.restartCount})"
                )

    if increased_containers:
        raise PodContainersRestartError(
            f"Container restart counts increased after upgrade: {increased_containers}"
        )
