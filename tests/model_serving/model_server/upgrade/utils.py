import json
from typing import TypedDict

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.gateway import Gateway
from ocp_resources.inference_service import InferenceService
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.prometheus import Prometheus
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from utilities.constants import Annotations
from utilities.exceptions import PodContainersRestartError, ResourceMismatchError
from utilities.infra import get_inference_serving_runtime, get_pods_by_isvc_label
from utilities.resources.http_route import HTTPRoute
from utilities.resources.inference_pool import InferencePool

UPGRADE_BASELINE_CM_NAME = "upgrade-test-baseline"
UPGRADE_AUTH_TOKEN_SECRET_NAME = "upgrade-test-auth-token"  # pragma: allowlist secret


class ISVCBaseline(TypedDict):
    isvc_observed_generation: int
    runtime_name: str
    runtime_generation: int
    pod_restart_counts: dict[str, dict[str, int]]


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
    logger = structlog.get_logger(name=__name__)

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
    logger.info(f"Captured baseline for {isvc.name}: {baseline}")
    return baseline


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
        except Exception as exc:
            if "409" in str(exc) or "Conflict" in str(exc):
                last_conflict = exc
                continue
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


def save_auth_token_to_secret(
    client: DynamicClient,
    namespace: str,
    token: str,
) -> None:
    """
    Persist the pre-upgrade auth token into a Secret so the post-upgrade run
    can reuse the exact same token to prove that the pre-existing auth setup
    survives the upgrade.

    A Secret is used instead of a ConfigMap to avoid storing credentials in
    plaintext cluster metadata (CWE-312).

    Args:
        client: DynamicClient instance
        namespace: Namespace where the Secret will be created
        token: The bearer token to persist
    """
    secret = Secret(
        client=client,
        name=UPGRADE_AUTH_TOKEN_SECRET_NAME,
        namespace=namespace,
    )

    if secret.exists:
        resource_dict = secret.instance.to_dict()
        resource_dict.setdefault("stringData", {})
        resource_dict["stringData"]["auth_token"] = token
        secret.update(resource_dict=resource_dict)
    else:
        Secret(
            client=client,
            name=UPGRADE_AUTH_TOKEN_SECRET_NAME,
            namespace=namespace,
            type="Opaque",
            string_data={"auth_token": token},
        ).deploy()


def load_auth_token_from_secret(
    client: DynamicClient,
    namespace: str,
) -> str:
    """
    Load the pre-upgrade auth token from the Secret.

    Args:
        client: DynamicClient instance
        namespace: Namespace where the Secret lives

    Returns:
        The pre-upgrade bearer token

    Raises:
        AssertionError: If Secret or token key is missing
    """
    import base64

    secret = Secret(
        client=client,
        name=UPGRADE_AUTH_TOKEN_SECRET_NAME,
        namespace=namespace,
    )

    if not secret.exists:
        raise AssertionError(
            f"Auth token Secret '{UPGRADE_AUTH_TOKEN_SECRET_NAME}' not found in namespace '{namespace}'. "
            f"Ensure pre-upgrade tests ran successfully."
        )

    secret_data = secret.instance.data or {}
    encoded_token = secret_data.get("auth_token")
    if not encoded_token:
        raise AssertionError(
            f"Auth token Secret '{UPGRADE_AUTH_TOKEN_SECRET_NAME}' has no 'auth_token' key. "
            f"Ensure the pre-upgrade auth tests captured the token."
        )

    return base64.b64decode(encoded_token).decode()


def get_isvc_baseline(baselines: dict[str, ISVCBaseline], isvc_name: str) -> ISVCBaseline:
    """
    Retrieve the baseline for a specific ISVC, failing fast if missing.

    Args:
        baselines: Dict mapping ISVC names to their baseline dicts
        isvc_name: Name of the InferenceService

    Returns:
        The ISVCBaseline for the given ISVC

    Raises:
        AssertionError: If the baseline is missing for this ISVC
    """
    baseline = baselines.get(isvc_name)
    assert baseline is not None, (
        f"Missing baseline for InferenceService '{isvc_name}'. "
        f"Ensure pre-upgrade tests captured the baseline. Available: {sorted(baselines.keys())}"
    )
    return baseline


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

    current_pod_names = {pod.name for pod in pods}
    baseline_pod_names = set(baseline_restart_counts.keys())
    missing_pods = baseline_pod_names - current_pod_names
    new_pods = current_pod_names - baseline_pod_names
    if missing_pods or new_pods:
        raise PodContainersRestartError(
            f"Pod set changed after upgrade for {isvc.name}. missing={sorted(missing_pods)}, new={sorted(new_pods)}"
        )

    for pod in pods:
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


# ---------------------------------------------------------------------------
# Utils functions used by LLMInferenceService upgrade tests
# ---------------------------------------------------------------------------

LOGGER = structlog.get_logger(name=__name__)


class LLMISVCBaseline(TypedDict):
    namespace: str
    pre_upgrade_rhoai_version: str
    spec_generation: int
    url: str
    replicas: int
    model_uri: str
    kueue_integration_stats: dict[str, int]
    config_ref_names: list[str]
    container_images: dict[str, dict[str, str]]
    restart_counts: dict[str, dict[str, int]]


def capture_llmisvc_baseline(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> LLMISVCBaseline:
    """Capture pre-upgrade state using the same getter functions that post-upgrade tests use."""
    from utilities.infra import get_product_version

    LOGGER.info(event=f"[BASELINE] Capturing baseline for LLMISVC '{llmisvc.name}' in ns '{llmisvc.namespace}'")

    baseline: LLMISVCBaseline = {
        "namespace": llmisvc.namespace,
        "pre_upgrade_rhoai_version": str(get_product_version(admin_client=client)),
        "spec_generation": get_llmisvc_generation(llmisvc=llmisvc),
        "url": get_llmisvc_url(llmisvc=llmisvc),
        "replicas": get_llmisvc_replicas(llmisvc=llmisvc),
        "model_uri": get_llmisvc_model_uri(llmisvc=llmisvc),
        "kueue_integration_stats": get_llmisvc_kueue_integration_stats(client=client, llmisvc=llmisvc),
        "config_ref_names": get_llmisvc_config_ref_names(llmisvc=llmisvc),
        "container_images": get_llmisvc_container_images(client=client, llmisvc=llmisvc),
        "restart_counts": get_llmisvc_restart_counts(client=client, llmisvc=llmisvc),
    }

    LOGGER.info(event=f"[BASELINE] Captured baseline for '{llmisvc.name}'", baseline=baseline)
    return baseline


def _get_llmisvc_pods(client: DynamicClient, llmisvc: LLMInferenceService) -> list:
    """Fetch all pods associated with an LLMInferenceService.

    Args:
        client: Kubernetes dynamic client.
        llmisvc: The LLMInferenceService to get pods for.

    Returns:
        List of Pod objects (workload replicas + router-scheduler).
    """
    from ocp_resources.pod import Pod

    return list(
        Pod.get(
            client=client,
            namespace=llmisvc.namespace,
            label_selector=(
                f"{Pod.ApiGroup.APP_KUBERNETES_IO}/part-of=llminferenceservice,"
                f"{Pod.ApiGroup.APP_KUBERNETES_IO}/name={llmisvc.name}"
            ),
        )
    )


# --- Getter functions ---


def get_llmisvc_generation(llmisvc: LLMInferenceService) -> int:
    """Get status.observedGeneration from an LLMInferenceService.

    Uses observedGeneration (not metadata.generation) because it only changes
    when the controller has actually reconciled a new spec — a webhook or
    defaulting that bumps metadata.generation without a real spec change
    won't trigger a false positive.

    Returns:
        The last generation the controller reconciled.
    """
    return llmisvc.instance.status.observedGeneration


def get_llmisvc_url(llmisvc: LLMInferenceService) -> str:
    """Get the serving URL from an LLMInferenceService status.

    Used by capture_llmisvc_baseline (pre-upgrade) and post-upgrade tests.

    Returns:
        The URL string.
    """
    return llmisvc.instance.status.url


def get_llmisvc_replicas(llmisvc: LLMInferenceService) -> int:
    """Get spec.replicas from an LLMInferenceService.

    Used by capture_llmisvc_baseline (pre-upgrade) and post-upgrade tests.

    Returns:
        The replica count.
    """
    return llmisvc.instance.spec.replicas


def get_llmisvc_model_uri(llmisvc: LLMInferenceService) -> str:
    """Get the model URI from an LLMInferenceService spec.

    Used by capture_llmisvc_baseline (pre-upgrade) and post-upgrade tests.

    Returns:
        The model URI string.
    """
    return llmisvc.instance.spec.model.uri


def get_llmisvc_config_ref_names(llmisvc: LLMInferenceService) -> list[str]:
    """Get LLMInferenceServiceConfig CR names from status annotations.

    The controller stores config ref names as status annotations with prefix
    ``serving.kserve.io/config-llm-``. Each annotation value is the name of a
    LLMInferenceServiceConfig CR in the redhat-ods-applications namespace.

    Used by capture_llmisvc_baseline (pre-upgrade) and post-upgrade tests.

    Returns:
        Sorted list of config ref names.
    """
    _CONFIG_REF_ANNOTATION_PREFIX = "serving.kserve.io/config-llm-"
    refs: list[str] = []
    annotations = getattr(llmisvc.instance.status, "annotations", None) or {}
    for key, value in annotations.items():
        if key.startswith(_CONFIG_REF_ANNOTATION_PREFIX) and value:
            refs.append(value)
    return sorted(refs)


def get_llmisvc_container_images(client: DynamicClient, llmisvc: LLMInferenceService) -> dict[str, dict[str, str]]:
    """Get container images for all pods associated with an LLMInferenceService.

    Used by capture_llmisvc_baseline (pre-upgrade) and post-upgrade tests.

    Returns:
        Dict mapping pod name to {container_name: image} for each pod.
    """
    pods = _get_llmisvc_pods(client=client, llmisvc=llmisvc)
    return {pod.name: {container.name: container.image for container in pod.instance.spec.containers} for pod in pods}


def get_llmisvc_restart_counts(client: DynamicClient, llmisvc: LLMInferenceService) -> dict[str, dict[str, int]]:
    """Get container restart counts for all pods associated with an LLMInferenceService.

    Used by capture_llmisvc_baseline (pre-upgrade) and post-upgrade tests.

    Returns:
        Dict mapping pod name to {container_name: restart_count} for each pod.
    """
    pods = _get_llmisvc_pods(client=client, llmisvc=llmisvc)
    return {
        pod.name: {
            container.name: container.restartCount for container in (pod.instance.status.containerStatuses or [])
        }
        for pod in pods
    }


def get_llmisvc_kueue_integration_stats(client: DynamicClient, llmisvc: LLMInferenceService) -> dict[str, int]:
    """Get Kueue integration stats (running and gated pod counts) for an LLMInferenceService.

    Used by capture_llmisvc_baseline (pre-upgrade) and post-upgrade tests.

    Returns:
        Dict with "running" and "gated" counts.
    """
    from utilities.kueue_utils import check_gated_pods_and_running_pods

    selector_labels = [f"app.kubernetes.io/name={llmisvc.name}", "kserve.io/component=workload"]
    running, gated = check_gated_pods_and_running_pods(
        labels=selector_labels,
        namespace=llmisvc.namespace,
        admin_client=client,
    )
    return {"running": running, "gated": gated}


# --- Verify functions ---
# Used by post-upgrade tests to assert cluster state matches the pre-upgrade baseline.


def verify_llmisvc_exists(llmisvc: LLMInferenceService, baseline: dict) -> None:
    """Verify LLMInferenceService exists after upgrade.

    Steps:
        1. Assert the LLMInferenceService resource exists on the cluster.
        2. Log the pre-upgrade RHOAI version from baseline.

    Args:
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If the LLMInferenceService does not exist.
    """
    assert llmisvc.exists, f"LLMInferenceService '{llmisvc.name}' not found after upgrade"
    pre_version = baseline.get("pre_upgrade_rhoai_version", "unknown")
    LOGGER.info(event=f"[POST-UPGRADE] PASS: '{llmisvc.name}' exists, pre_upgrade_rhoai_version={pre_version}")


def verify_llmisvc_generation_unchanged(llmisvc: LLMInferenceService, baseline: dict) -> None:
    """Verify observedGeneration has not changed during the upgrade.

    Steps:
        1. Get status.observedGeneration from the LLMInferenceService.
        2. Compare against the pre-upgrade baseline value.
        3. Assert generation has not changed.

    Args:
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If generation has changed.
    """
    expected = baseline["spec_generation"]
    current = get_llmisvc_generation(llmisvc=llmisvc)
    LOGGER.info(event=f"[POST-UPGRADE] generation: expected={expected}, current={current}")
    assert current == expected, f"generation changed: {expected} -> {current}"


def verify_llmisvc_url_unchanged(llmisvc: LLMInferenceService, baseline: dict) -> None:
    """Verify the serving URL has not changed during the upgrade.

    Steps:
        1. Get the serving URL from the LLMInferenceService status.
        2. Compare against the pre-upgrade baseline URL.
        3. Assert the URL has not changed.

    Args:
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If the URL has changed.
    """
    expected = baseline["url"]
    current = get_llmisvc_url(llmisvc=llmisvc)
    LOGGER.info(event=f"[POST-UPGRADE] url: expected={expected}, current={current}")
    if expected:
        assert current == expected, f"URL changed: {expected} -> {current}"


def verify_llmisvc_replicas_unchanged(llmisvc: LLMInferenceService, baseline: dict) -> None:
    """Verify spec.replicas has not changed during the upgrade.

    Steps:
        1. Get spec.replicas from the LLMInferenceService.
        2. Compare against the pre-upgrade baseline value.
        3. Assert replicas have not changed.

    Args:
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If replicas have changed.
    """
    expected = baseline["replicas"]
    current = get_llmisvc_replicas(llmisvc=llmisvc)
    LOGGER.info(event=f"[POST-UPGRADE] replicas: expected={expected}, current={current}")
    assert current == expected, f"replicas changed: {expected} -> {current}"


def verify_llmisvc_model_uri_unchanged(llmisvc: LLMInferenceService, baseline: dict) -> None:
    """Verify the model URI has not changed during the upgrade.

    Steps:
        1. Get the model URI from the LLMInferenceService spec.
        2. Compare against the pre-upgrade baseline value.
        3. Assert the model URI has not changed.

    Skips if model_uri is absent from baseline and pre-upgrade was 3.3 (field not captured).

    Args:
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If the model URI has changed.
    """
    import pytest

    pre_upgrade_rhoai_version = baseline.get("pre_upgrade_rhoai_version")
    expected = baseline.get("model_uri")
    if expected is None and (pre_upgrade_rhoai_version is None or pre_upgrade_rhoai_version.startswith("3.3")):
        reason = f"model_uri not in baseline — pre-upgrade was {pre_upgrade_rhoai_version}, field not captured"
        LOGGER.info(event=f"[POST-UPGRADE] SKIP: {reason}")
        pytest.skip(reason=reason)

    current = get_llmisvc_model_uri(llmisvc=llmisvc)
    LOGGER.info(event=f"[POST-UPGRADE] model_uri: expected={expected}, current={current}")
    assert current == expected, f"model URI changed: {expected} -> {current}"


def verify_llmisvc_container_images_unchanged(
    client: DynamicClient, llmisvc: LLMInferenceService, baseline: dict
) -> None:
    """Verify container images have not changed during the upgrade.

    Steps:
        1. Get all container images from the LLMInferenceService pods.
        2. Compare against the pre-upgrade baseline images.
        3. Assert container images have not changed.

    Args:
        client: Kubernetes dynamic client.
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If container images have changed.
    """
    expected = baseline["container_images"]
    current = get_llmisvc_container_images(client=client, llmisvc=llmisvc)
    LOGGER.info(event=f"[POST-UPGRADE] container_images: expected={expected}, current={current}")
    assert current == expected, f"container images changed: {expected} -> {current}"


def verify_llmisvc_restart_counts_unchanged(
    client: DynamicClient, llmisvc: LLMInferenceService, baseline: dict
) -> None:
    """Verify no container has restarted during the upgrade.

    Steps:
        1. Get restart counts for all containers in the LLMInferenceService pods.
        2. Compare against the pre-upgrade baseline restart counts.
        3. Assert no container has restarted.

    Args:
        client: Kubernetes dynamic client.
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If any container has restarted.
    """
    expected = baseline["restart_counts"]
    current = get_llmisvc_restart_counts(client=client, llmisvc=llmisvc)
    LOGGER.info(event=f"[POST-UPGRADE] restart_counts: expected={expected}, current={current}")
    assert current == expected, f"restart counts changed: {expected} -> {current}"


def verify_llmisvc_config_refs_unchanged(llmisvc: LLMInferenceService, baseline: dict) -> None:
    """Verify config ref names have not been silently swapped during the upgrade.

    Steps:
        1. Get the current config ref names from the LLMInferenceService status annotations.
        2. Compare against the pre-upgrade baseline config refs.
        3. Assert config refs have not changed.

    Args:
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If config refs have changed.
    """
    expected = baseline["config_ref_names"]
    current = get_llmisvc_config_ref_names(llmisvc=llmisvc)
    LOGGER.info(event=f"[POST-UPGRADE] config_refs: expected={expected}, current={current}")
    assert current == expected, f"config refs changed: {expected} -> {current}"


def verify_llmisvc_gateway(gateway: Gateway) -> None:
    """Verify that the openshift-ai-inference Gateway exists and has Accepted and Programmed conditions.

    Steps:
        1. Assert the Gateway resource exists.
        2. Assert the Gateway has Accepted condition set to True.
        3. Assert the Gateway has Programmed condition set to True.

    Args:
        gateway: The Gateway instance to verify.

    Raises:
        AssertionError: If gateway does not exist, is not accepted, or is not programmed.
    """
    LOGGER.info(event=f"[POST-UPGRADE] Gateway check: '{gateway.name}' in ns '{gateway.namespace}'")
    assert gateway.exists, f"Gateway {gateway.name} does not exist in namespace {gateway.namespace}"

    conditions = gateway.instance.status.get("conditions", [])
    LOGGER.info(event=f"[POST-UPGRADE] Gateway conditions: {conditions}")
    is_accepted = any(
        condition.get("type") == "Accepted" and condition.get("status") == "True" for condition in conditions
    )
    is_programmed = any(
        condition.get("type") == "Programmed" and condition.get("status") == "True" for condition in conditions
    )
    LOGGER.info(event=f"[POST-UPGRADE] Gateway Accepted: {is_accepted}, Programmed: {is_programmed}")
    assert is_accepted, f"Gateway {gateway.name} is not Accepted. Conditions: {conditions}"
    assert is_programmed, f"Gateway {gateway.name} is not Programmed (not routing traffic). Conditions: {conditions}"
    LOGGER.info(event=f"[POST-UPGRADE] PASS: Gateway '{gateway.name}' is Accepted and Programmed")


def verify_llmisvc_config_refs_exist(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    baseline: dict,
) -> None:
    """Verify that LLMInferenceServiceConfig CRs referenced pre-upgrade still exist.

    Steps:
        1. Get config ref names from the pre-upgrade baseline.
        2. Look up each LLMInferenceServiceConfig CR in redhat-ods-applications.
        3. Assert all config CRs still exist.

    Skips if config_ref_names is empty and pre-upgrade was 3.3 (refs not captured).

    Args:
        client: Kubernetes dynamic client.
        llmisvc: The LLMInferenceService to verify.
        baseline: Pre-upgrade baseline dict.

    Raises:
        AssertionError: If any config CR is missing after upgrade.
    """
    import pytest

    from utilities.resources.llm_inference_service_config import LLMInferenceServiceConfig

    LLMISVC_CONFIG_NAMESPACE = "redhat-ods-applications"
    config_ref_names = baseline["config_ref_names"]
    pre_upgrade_rhoai_version = baseline.get("pre_upgrade_rhoai_version")
    if not config_ref_names and (pre_upgrade_rhoai_version is None or pre_upgrade_rhoai_version.startswith("3.3")):
        pytest.skip(
            reason=f"config_ref_names empty in baseline — pre-upgrade version recorded '{pre_upgrade_rhoai_version}'."
        )
    LOGGER.info(
        event=f"[POST-UPGRADE] Config refs check for '{llmisvc.name}': "
        f"{len(config_ref_names)} ref(s) to verify in ns '{LLMISVC_CONFIG_NAMESPACE}': {config_ref_names}"
    )
    missing: list[str] = []
    for config_name in config_ref_names:
        config_cr = LLMInferenceServiceConfig(client=client, name=config_name, namespace=LLMISVC_CONFIG_NAMESPACE)
        if config_cr.exists:
            LOGGER.info(event=f"[POST-UPGRADE] LLMInferenceServiceConfig '{config_name}': found")
        else:
            LOGGER.warning(event=f"[POST-UPGRADE] LLMInferenceServiceConfig '{config_name}': MISSING")
            missing.append(config_name)
    assert not missing, f"LLMInferenceServiceConfig CRs missing after upgrade for '{llmisvc.name}': {missing}"
    LOGGER.info(event=f"[POST-UPGRADE] PASS: All {len(config_ref_names)} config ref(s) still exist")


def verify_llmisvc_controller_healthy(client: DynamicClient) -> None:
    """Verify llmisvc-controller-manager Deployment is Available.

    Steps:
        1. Get the llmisvc-controller-manager Deployment in redhat-ods-applications.
        2. Assert the Available condition is True.

    Args:
        client: Kubernetes dynamic client.

    Raises:
        AssertionError: If the Deployment does not exist or is not Available.
    """

    LOGGER.info(event="[POST-UPGRADE] Checking llmisvc-controller-manager health")
    deploy = Deployment(client=client, name="llmisvc-controller-manager", namespace="redhat-ods-applications")
    assert deploy.exists, "llmisvc-controller-manager Deployment not found"
    conditions = deploy.instance.status.conditions or []
    available = next((condition for condition in conditions if condition.type == "Available"), None)
    assert available, "llmisvc-controller-manager has no Available condition"
    LOGGER.info(event=f"[POST-UPGRADE] llmisvc-controller-manager Available: {available.status}")
    assert available.status == "True", (
        f"llmisvc-controller-manager not Available: status={available.status}, "
        f"reason={getattr(available, 'reason', 'N/A')}"
    )
    LOGGER.info(event="[POST-UPGRADE] PASS: llmisvc-controller-manager is Available")


def verify_llmisvc_inference_pool_exists(client: DynamicClient, llmisvc: LLMInferenceService) -> None:
    """Verify InferencePool exists and is owned by the LLMInferenceService.

    Steps:
        1. Look up the InferencePool resource by name convention ({llmisvc.name}-inference-pool).
        2. Assert it exists in the LLMInferenceService namespace.
        3. Assert it is owned by the LLMInferenceService via ownerReferences.

    Args:
        client: Kubernetes dynamic client.
        llmisvc: The LLMInferenceService to verify.

    Raises:
        AssertionError: If InferencePool does not exist or is not owned by the LLMInferenceService.
    """
    pool_name = f"{llmisvc.name}-inference-pool"
    LOGGER.info(event=f"[POST-UPGRADE] InferencePool check: '{pool_name}' in ns '{llmisvc.namespace}'")
    pool = InferencePool(client=client, name=pool_name, namespace=llmisvc.namespace)
    assert pool.exists, f"InferencePool '{pool_name}' not found in namespace '{llmisvc.namespace}'"
    LOGGER.info(event=f"[POST-UPGRADE] InferencePool '{pool_name}' exists")
    owner_refs = pool.instance.metadata.ownerReferences or []
    LOGGER.info(event=f"[POST-UPGRADE] InferencePool ownerReferences: {owner_refs}")
    owned = any(ref.name == llmisvc.name and ref.kind == "LLMInferenceService" for ref in owner_refs)
    assert owned, f"InferencePool '{pool_name}' is not owned by '{llmisvc.name}'. Owners: {owner_refs}"
    LOGGER.info(event=f"[POST-UPGRADE] PASS: InferencePool '{pool_name}' exists and is owned by '{llmisvc.name}'")


def verify_llmisvc_httproute_exists(client: DynamicClient, llmisvc: LLMInferenceService) -> None:
    """Verify HTTPRoute exists for the LLMInferenceService.

    Steps:
        1. List HTTPRoutes in the LLMInferenceService namespace.
        2. Find routes matching the LLMInferenceService name.
        3. Assert at least one matching HTTPRoute exists.

    Args:
        client: Kubernetes dynamic client.
        llmisvc: The LLMInferenceService to verify.

    Raises:
        AssertionError: If no matching HTTPRoute is found.
    """

    routes = list(
        HTTPRoute.get(
            client=client,
            namespace=llmisvc.namespace,
            label_selector=f"app.kubernetes.io/name={llmisvc.name},app.kubernetes.io/part-of=llminferenceservice",
        )
    )
    LOGGER.info(event=f"[POST-UPGRADE] HTTPRoute check for '{llmisvc.name}': found {len(routes)} route(s)")
    assert routes, f"No HTTPRoute found for '{llmisvc.name}' in namespace '{llmisvc.namespace}'"
    LOGGER.info(event=f"[POST-UPGRADE] HTTPRoute '{routes[0].name}' exists")
    LOGGER.info(event=f"[POST-UPGRADE] PASS: HTTPRoute exists for '{llmisvc.name}'")
