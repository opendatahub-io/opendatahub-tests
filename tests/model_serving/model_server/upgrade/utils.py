import json
from typing import TypedDict

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.gateway import Gateway
from ocp_resources.inference_service import InferenceService
from ocp_resources.llm_inference_service import LLMInferenceService
from ocp_resources.prometheus import Prometheus
from ocp_resources.route import Route
from ocp_resources.secret import Secret

from utilities.constants import Annotations
from utilities.exceptions import PodContainersRestartError, ResourceMismatchError
from utilities.infra import get_inference_serving_runtime, get_pods_by_isvc_label

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_BASELINE_CM_NAME = "upgrade-test-baseline"
UPGRADE_LLMD_BASELINE_CM_NAME = "upgrade-llmd-test-baseline"
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
                        LOGGER.info(f"Metrics value {value} meets minimum threshold {min_value}")
                        return
                    LOGGER.info(f"Current metrics value: {value}, waiting for: {min_value}")
            else:
                LOGGER.info(f"No metrics found yet for query: {query}")
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

    LOGGER.info(event=f"[VERIFY] Gateway check: '{gateway.name}' in ns '{gateway.namespace}'")
    LOGGER.info(event=f"[VERIFY] Gateway exists: {gateway.exists}")
    if not gateway.exists:
        raise AssertionError(f"Gateway {gateway.name} does not exist in namespace {gateway.namespace}")

    conditions = gateway.instance.status.get("conditions", [])
    LOGGER.info(event=f"[VERIFY] Gateway conditions: {conditions}")
    is_accepted = any(
        condition.get("type") == "Accepted" and condition.get("status") == "True" for condition in conditions
    )
    LOGGER.info(event=f"[VERIFY] Gateway Accepted: {is_accepted}")
    if not is_accepted:
        raise AssertionError(f"Gateway {gateway.name} is not Accepted. Conditions: {conditions}")
    LOGGER.info(event=f"[VERIFY] PASS: Gateway '{gateway.name}' is Accepted")


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


def save_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baselines: dict,
    cm_name: str = UPGRADE_BASELINE_CM_NAME,
) -> ConfigMap:
    """Save captured baselines to a ConfigMap on the cluster."""
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
) -> dict:
    """Load baselines from a ConfigMap on the cluster."""
    cm = ConfigMap(client=client, name=cm_name, namespace=namespace)

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
# LLMISVC upgrade baseline and verification
# ---------------------------------------------------------------------------


class LLMISVCBaseline(TypedDict):
    spec_generation: int
    url: str
    replicas: int
    config_ref_names: list[str]
    container_images: dict[str, dict[str, str]]
    restart_counts: dict[str, dict[str, int]]


def capture_llmisvc_baseline(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> LLMISVCBaseline:
    """Capture pre-upgrade state for an LLMInferenceService."""

    spec = llmisvc.instance.spec
    pods = _get_all_llmisvc_pods(client=client, llmisvc=llmisvc)

    LOGGER.info(event=f"[BASELINE] Capturing baseline for LLMISVC '{llmisvc.name}' in ns '{llmisvc.namespace}'")
    LOGGER.info(event=f"[BASELINE] Found {len(pods)} pod(s): {[p.name for p in pods]}")

    generation = llmisvc.instance.metadata.generation
    url = _get_llmisvc_url(status=llmisvc.instance.status)
    replicas = getattr(spec, "replicas", 1) or 1
    config_ref_names = _extract_config_ref_names(llmisvc=llmisvc)
    container_images = _collect_container_images(pods=pods)
    restart_counts = _collect_restart_counts(pods=pods)

    LOGGER.info(event=f"[BASELINE] spec_generation={generation}")
    LOGGER.info(event=f"[BASELINE] url='{url}'")
    LOGGER.info(event=f"[BASELINE] replicas={replicas}")
    LOGGER.info(event=f"[BASELINE] config_ref_names ({len(config_ref_names)}): {config_ref_names}")
    if not config_ref_names:
        LOGGER.warning(
            event="[BASELINE] WARNING: No config refs found in LLMISVC status annotations. "
            "test_llmd_config_refs_survive_upgrade will be a NO-OP. "
            f"Expected annotations with prefix '{_CONFIG_REF_ANNOTATION_PREFIX}' in status.annotations."
        )
    for pod_name, images in container_images.items():
        for cname, cimage in images.items():
            LOGGER.info(event=f"[BASELINE] container_image: pod={pod_name} container={cname} image={cimage}")
    for pod_name, counts in restart_counts.items():
        for cname, count in counts.items():
            LOGGER.info(event=f"[BASELINE] restart_count: pod={pod_name} container={cname} count={count}")

    baseline: LLMISVCBaseline = {
        "spec_generation": generation,
        "url": url,
        "replicas": replicas,
        "config_ref_names": config_ref_names,
        "container_images": container_images,
        "restart_counts": restart_counts,
    }
    return baseline


_CONFIG_REF_ANNOTATION_PREFIX = "serving.kserve.io/config-llm-"


def _extract_config_ref_names(llmisvc: LLMInferenceService) -> list[str]:
    """Extract LLMInferenceServiceConfig CR names from LLMISVC status annotations.

    The controller stores config ref names as status annotations with the prefix
    ``serving.kserve.io/config-llm-``. Each annotation value is the name of a
    LLMInferenceServiceConfig CR in the ``redhat-ods-applications`` namespace.
    """

    refs: list[str] = []

    status = llmisvc.instance.status
    annotations = getattr(status, "annotations", None) or {}
    if isinstance(annotations, dict):
        status_annotations = annotations
    else:
        status_annotations = dict(annotations) if annotations else {}

    for key, value in status_annotations.items():
        if key.startswith(_CONFIG_REF_ANNOTATION_PREFIX) and value:
            LOGGER.info(event=f"[BASELINE] Found config ref annotation: {key}={value}")
            refs.append(value)

    return sorted(set(refs))


def _attr(obj: object, key: str, default: str = "") -> str:
    """Access a key from a dict or attribute-style object uniformly."""
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def _get_llmisvc_url(status: object) -> str:
    """Extract the URL from an LLMISVC status, checking addresses then url field."""
    addresses = getattr(status, "addresses", None) or []
    if addresses:
        url = addresses[0].get("url", "")
        if url:
            return url
    return getattr(status, "url", "") or ""


def _get_all_llmisvc_pods(client: DynamicClient, llmisvc: LLMInferenceService) -> list:
    """Fetch all LLMISVC pods (workload + router) in a single pass."""
    from tests.model_serving.model_server.llmd.utils import (
        get_llmd_router_scheduler_pod,
        get_llmd_workload_pods,
    )

    pods = list(get_llmd_workload_pods(client=client, llmisvc=llmisvc))
    router_pod = get_llmd_router_scheduler_pod(client=client, llmisvc=llmisvc)
    if router_pod:
        pods.append(router_pod)
    return pods


def _collect_container_images(pods: list) -> dict[str, dict[str, str]]:
    """Collect container images from a list of pods."""
    return {pod.name: {c.name: c.image for c in (pod.instance.spec.containers or [])} for pod in pods}


def _collect_restart_counts(pods: list) -> dict[str, dict[str, int]]:
    """Collect per-container restart counts from a list of pods."""
    return {pod.name: {c.name: c.restartCount for c in (pod.instance.status.containerStatuses or [])} for pod in pods}


def verify_llmisvc_generation_unchanged(
    llmisvc: LLMInferenceService,
    baseline: LLMISVCBaseline,
) -> None:
    """Verify spec was not mutated by comparing metadata.generation."""

    current = llmisvc.instance.metadata.generation
    expected = baseline["spec_generation"]
    LOGGER.info(event=f"[VERIFY] Generation check for '{llmisvc.name}': pre={expected}, post={current}")
    if current != expected:
        raise ResourceMismatchError(
            f"LLMInferenceService {llmisvc.name} spec was mutated during upgrade "
            f"(generation: pre={expected}, post={current})"
        )
    LOGGER.info(event=f"[VERIFY] PASS: Generation unchanged ({current})")


def verify_llmisvc_status_fields(
    llmisvc: LLMInferenceService,
    baseline: LLMISVCBaseline,
) -> None:
    """Verify Ready condition, URL, and replicas survived upgrade."""

    status = llmisvc.instance.status
    conditions = getattr(status, "conditions", None) or []

    LOGGER.info(event=f"[VERIFY] Status fields check for '{llmisvc.name}'")
    LOGGER.info(event=f"[VERIFY] All conditions: {conditions}")

    ready = next((condition for condition in conditions if _attr(obj=condition, key="type") == "Ready"), None)
    if not ready:
        raise AssertionError(f"LLMInferenceService {llmisvc.name} has no Ready condition after upgrade")

    ready_status = _attr(obj=ready, key="status")
    LOGGER.info(event=f"[VERIFY] Ready condition status='{ready_status}'")
    if ready_status != "True":
        raise AssertionError(f"LLMInferenceService {llmisvc.name} is not Ready after upgrade: {ready}")
    LOGGER.info(event="[VERIFY] PASS: Ready condition is True")

    current_url = _get_llmisvc_url(status=status)
    expected_url = baseline["url"]
    LOGGER.info(event=f"[VERIFY] URL check: pre='{expected_url}', post='{current_url}'")
    if expected_url and current_url != expected_url:
        raise ResourceMismatchError(
            f"LLMInferenceService {llmisvc.name} URL changed: pre='{expected_url}', post='{current_url}'"
        )
    LOGGER.info(event="[VERIFY] PASS: URL unchanged")

    current_replicas = getattr(llmisvc.instance.spec, "replicas", 1) or 1
    expected_replicas = baseline["replicas"]
    LOGGER.info(event=f"[VERIFY] Replicas check: pre={expected_replicas}, post={current_replicas}")
    if current_replicas != expected_replicas:
        raise ResourceMismatchError(
            f"LLMInferenceService {llmisvc.name} replicas changed: pre={expected_replicas}, post={current_replicas}"
        )
    LOGGER.info(event=f"[VERIFY] PASS: Replicas unchanged ({current_replicas})")


def verify_llmisvc_container_images_unchanged(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    baseline: LLMISVCBaseline,
) -> None:
    """Verify container images were not changed during upgrade."""

    pods = _get_all_llmisvc_pods(client=client, llmisvc=llmisvc)
    current_images = _collect_container_images(pods=pods)
    baseline_images = baseline["container_images"]

    LOGGER.info(event=f"[VERIFY] Container images check for '{llmisvc.name}'")
    LOGGER.info(event=f"[VERIFY] Baseline pods: {list(baseline_images.keys())}")
    LOGGER.info(event=f"[VERIFY] Current pods:  {list(current_images.keys())}")

    mismatches: list[str] = []
    for pod_name, containers in baseline_images.items():
        if pod_name not in current_images:
            LOGGER.warning(event=f"[VERIFY] MISSING: pod '{pod_name}' from baseline not found after upgrade")
            mismatches.append(f"pod {pod_name} missing after upgrade")
            continue
        for container_name, expected_image in containers.items():
            actual_image = current_images.get(pod_name, {}).get(container_name, "")
            if actual_image != expected_image:
                LOGGER.warning(
                    event=f"[VERIFY] MISMATCH: {pod_name}/{container_name}: "
                    f"pre='{expected_image}', post='{actual_image}'"
                )
                mismatches.append(f"{pod_name}/{container_name}: pre='{expected_image}', post='{actual_image}'")
            else:
                LOGGER.info(event=f"[VERIFY] OK: {pod_name}/{container_name} image unchanged: {actual_image}")

    if mismatches:
        raise ResourceMismatchError(f"Container images changed for {llmisvc.name}: {'; '.join(mismatches)}")
    LOGGER.info(event=f"[VERIFY] PASS: All container images unchanged across {len(baseline_images)} pod(s)")


LLMISVC_CONFIG_NAMESPACE = "redhat-ods-applications"


def verify_llmisvc_config_refs_exist(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    baseline: LLMISVCBaseline,
) -> None:
    """Verify that LLMInferenceServiceConfig CRs referenced pre-upgrade still exist.

    Catches regressions like RHOAIENG-65791 where configs are deleted during upgrade.
    LLMInferenceServiceConfig CRs are created by the controller in redhat-ods-applications,
    not in the LLMISVC's namespace.
    """

    config_ref_names = baseline["config_ref_names"]

    LOGGER.info(
        event=f"[VERIFY] Config refs check for '{llmisvc.name}': "
        f"{len(config_ref_names)} ref(s) to verify in ns '{LLMISVC_CONFIG_NAMESPACE}': {config_ref_names}"
    )

    if not config_ref_names:
        LOGGER.warning(
            event="[VERIFY] WARNING: baseline has 0 config refs — this check is a NO-OP. "
            "No status annotations with prefix 'serving.kserve.io/config-llm-' were found pre-upgrade."
        )
        return

    from utilities.resources.llm_inference_service_config import LLMInferenceServiceConfig

    missing: list[str] = []
    for config_name in config_ref_names:
        config_cr = LLMInferenceServiceConfig(
            client=client,
            name=config_name,
            namespace=LLMISVC_CONFIG_NAMESPACE,
        )
        exists = config_cr.exists
        if exists:
            LOGGER.info(event=f"[VERIFY] LLMInferenceServiceConfig '{config_name}': exists=True")
        else:
            LOGGER.warning(event=f"[VERIFY] LLMInferenceServiceConfig '{config_name}': exists=False")
            missing.append(config_name)

    if missing:
        raise AssertionError(
            f"LLMInferenceServiceConfig CRs deleted during upgrade for {llmisvc.name} "
            f"in ns '{LLMISVC_CONFIG_NAMESPACE}': {missing}"
        )
    LOGGER.info(event=f"[VERIFY] PASS: All {len(config_ref_names)} config ref(s) still exist")


def verify_llmisvc_pods_not_restarted_against_baseline(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
    baseline: LLMISVCBaseline,
) -> None:
    """Verify LLMISVC pod restart counts have not increased since baseline."""

    baseline_counts = baseline["restart_counts"]
    pods = _get_all_llmisvc_pods(client=client, llmisvc=llmisvc)
    current_names = {pod.name for pod in pods}
    baseline_names = set(baseline_counts.keys())

    LOGGER.info(event=f"[VERIFY] Pod restart counts check for '{llmisvc.name}'")
    LOGGER.info(event=f"[VERIFY] Baseline pods ({len(baseline_names)}): {baseline_names}")
    LOGGER.info(event=f"[VERIFY] Current pods  ({len(current_names)}): {current_names}")

    missing_pods = baseline_names - current_names
    if missing_pods:
        LOGGER.error(event=f"[VERIFY] MISSING PODS: {missing_pods}")
        raise PodContainersRestartError(
            f"LLMISVC pods from baseline missing after upgrade for {llmisvc.name}: {missing_pods}"
        )

    new_pods = current_names - baseline_names
    if new_pods:
        LOGGER.warning(event=f"[VERIFY] New pods not in baseline (will check with restartCount > 0): {new_pods}")

    increased: dict[str, list[str]] = {}
    for pod in pods:
        pod_baseline = baseline_counts.get(pod.name, {})
        for container in pod.instance.status.containerStatuses or []:
            pre_count = pod_baseline.get(container.name, 0)
            post_count = container.restartCount
            if post_count > pre_count:
                LOGGER.warning(
                    event=f"[VERIFY] RESTART: pod={pod.name} container={container.name} "
                    f"pre={pre_count} post={post_count}"
                )
                increased.setdefault(pod.name, []).append(f"{container.name} (pre={pre_count}, post={post_count})")
            else:
                LOGGER.info(
                    event=f"[VERIFY] OK: pod={pod.name} container={container.name} "
                    f"restartCount pre={pre_count} post={post_count}"
                )

    if increased:
        raise PodContainersRestartError(f"LLMISVC pod restart counts increased after upgrade: {increased}")
    LOGGER.info(event=f"[VERIFY] PASS: No container restarts across {len(pods)} pod(s)")
