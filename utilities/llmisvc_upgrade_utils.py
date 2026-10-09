"""Helpers for checking LLMInferenceService state before and after upgrades.

The functions capture a pre-upgrade baseline, save and load it from a ConfigMap,
and read pod restart counts for post-upgrade checks.
"""

import json
from typing import TypedDict

import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod

from utilities.infra import get_product_version
from utilities.kueue_utils import check_gated_pods_and_running_pods
from utilities.resources.llm_inference_service import LLMInferenceService

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_BASELINE_CM_NAME = "upgrade-test-baseline"


class LLMISVCBaseline(TypedDict):
    """Captured pre-upgrade state for an LLMInferenceService."""

    namespace: str
    pre_upgrade_rhoai_version: str
    spec_generation: int
    url: str
    replicas: int
    model_uri: str
    kueue_integration_stats: dict[str, int]
    config_ref_names: list[str]
    config_ref_pins: dict[str, str]
    container_images: dict[str, dict[str, str]]
    restart_counts: dict[str, dict[str, int]]


def save_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baselines: dict[str, LLMISVCBaseline],
    cm_name: str = UPGRADE_BASELINE_CM_NAME,
) -> ConfigMap:
    """Save captured LLMInferenceService baselines to a ConfigMap."""
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
) -> dict[str, LLMISVCBaseline]:
    """Load LLMInferenceService baselines from a ConfigMap."""
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


def capture_llmisvc_baseline(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> LLMISVCBaseline:
    """Capture pre-upgrade state for an LLMInferenceService."""
    LOGGER.info(event=f"[BASELINE] Capturing baseline for LLMISVC '{llmisvc.name}' in ns '{llmisvc.namespace}'")

    baseline: LLMISVCBaseline = {
        "namespace": llmisvc.namespace,
        "pre_upgrade_rhoai_version": str(get_product_version(admin_client=client)),
        "spec_generation": _get_llmisvc_generation(llmisvc=llmisvc),
        "url": _get_llmisvc_url(llmisvc=llmisvc),
        "replicas": _get_llmisvc_replicas(llmisvc=llmisvc),
        "model_uri": _get_llmisvc_model_uri(llmisvc=llmisvc),
        "kueue_integration_stats": _get_llmisvc_kueue_integration_stats(client=client, llmisvc=llmisvc),
        "config_ref_names": _get_llmisvc_config_ref_names(llmisvc=llmisvc),
        "config_ref_pins": _get_llmisvc_config_ref_pins(llmisvc=llmisvc),
        "container_images": _get_llmisvc_container_images(client=client, llmisvc=llmisvc),
        "restart_counts": get_llmisvc_restart_counts(client=client, llmisvc=llmisvc),
    }

    LOGGER.info(event=f"[BASELINE] Captured baseline for '{llmisvc.name}'", baseline=baseline)
    return baseline


def get_llmisvc_restart_counts(client: DynamicClient, llmisvc: LLMInferenceService) -> dict[str, dict[str, int]]:
    """Get container restart counts for all pods associated with an LLMInferenceService."""
    pods = _get_llmisvc_pods(client=client, llmisvc=llmisvc)
    return {
        pod.name: {
            container.name: container.restartCount for container in (pod.instance.status.containerStatuses or [])
        }
        for pod in pods
    }


def _get_llmisvc_pods(client: DynamicClient, llmisvc: LLMInferenceService) -> list[Pod]:
    """Fetch all pods associated with an LLMInferenceService."""
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


def _get_llmisvc_generation(llmisvc: LLMInferenceService) -> int:
    """Get the observed controller generation for an LLMInferenceService."""
    return llmisvc.instance.status.observedGeneration


def _get_llmisvc_url(llmisvc: LLMInferenceService) -> str:
    """Get the serving URL from an LLMInferenceService status."""
    return llmisvc.instance.status.url


def _get_llmisvc_replicas(llmisvc: LLMInferenceService) -> int:
    """Get the configured replica count for an LLMInferenceService."""
    return llmisvc.instance.spec.replicas


def _get_llmisvc_model_uri(llmisvc: LLMInferenceService) -> str:
    """Get the model URI from an LLMInferenceService spec."""
    return llmisvc.instance.spec.model.uri


def _get_llmisvc_config_ref_pins(llmisvc: LLMInferenceService) -> dict[str, str]:
    """Get LLMInferenceService config reference annotations."""
    config_ref_annotation_prefix = "serving.kserve.io/config-llm-"
    annotations = getattr(llmisvc.instance.status, "annotations", None) or {}
    return {key: value for key, value in annotations.items() if key.startswith(config_ref_annotation_prefix) and value}


def _get_llmisvc_config_ref_names(llmisvc: LLMInferenceService) -> list[str]:
    """Get sorted LLMInferenceService config reference names."""
    return sorted(_get_llmisvc_config_ref_pins(llmisvc=llmisvc).values())


def _get_llmisvc_container_images(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> dict[str, dict[str, str]]:
    """Get container images for all pods associated with an LLMInferenceService."""
    pods = _get_llmisvc_pods(client=client, llmisvc=llmisvc)
    return {pod.name: {container.name: container.image for container in pod.instance.spec.containers} for pod in pods}


def _get_llmisvc_kueue_integration_stats(
    client: DynamicClient,
    llmisvc: LLMInferenceService,
) -> dict[str, int]:
    """Get running and gated pod counts for an LLMInferenceService."""
    selector_labels = [f"app.kubernetes.io/name={llmisvc.name}", "kserve.io/component=workload"]
    running, gated = check_gated_pods_and_running_pods(
        labels=selector_labels,
        namespace=llmisvc.namespace,
        admin_client=client,
    )
    return {"running": running, "gated": gated}
