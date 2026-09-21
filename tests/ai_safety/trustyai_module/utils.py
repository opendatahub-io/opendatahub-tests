from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment

from tests.ai_safety.trustyai_module.constants import (
    CONDITION_PROVISIONING_SUCCEEDED,
    CONDITION_READY,
)
from utilities.resources.trustyai_module import TrustyAI


def get_condition_status(trustyai_module: TrustyAI, condition_type: str) -> str | None:
    """Return the status string for a module condition type, if present."""
    conditions = trustyai_module.instance.status.conditions or []
    for condition in conditions:
        if condition.type == condition_type:
            return condition.status
    return None


def validate_trustyai_module_ready(trustyai_module: TrustyAI, timeout: int = 300) -> None:
    """Wait until the TrustyAI module CR reports Ready=True."""
    trustyai_module.wait_for_condition(condition=CONDITION_READY, status="True", timeout=timeout)


def validate_trustyai_module_provisioning_succeeded(trustyai_module: TrustyAI) -> None:
    """Assert the module operator finished applying manifests."""
    status = get_condition_status(
        trustyai_module=trustyai_module,
        condition_type=CONDITION_PROVISIONING_SUCCEEDED,
    )
    assert status == "True", (
        f"Expected {CONDITION_PROVISIONING_SUCCEEDED}=True on TrustyAI/{trustyai_module.name}, got {status!r}"
    )


def validate_trustyai_module_observed_generation(trustyai_module: TrustyAI) -> None:
    """Assert status.observedGeneration tracks the current spec generation."""
    observed_generation = trustyai_module.instance.status.observedGeneration
    metadata_generation = trustyai_module.instance.metadata.generation
    assert observed_generation == metadata_generation, (
        f"Expected observedGeneration={metadata_generation} on TrustyAI/{trustyai_module.name}, "
        f"got {observed_generation}"
    )


def validate_deployment_available(deployment: Deployment, timeout: int = 300) -> None:
    """Wait until a Deployment reports Available=True."""
    deployment.wait_for_condition(condition="Available", status="True", timeout=timeout)


def validate_configmap_exists(configmap: ConfigMap) -> None:
    """Assert a ConfigMap exists in the cluster."""
    assert configmap.exists, f"ConfigMap {configmap.name} does not exist in namespace {configmap.namespace}"
