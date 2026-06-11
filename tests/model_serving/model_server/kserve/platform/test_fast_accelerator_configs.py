"""Tests for fast-1/fast-2 accelerator LLMInferenceServiceConfig overlays."""

from typing import Any

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_service_version import ClusterServiceVersion
from pytest_testconfig import config as py_config

from utilities.llm_inference_service_config import LLMInferenceServiceConfig

LOGGER = structlog.get_logger(name=__name__)

SUPPORT_STATUS_ANNOTATION = "opendatahub.io/support-status"
WELL_KNOWN_CONFIG_ANNOTATION = "serving.kserve.io/well-known-config"
CONFIG_TYPE_LABEL = "opendatahub.io/config-type"

FAST_SUFFIXES = ("-fast-1", "-fast-2")

EXPECTED_ACCELERATOR_TYPES = ("nvidia-cuda", "amd-rocm", "intel-gaudi", "ibm-spyre")

pytestmark = [pytest.mark.tier1]


def _get_rhoai_version_prefix(client: DynamicClient) -> str | None:
    """Resolve the RHOAI version prefix from the operator CSV.

    Args:
        client: Kubernetes dynamic client.

    Returns:
        Version prefix string (e.g. "v3-4-0"), or None if no RHOAI CSV is found.
    """
    for csv in ClusterServiceVersion.get(client=client, namespace="redhat-ods-operator"):
        if csv.name.startswith("rhods-operator") and csv.status == csv.Status.SUCCEEDED:
            version = csv.instance.spec.version
            return f"v{version.replace('.', '-')}"
    return None


def _extract_container_images(obj: Any) -> list[str]:
    """Recursively extract container image strings from a resource spec dict.

    Args:
        obj: A dict, list, or scalar from a deserialized Kubernetes resource spec.

    Returns:
        List of image strings found under any "image" key.
    """
    images: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "image" and isinstance(value, str):
                images.append(value)
            else:
                images.extend(_extract_container_images(obj=value))
    elif isinstance(obj, list):
        for item in obj:
            images.extend(_extract_container_images(obj=item))
    return images


def _stable_name_for_fast(fast_name: str) -> str:
    """Derive the stable config name by stripping the fast suffix.

    Args:
        fast_name: Name of a fast overlay config (ending in -fast-1 or -fast-2).

    Returns:
        The corresponding stable config name.
    """
    for suffix in FAST_SUFFIXES:
        if fast_name.endswith(suffix):
            return fast_name[: -len(suffix)]
    raise ValueError(f"Not a fast config name: {fast_name}")


@pytest.fixture(scope="module")
def all_llm_configs(admin_client: DynamicClient) -> list[LLMInferenceServiceConfig]:
    """All LLMInferenceServiceConfig resources in the applications namespace."""
    namespace = py_config["applications_namespace"]
    configs = list(LLMInferenceServiceConfig.get(client=admin_client, namespace=namespace))
    LOGGER.info(f"Found {len(configs)} LLMInferenceServiceConfig resources in {namespace}")
    for config in configs:
        LOGGER.info(f"  - {config.name}")
    return configs


@pytest.fixture(scope="module")
def fast_configs(all_llm_configs: list[LLMInferenceServiceConfig]) -> list[LLMInferenceServiceConfig]:
    """Fast overlay configs (-fast-1, -fast-2), skips if none exist."""
    configs = [c for c in all_llm_configs if c.name.endswith(FAST_SUFFIXES)]
    if not configs:
        pytest.skip("No fast LLMInferenceServiceConfig resources found (fast image SHAs may match stable)")
    LOGGER.info(f"Found {len(configs)} fast configs: {[c.name for c in configs]}")
    return configs


@pytest.fixture(scope="module")
def stable_configs_by_name(
    all_llm_configs: list[LLMInferenceServiceConfig],
) -> dict[str, LLMInferenceServiceConfig]:
    """Map of stable config name to resource (non-fast configs)."""
    return {c.name: c for c in all_llm_configs if not c.name.endswith(FAST_SUFFIXES)}


@pytest.fixture(scope="module")
def version_prefix(admin_client: DynamicClient) -> str:
    """RHOAI version prefix for config names."""
    prefix = _get_rhoai_version_prefix(client=admin_client)
    if prefix is None:
        pytest.skip("RHOAI CSV not found — cannot verify version-prefixed names")
    return prefix


class TestFastAcceleratorConfigs:
    """Validate fast-1/fast-2 LLMInferenceServiceConfig overlays for accelerators.

    Verifies that the KServe module operator correctly creates fast overlay
    LLMInferenceServiceConfig resources when fast image SHAs differ from stable.
    """

    def test_fast_configs_exist_for_accelerator_types(
        self,
        fast_configs: list[LLMInferenceServiceConfig],
    ) -> None:
        """Given fast image SHAs differ from stable,
        When listing fast LLMInferenceServiceConfig resources,
        Then resources exist for each expected accelerator type.
        """
        fast_names = [c.name for c in fast_configs]
        missing = [accel for accel in EXPECTED_ACCELERATOR_TYPES if not any(accel in name for name in fast_names)]
        assert not missing, f"Fast configs missing for accelerator types: {missing}. Found fast configs: {fast_names}"

    def test_fast_configs_have_fast_1_and_fast_2(
        self,
        fast_configs: list[LLMInferenceServiceConfig],
    ) -> None:
        """Given fast image SHAs differ from stable,
        When listing fast LLMInferenceServiceConfig resources,
        Then both fast-1 and fast-2 variants exist for each accelerator.
        """
        fast_names = [c.name for c in fast_configs]
        for accel in EXPECTED_ACCELERATOR_TYPES:
            accel_names = [name for name in fast_names if accel in name]
            if not accel_names:
                continue
            for suffix in FAST_SUFFIXES:
                matching = [name for name in accel_names if name.endswith(suffix)]
                assert matching, f"No {suffix} config found for accelerator '{accel}'. Found: {accel_names}"

    def test_fast_configs_support_status_annotation(
        self,
        fast_configs: list[LLMInferenceServiceConfig],
    ) -> None:
        """Given fast LLMInferenceServiceConfig resources exist,
        When checking annotations,
        Then each carries opendatahub.io/support-status: unsupported.
        """
        for config in fast_configs:
            annotations = config.instance.metadata.annotations or {}
            assert annotations.get(SUPPORT_STATUS_ANNOTATION) == "unsupported", (
                f"Config '{config.name}': expected '{SUPPORT_STATUS_ANNOTATION}=unsupported', "
                f"got '{annotations.get(SUPPORT_STATUS_ANNOTATION)}'"
            )

    def test_fast_configs_well_known_config_annotation(
        self,
        fast_configs: list[LLMInferenceServiceConfig],
    ) -> None:
        """Given fast LLMInferenceServiceConfig resources exist,
        When checking annotations,
        Then each carries serving.kserve.io/well-known-config: "true".
        """
        for config in fast_configs:
            annotations = config.instance.metadata.annotations or {}
            assert annotations.get(WELL_KNOWN_CONFIG_ANNOTATION) == "true", (
                f"Config '{config.name}': expected '{WELL_KNOWN_CONFIG_ANNOTATION}=true', "
                f"got '{annotations.get(WELL_KNOWN_CONFIG_ANNOTATION)}'"
            )

    def test_fast_configs_config_type_label(
        self,
        fast_configs: list[LLMInferenceServiceConfig],
    ) -> None:
        """Given fast LLMInferenceServiceConfig resources exist,
        When checking labels,
        Then each carries opendatahub.io/config-type: accelerator.
        """
        for config in fast_configs:
            labels = config.instance.metadata.labels or {}
            assert labels.get(CONFIG_TYPE_LABEL) == "accelerator", (
                f"Config '{config.name}': expected '{CONFIG_TYPE_LABEL}=accelerator', "
                f"got '{labels.get(CONFIG_TYPE_LABEL)}'"
            )

    def test_fast_configs_version_prefixed_names(
        self,
        fast_configs: list[LLMInferenceServiceConfig],
        version_prefix: str,
    ) -> None:
        """Given fast LLMInferenceServiceConfig resources exist,
        When checking their names,
        Then each has a version-prefixed name matching the RHOAI operator version.
        """
        for config in fast_configs:
            assert config.name.startswith(version_prefix), (
                f"Config '{config.name}' does not start with version prefix '{version_prefix}'"
            )

    def test_fast_config_images_differ_from_stable(
        self,
        fast_configs: list[LLMInferenceServiceConfig],
        stable_configs_by_name: dict[str, LLMInferenceServiceConfig],
    ) -> None:
        """Given fast LLMInferenceServiceConfig resources exist,
        When comparing container images to their stable counterparts,
        Then fast images differ from stable images.
        """
        compared = 0
        for fast_config in fast_configs:
            stable_name = _stable_name_for_fast(fast_name=fast_config.name)
            stable_config = stable_configs_by_name.get(stable_name)
            if stable_config is None:
                LOGGER.warning(f"No stable counterpart '{stable_name}' for '{fast_config.name}'")
                continue

            fast_images = _extract_container_images(obj=fast_config.instance.to_dict().get("spec", {}))
            stable_images = _extract_container_images(obj=stable_config.instance.to_dict().get("spec", {}))

            assert fast_images, f"No container images found in fast config '{fast_config.name}'"
            assert stable_images, f"No container images found in stable config '{stable_name}'"
            assert fast_images != stable_images, (
                f"Fast config '{fast_config.name}' images should differ from "
                f"stable '{stable_name}' but both have: {fast_images}"
            )
            compared += 1

        assert compared > 0, "No fast/stable config pairs could be compared"


class TestNoFastConfigsWhenSHAsMatch:
    """Validate that no fast configs exist when fast image SHAs match stable."""

    def test_no_fast_configs_when_shas_match(
        self,
        all_llm_configs: list[LLMInferenceServiceConfig],
    ) -> None:
        """Given fast image SHAs match stable,
        When listing LLMInferenceServiceConfig resources,
        Then no fast-1 or fast-2 resources exist.
        """
        fast = [c for c in all_llm_configs if c.name.endswith(FAST_SUFFIXES)]
        if fast:
            pytest.skip("Fast configs exist (SHAs differ from stable) — negative test not applicable")
        stable = [c for c in all_llm_configs if not c.name.endswith(FAST_SUFFIXES)]
        assert stable, "Expected at least some stable LLMInferenceServiceConfig resources on the cluster"
