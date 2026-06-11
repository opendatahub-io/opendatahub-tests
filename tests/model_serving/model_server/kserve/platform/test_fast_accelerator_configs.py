"""Tests for fast-1/fast-2 accelerator LLMInferenceServiceConfig overlays."""

import pytest
import structlog

from tests.model_serving.model_server.kserve.platform.utils import (
    FAST_SUFFIXES,
    extract_container_images,
    stable_name_for_fast,
)
from utilities.resources.llm_inference_service_config import LLMInferenceServiceConfig

LOGGER = structlog.get_logger(name=__name__)

SUPPORT_STATUS_ANNOTATION = "opendatahub.io/support-status"
WELL_KNOWN_CONFIG_ANNOTATION = "serving.kserve.io/well-known-config"
CONFIG_TYPE_LABEL = "opendatahub.io/config-type"

EXPECTED_ACCELERATOR_TYPES = ("nvidia-cuda", "amd-rocm", "intel-gaudi", "ibm-spyre")

pytestmark = [pytest.mark.tier1]


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
        fast_names = [config.name for config in fast_configs]
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
        fast_names = [config.name for config in fast_configs]
        for accel in EXPECTED_ACCELERATOR_TYPES:
            accel_names = [name for name in fast_names if accel in name]
            if not accel_names:
                continue
            for suffix in FAST_SUFFIXES:
                matching = [name for name in accel_names if name.endswith(suffix)]
                assert matching, f"No {suffix} config found for accelerator '{accel}'. Found: {accel_names}"

    @pytest.mark.parametrize(
        "metadata_key, metadata_field, expected_value",
        [
            pytest.param(
                SUPPORT_STATUS_ANNOTATION,
                "annotations",
                "unsupported",
                id="test_support_status_annotation",
            ),
            pytest.param(
                WELL_KNOWN_CONFIG_ANNOTATION,
                "annotations",
                "true",
                id="test_well_known_config_annotation",
            ),
            pytest.param(
                CONFIG_TYPE_LABEL,
                "labels",
                "accelerator",
                id="test_config_type_label",
            ),
        ],
    )
    def test_fast_configs_metadata(
        self,
        fast_configs: list[LLMInferenceServiceConfig],
        metadata_key: str,
        metadata_field: str,
        expected_value: str,
    ) -> None:
        """Given fast LLMInferenceServiceConfig resources exist,
        When checking their metadata (annotations or labels),
        Then each carries the expected key-value pair.
        """
        for config in fast_configs:
            metadata = getattr(config.instance.metadata, metadata_field, None) or {}
            assert metadata.get(metadata_key) == expected_value, (
                f"Config '{config.name}': expected '{metadata_key}={expected_value}' "
                f"in {metadata_field}, got '{metadata.get(metadata_key)}'"
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
            stable_name = stable_name_for_fast(fast_name=fast_config.name)
            stable_config = stable_configs_by_name.get(stable_name)
            if stable_config is None:
                LOGGER.warning(f"No stable counterpart '{stable_name}' for '{fast_config.name}'")
                continue

            fast_images = extract_container_images(obj=fast_config.instance.to_dict().get("spec", {}))
            stable_images = extract_container_images(obj=stable_config.instance.to_dict().get("spec", {}))

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
        fast = [config for config in all_llm_configs if config.name.endswith(FAST_SUFFIXES)]
        if fast:
            pytest.fail(
                f"Fast configs exist (SHAs differ from stable) -- negative test not applicable. "
                f"Found: {[config.name for config in fast]}"
            )
        stable = [config for config in all_llm_configs if not config.name.endswith(FAST_SUFFIXES)]
        assert stable, "Expected at least some stable LLMInferenceServiceConfig resources on the cluster"
