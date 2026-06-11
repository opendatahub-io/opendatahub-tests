"""Tests for fast-1 and fast-2 vLLM ServingRuntime template overlays.

Validates that the odh-model-controller creates correctly structured
fast template overlays when fast image SHAs differ from stable.
"""

from typing import Any, Self

import pytest
from ocp_resources.template import Template

from tests.model_serving.model_runtime.vllm.fast_templates.constant import (
    FAST_TEMPLATE_PARAMS,
    SUPPORT_STATUS_ANNOTATION,
    UNSUPPORTED_VALUE,
)

pytestmark = [pytest.mark.tier1, pytest.mark.downstream_only, pytest.mark.skip_must_gather]


@pytest.mark.parametrize("fast_template_config", FAST_TEMPLATE_PARAMS, indirect=True)
class TestFastTemplateOverlays:
    """Validate fast-1 and fast-2 vLLM ServingRuntime template overlays."""

    def test_fast_template_has_unsupported_annotation(
        self: Self,
        fast_template: Template,
    ) -> None:
        """Given a fast vLLM Template on the cluster
        When its annotations are inspected
        Then it carries opendatahub.io/support-status: unsupported
        """
        annotations = fast_template.instance.metadata.annotations or {}
        assert annotations.get(SUPPORT_STATUS_ANNOTATION) == UNSUPPORTED_VALUE, (
            f"Fast template {fast_template.name} missing {SUPPORT_STATUS_ANNOTATION}: {UNSUPPORTED_VALUE} annotation"
        )

    def test_embedded_runtime_name_is_suffixed(
        self: Self,
        fast_template_config: dict[str, str],
        fast_embedded_runtime: dict[str, Any],
        stable_embedded_runtime: dict[str, Any],
    ) -> None:
        """Given fast and stable vLLM Templates on the cluster
        When the embedded ServingRuntime name is inspected
        Then it equals the stable runtime name with the fast suffix appended
        """
        stable_name = stable_embedded_runtime["metadata"]["name"]
        fast_name = fast_embedded_runtime["metadata"]["name"]
        suffix = fast_template_config["suffix"]
        expected_name = f"{stable_name}-{suffix}"
        assert fast_name == expected_name, (
            f"Embedded runtime name {fast_name!r} does not match expected {expected_name!r}"
        )

    def test_embedded_runtime_has_unsupported_annotation(
        self: Self,
        fast_embedded_runtime: dict[str, Any],
    ) -> None:
        """Given a fast vLLM Template on the cluster
        When the embedded ServingRuntime annotations are inspected
        Then it carries opendatahub.io/support-status: unsupported
        """
        annotations = fast_embedded_runtime.get("metadata", {}).get("annotations", {})
        assert annotations.get(SUPPORT_STATUS_ANNOTATION) == UNSUPPORTED_VALUE, (
            f"Embedded runtime missing {SUPPORT_STATUS_ANNOTATION}: {UNSUPPORTED_VALUE} annotation"
        )

    def test_fast_image_differs_from_stable(
        self: Self,
        fast_embedded_runtime: dict[str, Any],
        stable_embedded_runtime: dict[str, Any],
    ) -> None:
        """Given fast and stable vLLM Templates on the cluster
        When their container images are compared
        Then the fast template image differs from the stable template image
        """
        stable_containers = stable_embedded_runtime.get("spec", {}).get("containers", [])
        fast_containers = fast_embedded_runtime.get("spec", {}).get("containers", [])
        assert stable_containers, "Stable template has no containers"
        assert fast_containers, "Fast template has no containers"
        stable_image = stable_containers[0]["image"]
        fast_image = fast_containers[0]["image"]
        assert fast_image != stable_image, f"Fast template image should differ from stable but both are {fast_image!r}"
