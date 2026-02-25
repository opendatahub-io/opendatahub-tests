"""
Tests to verify that serving runtime component images meet the requirements:
1. Images are hosted in registry.redhat.io
2. Images use sha256 digest instead of tags
3. Images are listed in the CSV's relatedImages section

For each runtime template we create ServingRuntime + InferenceService, wait for pod(s),
then validate the pod's container images against the cluster CSV (relatedImages) at runtime.
No hardcoded image SHAs—validation uses whatever CSV is installed (e.g. rhods-operator.3.3.0).
"""

from typing import Self, Set

import pytest
from simple_logger.logger import get_logger

from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.image_validation.conftest import RUNTIME_CONFIGS
from tests.model_registry.image_validation.utils import validate_images

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.downstream_only,
    pytest.mark.skip_must_gather,
    pytest.mark.smoke,
]


@pytest.mark.parametrize("serving_runtime_pods_for_runtime", RUNTIME_CONFIGS, indirect=True)
class TestServingRuntimeImagesPerTemplate:
    """
    For each runtime template: create ServingRuntime + InferenceService, wait for pod(s),
    validate pod images (registry.redhat.io, sha256, CSV), output runtimename : passed, then teardown.
    """

    def test_verify_serving_runtime_pod_images_from_template(
        self: Self,
        serving_runtime_pods_for_runtime: tuple[list[Pod], str],
        related_images_refs: Set[str],
    ) -> None:
        """
        For the parametrized runtime: create SR+ISVC from template, validate pod images, report name : passed.
        """
        pods, runtime_name = serving_runtime_pods_for_runtime
        validate_images(
            pods_to_validate=pods,
            related_images_refs=related_images_refs,
        )
        LOGGER.info(f"{runtime_name} : passed")
