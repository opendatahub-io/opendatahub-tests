import pytest
from typing import Self, Set
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient

from utilities.general import (
    validate_container_images,
)
from ocp_resources.pod import Pod

LOGGER = get_logger(name=__name__)


@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_class",
    "is_model_registry_oauth",
    "mysql_metadata_resources",
    "model_registry_instance_mysql",
)
@pytest.mark.downstream_only
class TestModelRegistryImages:
    """
    Tests to verify that all Model Registry component images (operator and instance container images)
    meet the requirements:
    1. Images are hosted in registry.redhat.io
    2. Images use sha256 digest instead of tags
    3. Images are listed in the CSV's relatedImages section
    """

    @pytest.mark.smoke
    def test_verify_model_registry_images(
        self: Self,
        admin_client: DynamicClient,
        model_registry_operator_pod: Pod,
        model_registry_instance_pod: Pod,
        related_images_refs: Set[str],
    ):
        validation_errors = []
        for pod in [model_registry_operator_pod, model_registry_instance_pod]:
            validation_errors.extend(
                validate_container_images(
                    pod=pod, valid_image_refs=related_images_refs, skip_patterns=["openshift-service-mesh"]
                )
            )

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
