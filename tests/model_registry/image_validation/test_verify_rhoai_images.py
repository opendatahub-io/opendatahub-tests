import pytest
from typing import Self
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from pytest_testconfig import config as py_config
from timeout_sampler import TimeoutSampler, TimeoutExpiredError

from utilities.constants import DscComponents, Labels
from utilities.general import (
    get_csv_related_images,
    get_pods_by_labels,
    validate_container_images,
)
from tests.model_registry.constants import MR_OPERATOR_NAME, MR_INSTANCE_NAME
from ocp_resources.model_registry import ModelRegistry

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        {
            "component_patch": {
                DscComponents.MODELREGISTRY: {
                    "managementState": DscComponents.ManagementState.MANAGED,
                    "registriesNamespace": py_config["model_registry_namespace"],
                }
            }
        }
    ],
    indirect=True,
)
@pytest.mark.usefixtures("updated_dsc_component_state_scope_class")
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
        self: Self, admin_client: DynamicClient, model_registry_instance: ModelRegistry
    ):
        # Get related images from CSV
        related_images = get_csv_related_images(admin_client=admin_client)
        related_image_refs = {img["image"] for img in related_images}

        # Get operator pod
        operator_pod = None
        try:
            for operator_pod_list in TimeoutSampler(
                wait_timeout=60,
                sleep=5,
                func=get_pods_by_labels,
                admin_client=admin_client,
                namespace=py_config["applications_namespace"],
                label_selector=f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
            ):
                if len(operator_pod_list) == 1:
                    operator_pod = operator_pod_list[0]
                    break
        except TimeoutExpiredError:
            pytest.fail("Failed to find exactly one operator pod after 60 seconds")

        # Get instance pod
        instance_pod = None
        try:
            for instance_pod_list in TimeoutSampler(
                wait_timeout=60,
                sleep=5,
                func=get_pods_by_labels,
                admin_client=admin_client,
                namespace=py_config["model_registry_namespace"],
                label_selector=f"app={MR_INSTANCE_NAME}",
            ):
                if len(instance_pod_list) == 1:
                    instance_pod = instance_pod_list[0]
                    break
        except TimeoutExpiredError:
            pytest.fail("Failed to find exactly one instance pod after 60 seconds")

        # Validate images in both pods
        validation_errors = []
        validation_errors.extend(
            validate_container_images(
                pod=operator_pod,
                valid_image_refs=related_image_refs,
            )
        )
        validation_errors.extend(
            validate_container_images(
                pod=instance_pod,
                valid_image_refs=related_image_refs,
                skip_patterns=["openshift-service-mesh"],
            )
        )

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
