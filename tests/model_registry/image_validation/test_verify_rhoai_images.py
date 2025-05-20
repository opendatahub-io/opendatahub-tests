import pytest
from typing import Self
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config

from utilities.constants import DscComponents, Labels
from utilities.general import get_csv_related_images, get_pod_images, validate_image_format
from tests.model_registry.constants import MR_OPERATOR_NAME, MR_INSTANCE_NAME, MR_NAMESPACE
from ocp_resources.model_registry import ModelRegistry

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_dsc_component_state_scope_class",
    [
        {
            "component_patch": {
                DscComponents.MODELREGISTRY: {
                    "managementState": DscComponents.ManagementState.MANAGED,
                    "registriesNamespace": MR_NAMESPACE,
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

        # Get operator pod from applications namespace
        operator_pods = list(
            Pod.get(
                dyn_client=admin_client,
                namespace=py_config["applications_namespace"],
                label_selector=f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
            )
        )
        if not operator_pods:
            pytest.fail(f"Operator pod not found in namespace {py_config['applications_namespace']}")
        if len(operator_pods) > 1:
            LOGGER.warning("Multiple operator pods found, using the first one")
        operator_pod = operator_pods[0]

        # Get MR instance pod(s)
        instance_pods = list(
            Pod.get(
                dyn_client=admin_client,
                namespace=MR_NAMESPACE,
                label_selector=f"app={MR_INSTANCE_NAME}",
            )
        )
        if not instance_pods:
            pytest.fail(f"Instance pod not found in namespace {MR_NAMESPACE}")
        if len(instance_pods) > 1:
            LOGGER.warning("Multiple instance pods found, using the first one")
        instance_pod = instance_pods[0]

        # Collect all validation errors
        validation_errors = []

        # Verify operator pod images
        operator_images = get_pod_images(pod=operator_pod)
        for image in operator_images:
            # Validate image format
            is_valid, error_msg = validate_image_format(image=image)
            if not is_valid:
                validation_errors.append(f"Operator pod image validation failed: {error_msg}")

            # Check if image is in relatedImages
            if image not in related_image_refs:
                validation_errors.append(f"Operator pod image {image} is not listed in CSV's relatedImages")

        # Verify instance pods images
        pod_images = get_pod_images(pod=instance_pod)
        for image in pod_images:
            # Validate image format
            is_valid, error_msg = validate_image_format(image=image)
            if not is_valid:
                validation_errors.append(f"Pod {instance_pod.name} image validation failed: {error_msg}")

            # If it's a sidecar image defined correctly (comes from registry and uses sha256 digest)
            # we don't need to check that it is in our relatedImages
            if "openshift-service-mesh" in image:
                LOGGER.warning(f"Skipping image {image} as it is a service mesh sidecar image")
                continue
            # Check if image is in relatedImages
            if image not in related_image_refs:
                validation_errors.append(f"Pod {instance_pod.name} image {image} is not listed in CSV's relatedImages")

        if validation_errors:
            pytest.fail("\n".join(validation_errors))
