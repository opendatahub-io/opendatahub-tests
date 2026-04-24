"""
Tests to verify that all Model Registry component images (operator, instance, and async job
container images) meet the requirements:
1. Images are hosted in registry.redhat.io
2. Images use sha256 digest instead of tags
3. Images are listed in the CSV's relatedImages section
"""

from typing import Self

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config

from tests.model_registry.constants import MR_INSTANCE_NAME, MR_OPERATOR_NAME, MR_POSTGRES_DEPLOYMENT_NAME_STR
from tests.model_registry.image_validation.utils import validate_images
from utilities.constants import Labels
from utilities.general import validate_image_format

LOGGER = structlog.get_logger(name=__name__)
pytestmark = [pytest.mark.downstream_only, pytest.mark.skip_must_gather]


class TestAIHubResourcesImages:
    @pytest.mark.parametrize(
        "resource_pods",
        [
            pytest.param(
                {"namespace": py_config["model_registry_namespace"], "label_selector": "component=model-catalog"},
                marks=pytest.mark.tier1,
                id="test_model_catalog_pods_images",
            ),
            pytest.param(
                {
                    "namespace": py_config["applications_namespace"],
                    "label_selector": f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
                },
                marks=pytest.mark.tier1,
                id="test_model_registry_operator_pods_images",
            ),
        ],
        indirect=True,
    )
    def test_verify_pod_images(
        self: Self,
        admin_client: DynamicClient,
        resource_pods: list[Pod],
        related_images_refs: set[str],
    ):
        validate_images(pods_to_validate=resource_pods, related_images_refs=related_images_refs)

    @pytest.mark.tier1
    def test_verify_async_job_image(
        self: Self,
        async_upload_image: str,
        related_images_refs: set[str],
    ):
        """
        Given the async upload job image from the model-registry-operator-parameters ConfigMap
        When validating the image reference
        Then verify it uses registry.redhat.io, sha256 digest, and is listed in CSV relatedImages
        """
        LOGGER.info(f"Validating async upload job image: {async_upload_image}")
        validation_errors = []

        is_valid, error_msg = validate_image_format(image=async_upload_image)
        if not is_valid:
            validation_errors.append(f"Async upload job image validation failed: {error_msg}")

        if async_upload_image not in related_images_refs:
            validation_errors.append(f"Async upload job image {async_upload_image} is not in CSV relatedImages")

        if validation_errors:
            pytest.fail("\n".join(validation_errors))


@pytest.mark.parametrize(
    "model_registry_metadata_db_resources, model_registry_instance, model_registry_instance_pods_by_label",
    [
        pytest.param({}, {}, {"label_selectors": [f"app={MR_INSTANCE_NAME}"]}, marks=pytest.mark.smoke),
        pytest.param(
            {"db_name": "default"},
            {"db_name": "default"},
            {
                "label_selectors": [
                    f"app.kubernetes.io/name={MR_INSTANCE_NAME}",
                    f"app.kubernetes.io/name={MR_POSTGRES_DEPLOYMENT_NAME_STR}",
                ]
            },
            marks=pytest.mark.tier2,
        ),
    ],
    indirect=True,
)
@pytest.mark.usefixtures(
    "updated_dsc_component_state_scope_session",
    "model_registry_metadata_db_resources",
    "model_registry_instance",
    "model_registry_instance_pods_by_label",
)
class TestModelRegistryImages:
    def test_verify_model_registry_pod_images(
        self: Self,
        admin_client: DynamicClient,
        model_registry_instance_pods_by_label: list[Pod],
        related_images_refs: set[str],
    ):
        validate_images(pods_to_validate=model_registry_instance_pods_by_label, related_images_refs=related_images_refs)
