"""
Tests to verify that all Model Registry component images (operator and instance container images)
meet the requirements:
1. Images are hosted in registry.redhat.io
2. Images use sha256 digest instead of tags
3. Images are listed in the CSV's relatedImages section
"""

import pytest
from typing import Self, Set
from simple_logger.logger import get_logger
from kubernetes.dynamic import DynamicClient

from tests.model_registry.constants import MR_INSTANCE_NAME, MR_POSTGRES_DEPLOYMENT_NAME_STR, MR_OPERATOR_NAME
from tests.model_registry.image_validation.utils import validate_images
from utilities.constants import Labels
from ocp_resources.pod import Pod
from pytest_testconfig import config as py_config

LOGGER = get_logger(name=__name__)
pytestmark = [pytest.mark.downstream_only, pytest.mark.skip_must_gather]


class TestAIHubResourcesImages:
    @pytest.mark.parametrize(
        "resource_pods",
        [
            pytest.param(
                {"namespace": py_config["model_registry_namespace"], "label_selector": "component=model-catalog"},
                marks=pytest.mark.sanity,
                id="test_model_catalog_pods_images",
            ),
            pytest.param(
                {
                    "namespace": py_config["applications_namespace"],
                    "label_selector": f"{Labels.OpenDataHubIo.NAME}={MR_OPERATOR_NAME}",
                },
                marks=pytest.mark.sanity,
                id="test_model_registry_operator_pods_images",
            ),
        ],
        indirect=True,
    )
    def test_verify_pod_images(
        self: Self,
        admin_client: DynamicClient,
        resource_pods: list[Pod],
        related_images_refs: Set[str],
    ):
        validate_images(pods_to_validate=resource_pods, related_images_refs=related_images_refs)


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
            marks=pytest.mark.tier1,
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
        related_images_refs: Set[str],
    ):
        validate_images(pods_to_validate=model_registry_instance_pods_by_label, related_images_refs=related_images_refs)
