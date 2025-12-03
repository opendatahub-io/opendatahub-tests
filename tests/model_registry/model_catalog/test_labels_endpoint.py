import pytest
import yaml

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor

from utilities.infra import get_openshift_token
from timeout_sampler import TimeoutSampler

from tests.model_registry.model_catalog.utils import (
    get_labels_from_configmaps,
    get_labels_from_api,
    verify_labels_match,
)

LOGGER = get_logger(name=__name__)


class TestLabelsEndpoint:
    """Test class for the model catalog labels endpoint."""

    @pytest.mark.smoke
    def test_labels_endpoint_default_data(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
    ):
        """
        Smoke test: Validate default labels from ConfigMaps are returned by the endpoint.
        """
        LOGGER.info("Testing labels endpoint with default data")

        # Get expected labels from ConfigMaps
        expected_labels = get_labels_from_configmaps(admin_client=admin_client, namespace=model_registry_namespace)

        # Get labels from API
        api_labels = get_labels_from_api(
            model_catalog_rest_url=model_catalog_rest_url[0], user_token=get_openshift_token()
        )

        # Verify they match
        verify_labels_match(expected_labels=expected_labels, api_labels=api_labels)

    @pytest.mark.sanity
    def test_labels_endpoint_configmap_updates(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
    ):
        """
        Sanity test: Edit the editable ConfigMap and verify changes are reflected in API.
        """

        # Get the editable ConfigMap
        sources_cm = ConfigMap(name="model-catalog-sources", client=admin_client, namespace=model_registry_namespace)

        # Parse current data and add test label
        current_data = yaml.safe_load(sources_cm.instance.data["sources.yaml"])

        new_label = {
            "name": "test-dynamic",
            "displayName": "Dynamic Test Label",
            "description": "A label added during test execution",
        }

        if "labels" not in current_data:
            current_data["labels"] = []
        current_data["labels"].append(new_label)

        # Update ConfigMap temporarily
        patches = {"data": {"sources.yaml": yaml.dump(current_data, default_flow_style=False)}}

        with ResourceEditor(patches={sources_cm: patches}):

            def _check_updated_labels():
                # Get updated expected labels from ConfigMaps
                expected_labels = get_labels_from_configmaps(
                    admin_client=admin_client, namespace=model_registry_namespace
                )

                # Get labels from API
                api_labels = get_labels_from_api(
                    model_catalog_rest_url=model_catalog_rest_url[0], user_token=get_openshift_token()
                )

                # Verify they match (including the new label)
                verify_labels_match(expected_labels=expected_labels, api_labels=api_labels)

            sampler = TimeoutSampler(wait_timeout=60, sleep=5, func=_check_updated_labels)
            next(iter(sampler))
