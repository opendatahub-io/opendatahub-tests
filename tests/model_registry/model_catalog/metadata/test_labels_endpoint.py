from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_registry.model_catalog.metadata.utils import (
    get_labels_from_api,
    get_labels_from_configmaps,
    verify_labels_match,
)
from utilities.infra import get_openshift_token

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

    @pytest.mark.tier1
    def test_labels_endpoint_configmap_updates(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        labels_configmap_patch: dict[str, Any],
    ):
        """
        Sanity test: Edit the editable ConfigMap and verify changes are reflected in API.
        """
        _ = labels_configmap_patch

        def _check_updated_labels():
            # Get updated expected labels from ConfigMaps
            expected_labels = get_labels_from_configmaps(admin_client=admin_client, namespace=model_registry_namespace)

            # Get labels from API
            api_labels = get_labels_from_api(
                model_catalog_rest_url=model_catalog_rest_url[0], user_token=get_openshift_token()
            )

            # Verify they match (including the new label)
            verify_labels_match(expected_labels=expected_labels, api_labels=api_labels)

        sampler = TimeoutSampler(wait_timeout=60, sleep=5, func=_check_updated_labels)
        for _ in sampler:
            break
