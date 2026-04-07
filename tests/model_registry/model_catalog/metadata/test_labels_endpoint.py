from typing import Any, Literal

import pytest
import structlog
from kubernetes.dynamic import DynamicClient
from timeout_sampler import TimeoutSampler

from tests.model_registry.model_catalog.metadata.utils import (
    get_labels_from_api,
    get_labels_from_configmaps,
    verify_labels_match,
)
from utilities.infra import get_openshift_token

LOGGER = structlog.get_logger(name=__name__)


class TestLabelsEndpoint:
    """Test class for the model catalog labels endpoint."""

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "expected_labels_by_asset_type, asset_type",
        [
            pytest.param("models", "models", id="test_models"),
            pytest.param("mcp_servers", "mcp_servers", id="test_mcp_servers"),
        ],
        indirect=["expected_labels_by_asset_type"],
    )
    def test_labels_endpoint_default_data(
        self,
        model_catalog_rest_url: list[str],
        expected_labels_by_asset_type: list[dict[str, Any]],
        asset_type: Literal["models", "mcp_servers"],
    ):
        """
        Smoke test: Validate default labels from ConfigMaps are returned by the endpoint.
        """
        LOGGER.info(f"Testing labels endpoint with default data for asset type: {asset_type}")
        api_labels = get_labels_from_api(
            model_catalog_rest_url=model_catalog_rest_url[0], user_token=get_openshift_token(), asset_type=asset_type
        )
        verify_labels_match(expected_labels=expected_labels_by_asset_type, api_labels=api_labels)

    @pytest.mark.smoke
    def test_labels_endpoint_default_no_asset_type(
        self,
        model_catalog_rest_url: list[str],
    ):
        """
        Smoke test: Validate that /labels with no assetType param returns only model labels.
        """
        LOGGER.info("Testing labels endpoint with no asset type parameter")
        api_labels = get_labels_from_api(
            model_catalog_rest_url=model_catalog_rest_url[0], user_token=get_openshift_token()
        )
        assert api_labels, "Expected non-empty labels response"
        mismatches = [
            f"Label '{label.get('name')}' has assetType '{label.get('assetType')}', expected 'models'"
            for label in api_labels
            if label.get("assetType") != "models"
        ]
        assert not mismatches, "\n".join(mismatches)

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
            all_expected_labels = get_labels_from_configmaps(
                admin_client=admin_client, namespace=model_registry_namespace
            )

            token = get_openshift_token()
            url = model_catalog_rest_url[0]

            # Split expected labels by asset type
            mcp_expected_labels = [label for label in all_expected_labels if label.get("assetType") == "mcp_servers"]
            model_expected_labels = [label for label in all_expected_labels if label.get("assetType") == "models"]

            # Verify default /labels returns only model labels (no MCP cross-contamination)
            api_labels = get_labels_from_api(model_catalog_rest_url=url, user_token=token)
            verify_labels_match(expected_labels=model_expected_labels, api_labels=api_labels)

            # Verify assetType=mcp_servers returns only MCP labels (no model cross-contamination)
            mcp_api_labels = get_labels_from_api(model_catalog_rest_url=url, user_token=token, asset_type="mcp_servers")
            verify_labels_match(expected_labels=mcp_expected_labels, api_labels=mcp_api_labels)

        sampler = TimeoutSampler(wait_timeout=60, sleep=5, func=_check_updated_labels)
        for _ in sampler:
            break
