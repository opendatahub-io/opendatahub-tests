"""
Test suite for verifying RBAC permissions for Model Catalog ConfigMaps.
"""

import pytest
from kubernetes.client.rest import ApiException
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import get_client
from simple_logger.logger import get_logger

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG, DEFAULT_MODEL_CATALOG_CM

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


@pytest.mark.skip_must_gather
class TestCatalogRBAC:
    """Test suite for catalog ConfigMap RBAC"""

    @pytest.mark.pre_upgrade
    @pytest.mark.post_upgrade
    @pytest.mark.install
    @pytest.mark.parametrize(
        "configmap_name",
        [
            pytest.param(DEFAULT_MODEL_CATALOG_CM, marks=pytest.mark.smoke),
            pytest.param(DEFAULT_CUSTOM_MODEL_CATALOG, marks=pytest.mark.tier1),
        ],
    )
    def test_admin_can_read_catalog_configmaps(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        configmap_name: str,
    ):
        """
        RHOAIENG-41850: Verify that admin users can read both catalog ConfigMaps.

        Admins should have:
        - get/watch on model-catalog-default-sources (read-only)
        - get/watch/update/patch on model-catalog-sources (read/write)

        Note: Admin write access to model-catalog-sources is already tested by existing tests
        (test_custom_model_catalog.py, test_catalog_source_merge.py) which use admin_client
        to successfully update ConfigMaps via ResourceEditor.
        """
        catalog_cm = ConfigMap(
            name=configmap_name,
            namespace=model_registry_namespace,
            client=admin_client,
        )

        assert catalog_cm.exists, f"ConfigMap '{configmap_name}' not found in namespace '{model_registry_namespace}'"

        data = catalog_cm.instance.data
        assert data is not None, f"Admin should be able to read ConfigMap '{configmap_name}' data"

        sources_yaml = data.get("sources.yaml")
        assert sources_yaml is not None, f"ConfigMap '{configmap_name}' should contain 'sources.yaml' key"

        LOGGER.info(f"Admin successfully read ConfigMap '{configmap_name}'")

    @pytest.mark.parametrize(
        "configmap_name",
        [
            pytest.param(DEFAULT_MODEL_CATALOG_CM, marks=pytest.mark.tier1),
            pytest.param(DEFAULT_CUSTOM_MODEL_CATALOG, marks=pytest.mark.tier1),
        ],
    )
    def test_non_admin_cannot_access_catalog_configmaps(
        self,
        is_byoidc: bool,
        model_registry_namespace: str,
        user_credentials_rbac: dict[str, str],
        login_as_test_user: None,
        configmap_name: str,
    ):
        """
        RHOAIENG-41850: Verify that non-admin users cannot access catalog ConfigMaps,
        receiving a 403 Forbidden error.
        """
        if is_byoidc:
            pytest.skip(reason="BYOIDC test users may have pre-configured group memberships")

        # get_client() uses the current kubeconfig context (set by login_as_test_user fixture)
        user_client = get_client()

        with pytest.raises(ApiException) as exc_info:
            catalog_cm = ConfigMap(
                name=configmap_name,
                namespace=model_registry_namespace,
                client=user_client,
            )
            _ = catalog_cm.instance  # Access the ConfigMap instance to trigger the API call

        assert exc_info.value.status == 403, (
            f"Expected HTTP 403 Forbidden for non-admin user accessing '{configmap_name}', "
            f"but got {exc_info.value.status}: {exc_info.value.reason}"
        )
        LOGGER.info(
            f"Non-admin user '{user_credentials_rbac['username']}' correctly denied access "
            f"to ConfigMap '{configmap_name}'"
        )
