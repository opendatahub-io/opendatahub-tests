"""
Test suite for verifying RBAC permissions for Model Catalog ConfigMaps.
"""

import pytest
from simple_logger.logger import get_logger

from kubernetes.dynamic import DynamicClient
from kubernetes.client.rest import ApiException
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import get_client

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

    @pytest.mark.smoke
    @pytest.mark.parametrize(
        "user_params,configmap_name",
        [
            pytest.param(
                {},
                DEFAULT_MODEL_CATALOG_CM,
                id="admin_read_default_sources",
                marks=(pytest.mark.pre_upgrade, pytest.mark.post_upgrade, pytest.mark.install),
            ),
            pytest.param(
                {},
                DEFAULT_CUSTOM_MODEL_CATALOG,
                id="admin_read_custom_sources",
                marks=(pytest.mark.pre_upgrade, pytest.mark.post_upgrade, pytest.mark.install),
            ),
            pytest.param(
                {"user_type": "test"},
                DEFAULT_MODEL_CATALOG_CM,
                id="non_admin_denied_default_sources",
            ),
            pytest.param(
                {"user_type": "test"},
                DEFAULT_CUSTOM_MODEL_CATALOG,
                id="non_admin_denied_custom_sources",
            ),
        ],
    )
    def test_catalog_configmap_rbac(
        self,
        is_byoidc: bool,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        user_credentials_rbac: dict[str, str],
        login_as_test_user: None,
        user_params: dict,
        configmap_name: str,
    ):
        """
        RHOAIENG-41850: Verify RBAC permissions for catalog ConfigMaps.

        Admin users should have:
        - get/watch on model-catalog-default-sources (read-only)
        - get/watch/update/patch on model-catalog-sources (read/write)

        Non-admin users should receive 403 Forbidden when accessing either ConfigMap.

        Note: Admin write access to model-catalog-sources is already tested by existing tests
        (test_custom_model_catalog.py, test_catalog_source_merge.py) which use admin_client
        to successfully update ConfigMaps via ResourceEditor.
        """
        is_test_user = user_params.get("user_type") == "test"

        # Select client based on user type
        client = get_client() if is_test_user else admin_client
        catalog_cm = ConfigMap(
            name=configmap_name,
            namespace=model_registry_namespace,
            client=client,
        )

        if is_test_user:
            if is_byoidc:
                pytest.skip(reason="BYOIDC test users may have pre-configured group memberships")
            # Non-admin user - should receive 403 Forbidden
            with pytest.raises(ApiException) as exc_info:
                _ = catalog_cm.instance  # Trigger the API call

            assert exc_info.value.status == 403, (
                f"Expected HTTP 403 Forbidden for non-admin user accessing '{configmap_name}', "
                f"but got {exc_info.value.status}: {exc_info.value.reason}"
            )
            LOGGER.info(
                f"Non-admin user '{user_credentials_rbac['username']}' correctly denied access "
                f"to ConfigMap '{configmap_name}'"
            )
        else:
            # Admin user - should be able to read
            assert catalog_cm.exists, (
                f"ConfigMap '{configmap_name}' not found in namespace '{model_registry_namespace}'"
            )

            data = catalog_cm.instance.data
            assert data is not None, f"Admin should be able to read ConfigMap '{configmap_name}' data"

            sources_yaml = data.get("sources.yaml")
            assert sources_yaml is not None, f"ConfigMap '{configmap_name}' should contain 'sources.yaml' key"

            LOGGER.info(f"Admin successfully read ConfigMap '{configmap_name}'")
