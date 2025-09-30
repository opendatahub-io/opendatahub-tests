import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from dictdiffer import diff
from ocp_resources.deployment import Deployment
from simple_logger.logger import get_logger
from typing import Self, Any

from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from ocp_resources.route import Route
from ocp_resources.service import Service
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOG_ID, POD_DEFAULT_CATALOG_FILE, CATALOG_CONTAINER
from tests.model_registry.model_catalog.utils import (
    validate_model_catalog_enabled,
    execute_get_command,
    validate_model_catalog_resource,
    validate_default_catalog,
    get_validate_default_model_catalog_source,
)
from tests.model_registry.utils import get_rest_headers, get_model_catalog_pod
from utilities.user_utils import UserTestSession

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session", "model_registry_namespace", "original_user", "test_idp_user"
    )
]


@pytest.mark.skip_must_gather
class TestModelCatalogGeneral:
    @pytest.mark.post_upgrade
    def test_config_map_exists(self: Self, catalog_config_map: ConfigMap):
        # Check that the default configmaps is created when model registry is
        # enabled on data science cluster.
        assert catalog_config_map.exists, f"{catalog_config_map.name} does not exist"
        catalogs = yaml.safe_load(catalog_config_map.instance.data["sources.yaml"])["catalogs"]
        assert catalogs
        assert len(catalogs) == 1, f"{catalog_config_map.name} should have 1 catalog"
        validate_default_catalog(default_catalog=catalogs[0])

    @pytest.mark.parametrize(
        "resource_name",
        [
            pytest.param(
                Deployment,
                id="test_model_catalog_deployment_resource",
            ),
            pytest.param(
                Route,
                id="test_model_catalog_route_resource",
            ),
            pytest.param(
                Service,
                id="test_model_catalog_service_resource",
            ),
            pytest.param(
                Pod,
                id="test_model_catalog_pod_resource",
            ),
        ],
    )
    @pytest.mark.post_upgrade
    def test_model_catalog_resources_exists(
        self: Self, admin_client: DynamicClient, model_registry_namespace: str, resource_name: Any
    ):
        validate_model_catalog_resource(
            kind=resource_name, admin_client=admin_client, namespace=model_registry_namespace
        )

    def test_operator_pod_enabled_model_catalog(self: Self, model_registry_operator_pod: Pod):
        assert validate_model_catalog_enabled(pod=model_registry_operator_pod)


@pytest.mark.parametrize(
    "user_token_for_api_calls,",
    [
        pytest.param(
            {},
            id="test_model_catalog_source_admin_user",
        ),
        pytest.param(
            {"user_type": "test"},
            id="test_model_catalog_source_non_admin_user",
        ),
        pytest.param(
            {"user_type": "sa_user"},
            id="test_model_catalog_source_service_account",
        ),
    ],
    indirect=["user_token_for_api_calls"],
)
class TestModelCatalogDefault:
    def test_model_catalog_default_catalog_sources(
        self,
        test_idp_user: UserTestSession,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
    ):
        """
        Validate specific user can access default model catalog source
        """
        get_validate_default_model_catalog_source(
            token=user_token_for_api_calls, model_catalog_url=f"{model_catalog_rest_url[0]}sources"
        )

    def test_model_default_catalog_get_models_by_source(
        self: Self,
        model_catalog_rest_url: list[str],
        randomly_picked_model_from_default_catalog: dict[Any, Any],
    ):
        """
        Validate a specific user can access models api for model catalog associated with a default source
        """
        LOGGER.info(f"picked model: {randomly_picked_model_from_default_catalog}")
        assert randomly_picked_model_from_default_catalog

    def test_model_default_catalog_get_model_by_name(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        randomly_picked_model_from_default_catalog: dict[Any, Any],
    ):
        """
        Validate a specific user can access get Model by name associated with a default source
        """
        model_name = randomly_picked_model_from_default_catalog["name"]
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{DEFAULT_CATALOG_ID}/models/{model_name}",
            headers=get_rest_headers(token=user_token_for_api_calls),
        )
        differences = list(diff(randomly_picked_model_from_default_catalog, result))
        assert not differences, f"Expected no differences in model information for {model_name}: {differences}"

    def test_model_default_catalog_get_model_artifact(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        randomly_picked_model_from_default_catalog: dict[Any, Any],
    ):
        """
        Validate a specific user can access get Model artifacts for model associated with default source
        """
        model_name = randomly_picked_model_from_default_catalog["name"]
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{DEFAULT_CATALOG_ID}/models/{model_name}/artifacts",
            headers=get_rest_headers(token=user_token_for_api_calls),
        )["items"]
        assert result, f"No artifacts found for {model_name}"
        assert result[0]["uri"]

    def test_model_default_catalog_number_of_models(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
    ):
        """
        RHOAIENG-33667: Validate number of models in default catalog
        """

        model_catalog_pod = get_model_catalog_pod(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )[0]

        command = ["sh", "-c", f"grep '^ *- name:' {POD_DEFAULT_CATALOG_FILE} | wc -l"]

        result = model_catalog_pod.execute(command=command, container=CATALOG_CONTAINER)
        count = int(result.strip())
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={DEFAULT_CATALOG_ID}&pageSize=100",
            headers=get_rest_headers(token=user_token_for_api_calls),
        )
        assert count == len(result["items"]) == result["size"], (
            f"Expected count: {count}, Actual count: {len(result['items'])}, Actual size: {result['size']}"
        )
