import pytest
import yaml
import random
from kubernetes.dynamic import DynamicClient
from dictdiffer import diff
from ocp_resources.deployment import Deployment
from simple_logger.logger import get_logger
from typing import Self, Any

from ocp_resources.pod import Pod
from ocp_resources.config_map import ConfigMap
from ocp_resources.route import Route
from ocp_resources.service import Service
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOG_ID
from tests.model_registry.model_catalog.utils import (
    validate_model_catalog_enabled,
    execute_get_command,
    validate_model_catalog_resource,
    validate_default_catalog,
    get_validate_default_model_catalog_source,
)
from tests.model_registry.utils import get_rest_headers
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


@pytest.mark.skip_must_gather
class TestModelCatalogDefaultData:
    """Test class for validating default catalog data (not user-specific)"""

    def test_model_default_catalog_number_of_models(
        self: Self,
        default_catalog_api_response: dict[Any, Any],
        default_model_catalog_yaml_content: dict[Any, Any],
    ):
        """
        RHOAIENG-33667: Validate number of models in default catalog
        """

        count = len(default_model_catalog_yaml_content.get("models", []))

        assert count == default_catalog_api_response["size"], (
            f"Expected count: {count}, Actual size: {default_catalog_api_response['size']}"
        )
        LOGGER.info("Model count matches")

    def test_model_default_catalog_correspondence_of_model_name(
        self: Self,
        default_catalog_api_response: dict[Any, Any],
        default_model_catalog_yaml_content: dict[Any, Any],
    ):
        """
        RHOAIENG-35260: Validate the correspondence of model parameters in default catalog yaml and model catalog api
        """

        api_models = {model["name"]: model for model in default_catalog_api_response.get("items", [])}

        models_with_differences = {}

        for model in default_model_catalog_yaml_content.get("models", []):
            LOGGER.info(f"Validating model: {model['name']}")

            api_model = api_models.get(model["name"])

            # Exclude artifacts and null-valued properties from YAML model comparison
            model_filtered = {k: v for k, v in model.items() if k != "artifacts" and v is not None}

            differences = list(diff(model_filtered, api_model))
            if differences:
                models_with_differences[model["name"]] = differences
                LOGGER.warning(f"Found differences for {model['name']}: {differences}")

        assert not models_with_differences, (
            f"Found differences in {len(models_with_differences)} model(s): {models_with_differences}"
        )

    def test_model_default_catalog_random_artifact(
        self: Self,
        default_model_catalog_yaml_content: dict[Any, Any],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        RHOAIENG-35260: Validate the random artifact in default catalog yaml matches API response
        """

        random_model = random.choice(seq=default_model_catalog_yaml_content.get("models", []))
        LOGGER.info(f"Random model: {random_model['name']}")

        api_model_artifacts = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{DEFAULT_CATALOG_ID}/models/{random_model['name']}/artifacts",
            headers=model_registry_rest_headers,
        )["items"]

        yaml_artifacts = random_model.get("artifacts", [])

        assert api_model_artifacts, f"No artifacts found in API for {random_model['name']}"
        assert yaml_artifacts, f"No artifacts found in YAML for {random_model['name']}"

        # Compare artifacts (excluding timestamps which may differ)
        yaml_artifacts_filtered = [
            {k: v for k, v in artifact.items() if k not in ["lastUpdateTimeSinceEpoch"]} for artifact in yaml_artifacts
        ]
        api_artifacts_filtered = [
            {k: v for k, v in artifact.items() if k not in ["lastUpdateTimeSinceEpoch"]}
            for artifact in api_model_artifacts
        ]

        differences = list(diff(yaml_artifacts_filtered, api_artifacts_filtered))
        assert not differences, f"Artifacts mismatch for {random_model['name']}: {differences}"
        LOGGER.info("Artifacts match")
