import random
from typing import Any, Self

import pytest
import yaml
from dictdiffer import diff
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment
from ocp_resources.pod import Pod
from ocp_resources.route import Route
from ocp_resources.service import Service
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from tests.model_registry.constants import DEFAULT_CUSTOM_MODEL_CATALOG, DEFAULT_MODEL_CATALOG_CM
from tests.model_registry.model_catalog.catalog_config.utils import (
    extract_schema_fields,
    get_validate_default_model_catalog_source,
    validate_default_catalog,
    validate_model_catalog_enabled,
    validate_model_catalog_resource,
)
from tests.model_registry.model_catalog.constants import CATALOG_CONTAINER, DEFAULT_CATALOGS, REDHAT_AI_CATALOG_ID
from tests.model_registry.model_catalog.utils import execute_get_command
from tests.model_registry.utils import get_model_catalog_pod, get_rest_headers
from utilities.user_utils import UserTestSession

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
        "original_user",
        "test_idp_user",
    )
]


@pytest.mark.skip_must_gather
class TestModelCatalogGeneral:
    @pytest.mark.parametrize(
        "model_catalog_config_map, expected_catalogs, validate_catalog",
        [
            pytest.param(
                {"configmap_name": DEFAULT_CUSTOM_MODEL_CATALOG},
                0,
                False,
                id="test_model_catalog_sources_configmap_install",
                marks=pytest.mark.install,
            ),
            pytest.param(
                {"configmap_name": DEFAULT_CUSTOM_MODEL_CATALOG},
                1,
                False,
                id="test_model_catalog_sources_configmap_upgrade",
                marks=(pytest.mark.pre_upgrade, pytest.mark.post_upgrade),
            ),
            pytest.param(
                {"configmap_name": DEFAULT_MODEL_CATALOG_CM},
                2,
                True,
                id="test_model_catalog_default_sources_configmap",
            ),
        ],
        indirect=["model_catalog_config_map"],
    )
    def test_config_map_exists(
        self: Self, model_catalog_config_map: ConfigMap, expected_catalogs: int, validate_catalog: bool
    ) -> None:
        assert model_catalog_config_map.exists, f"{model_catalog_config_map.name} does not exist"
        catalogs = yaml.safe_load(model_catalog_config_map.instance.data["sources.yaml"])["catalogs"]
        assert len(catalogs) == expected_catalogs, (
            f"{model_catalog_config_map.name} should have {expected_catalogs} catalog"
        )
        if validate_catalog:
            validate_default_catalog(catalogs=catalogs)

    @pytest.mark.parametrize(
        "resource_name, expected_resource_count",
        [
            pytest.param(
                Deployment,
                1,
                id="test_model_catalog_deployment_resource",
            ),
            pytest.param(
                Route,
                1,
                id="test_model_catalog_route_resource",
            ),
            pytest.param(
                Service,
                1,
                id="test_model_catalog_service_resource",
            ),
            pytest.param(
                Pod,
                2,
                id="test_model_catalog_pod_resource",
            ),
        ],
    )
    @pytest.mark.post_upgrade
    @pytest.mark.pre_upgrade
    @pytest.mark.install
    def test_model_catalog_resources_exists(
        self: Self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        resource_name: Any,
        expected_resource_count: int,
    ):
        validate_model_catalog_resource(
            kind=resource_name,
            admin_client=admin_client,
            namespace=model_registry_namespace,
            expected_resource_count=expected_resource_count,
        )

    @pytest.mark.post_upgrade
    @pytest.mark.pre_upgrade
    @pytest.mark.install
    def test_operator_pod_enabled_model_catalog(self: Self, model_registry_operator_pod: Pod):
        assert validate_model_catalog_enabled(pod=model_registry_operator_pod)

    @pytest.mark.post_upgrade
    @pytest.mark.pre_upgrade
    @pytest.mark.install
    def test_model_catalog_uses_postgres(self: Self, admin_client: DynamicClient, model_registry_namespace: str):
        """
        Validate that model catalog pod is using PostgreSQL database
        """
        model_catalog_pods = get_model_catalog_pod(
            client=admin_client,
            model_registry_namespace=model_registry_namespace,
            label_selector="app.kubernetes.io/name=model-catalog",
        )
        assert len(model_catalog_pods) == 1
        model_catalog_pod = model_catalog_pods[0]
        model_catalog_pod_log = model_catalog_pod.log(container=CATALOG_CONTAINER)
        assert "Successfully connected to PostgreSQL database" in model_catalog_pod_log


@pytest.mark.parametrize(
    "user_token_for_api_calls,",
    [
        pytest.param(
            {},
            id="test_model_catalog_source_admin_user",
            marks=(pytest.mark.pre_upgrade, pytest.mark.post_upgrade, pytest.mark.install),
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
        pytestconfig: pytest.Config,
        test_idp_user: UserTestSession,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
    ):
        """
        Validate specific user can access default model catalog source
        """
        LOGGER.info("Attempting client connection with token")

        url = f"{model_catalog_rest_url[0]}sources"
        headers = get_rest_headers(token=user_token_for_api_calls)

        # Retry for up to 2 minutes to allow RBAC propagation
        # Accept ResourceNotFoundError (401) as a transient error during RBAC propagation
        sampler = TimeoutSampler(
            wait_timeout=120,
            sleep=5,
            func=lambda: execute_get_command(url=url, headers=headers)["items"],
            exceptions_dict={ResourceNotFoundError: []},
        )

        for result in sampler:
            break
        assert result
        items_to_validate = []
        if pytestconfig.option.pre_upgrade or pytestconfig.option.post_upgrade:
            items_to_validate.extend([catalog for catalog in result if catalog["id"] in DEFAULT_CATALOGS])
            assert len(items_to_validate) + 1 == len(result)
        else:
            items_to_validate = result
        get_validate_default_model_catalog_source(catalogs=items_to_validate)

    def test_model_default_catalog_get_models_by_source(
        self: Self,
        model_catalog_rest_url: list[str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """
        Validate a specific user can access models api for model catalog associated with a default source
        """
        random_model, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(f"picked model: {model_name} from catalog: {catalog_id}")
        assert random_model

    def test_model_default_catalog_get_model_by_name(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """
        Validate a specific user can access get Model by name associated with a default source
        """
        random_model, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{REDHAT_AI_CATALOG_ID}/models/{model_name}",
            headers=get_rest_headers(token=user_token_for_api_calls),
        )
        differences = list(diff(random_model, result))
        assert not differences, f"Expected no differences in model information for {model_name}: {differences}"

    def test_model_default_catalog_get_model_artifact(
        self: Self,
        model_catalog_rest_url: list[str],
        user_token_for_api_calls: str,
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """
        Validate a specific user can access get Model artifacts for model associated with default source
        """
        _, model_name, _ = randomly_picked_model_from_catalog_api_by_source
        result = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{REDHAT_AI_CATALOG_ID}/models/{model_name}/artifacts",
            headers=get_rest_headers(token=user_token_for_api_calls),
        )["items"]
        assert result, f"No artifacts found for {model_name}"
        assert result[0]["uri"]


@pytest.mark.post_upgrade
@pytest.mark.pre_upgrade
@pytest.mark.install
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
        catalog_openapi_schema: dict[Any, Any],
    ):
        """
        RHOAIENG-35260: Validate the correspondence of model parameters in default catalog yaml and model catalog api
        """

        all_model_fields, required_model_fields = extract_schema_fields(
            openapi_schema=catalog_openapi_schema, schema_name="CatalogModel"
        )
        LOGGER.info(f"All model fields from OpenAPI schema: {all_model_fields}")
        LOGGER.info(f"Required model fields from OpenAPI schema: {required_model_fields}")

        api_models = {model["name"]: model for model in default_catalog_api_response.get("items", [])}
        assert api_models

        models_with_differences = {}

        for model in default_model_catalog_yaml_content.get("models", []):
            LOGGER.info(f"Validating model: {model['name']}")

            api_model = api_models.get(model["name"])
            assert api_model, f"Model {model['name']} not found in API response"

            # Check required fields are present in both YAML and API
            yaml_missing_required = required_model_fields - set(model.keys())
            api_missing_required = required_model_fields - set(api_model.keys())

            assert not yaml_missing_required, (
                f"Model {model['name']} missing REQUIRED fields in YAML: {yaml_missing_required}"
            )
            assert not api_missing_required, (
                f"Model {model['name']} missing REQUIRED fields in API: {api_missing_required}"
            )

            # Check 'license' field presence without value comparison (API transforms format and tested u/s)
            if "license" in all_model_fields:
                yaml_has_license = "license" in model
                api_has_license = "license" in api_model
                assert yaml_has_license == api_has_license, (
                    f"License field presence mismatch for {model['name']}: "
                    f"YAML has license={yaml_has_license}, API has license={api_has_license}"
                )

            # Exclude 'license' field from value comparison
            comparable_fields = all_model_fields - {"license"}
            # Filter to only schema-defined fields for value comparison
            model_filtered = {key: value for key, value in model.items() if key in comparable_fields}
            api_model_filtered = {key: value for key, value in api_model.items() if key in comparable_fields}

            differences = list(diff(model_filtered, api_model_filtered))
            if differences:
                models_with_differences[model["name"]] = differences
                LOGGER.warning(f"Found value differences for {model['name']}: {differences}")

        # FAILS for null-valued properties in YAML model until https://issues.redhat.com/browse/RHOAIENG-35322 is fixed
        assert not models_with_differences, (
            f"Found differences in {len(models_with_differences)} model(s): {models_with_differences}"
        )
        LOGGER.info("Model correspondence matches")

    def test_model_default_catalog_random_artifact(
        self: Self,
        default_model_catalog_yaml_content: dict[Any, Any],
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        catalog_openapi_schema: dict[Any, Any],
    ):
        """
        RHOAIENG-35260: Validate the random artifact in default catalog yaml matches API response
        """

        all_artifact_fields, required_artifact_fields = extract_schema_fields(
            openapi_schema=catalog_openapi_schema, schema_name="CatalogModelArtifact"
        )
        LOGGER.info(f"All artifact fields from OpenAPI schema: {all_artifact_fields}")
        LOGGER.info(f"Required artifact fields from OpenAPI schema: {required_artifact_fields}")

        random_model = random.choice(seq=default_model_catalog_yaml_content.get("models", []))
        model_name = random_model["name"]
        LOGGER.info(f"Random model: {model_name}")

        api_model_artifacts = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{REDHAT_AI_CATALOG_ID}/models/{model_name}/artifacts",
            headers=model_registry_rest_headers,
        )["items"]

        yaml_artifacts = random_model.get("artifacts", [])
        assert api_model_artifacts, f"No artifacts found in API for {model_name}"
        assert yaml_artifacts, f"No artifacts found in YAML for {model_name}"
        assert len(yaml_artifacts) == len(api_model_artifacts), (
            f"Artifact count mismatch for {model_name}: YAML has {len(yaml_artifacts)}, API {len(api_model_artifacts)}"
        )

        for artifact in api_model_artifacts:
            missing_fields = required_artifact_fields - set(artifact.keys())
            assert not missing_fields, f"API artifact for {model_name} missing REQUIRED fields: {missing_fields}"

        comparable_fields = all_artifact_fields - {"artifactType"}

        # Filter artifacts to only include schema-defined fields for comparison
        yaml_artifacts_filtered = [
            {k: v for k, v in artifact.items() if k in comparable_fields} for artifact in yaml_artifacts
        ]
        api_artifacts_filtered = [
            {k: v for k, v in artifact.items() if k in comparable_fields} for artifact in api_model_artifacts
        ]

        differences = list(diff(yaml_artifacts_filtered, api_artifacts_filtered))
        assert not differences, f"Artifacts mismatch for {model_name}: {differences}"
        LOGGER.info("Artifacts match")
