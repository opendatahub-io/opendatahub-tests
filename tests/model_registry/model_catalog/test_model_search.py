import pytest
from dictdiffer import diff

from simple_logger.logger import get_logger
from typing import Self, Any
from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_FILTER,
    REDHAT_AI_VALIDATED_FILTER,
    REDHAT_AI_CATALOG_ID,
    VALIDATED_CATALOG_ID,
    MODEL_ARTIFACT_TYPE,
    METRICS_ARTIFACT_TYPE,
)
from tests.model_registry.model_catalog.utils import (
    get_models_from_catalog_api,
    fetch_all_artifacts_with_dynamic_paging,
)
from kubernetes.dynamic.exceptions import ResourceNotFoundError

LOGGER = get_logger(name=__name__)
pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace", "test_idp_user")
]


class TestSearchModelCatalog:
    @pytest.mark.smoke
    def test_search_model_catalog_source_label(
        self: Self, model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str]
    ):
        """
        RHOAIENG-33656: Validate search model catalog by source label
        """

        redhat_ai_filter_moldels_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_FILTER,
        )["size"]
        redhat_ai_validated_filter_models_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_VALIDATED_FILTER,
        )["size"]
        no_filtered_models_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url, model_registry_rest_headers=model_registry_rest_headers
        )["size"]
        both_filtered_models_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=f"{REDHAT_AI_VALIDATED_FILTER},{REDHAT_AI_FILTER}",
        )["size"]

        assert no_filtered_models_size == both_filtered_models_size
        assert redhat_ai_filter_moldels_size + redhat_ai_validated_filter_models_size == both_filtered_models_size

    def test_search_model_catalog_invalid_source_label(
        self: Self, model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str]
    ):
        """
        RHOAIENG-33656:
        Validate search model catalog by invalid source label
        """

        null_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label="null",
        )["size"]

        invalid_size = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label="invalid",
        )["size"]

        assert null_size == invalid_size == 0, (
            "Expected 0 models for null and invalid source label found {null_size} and {invalid_size}"
        )

    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source,source_filter",
        [
            pytest.param(
                {"source": VALIDATED_CATALOG_ID, "header_type": "registry"},
                REDHAT_AI_VALIDATED_FILTER,
                id="test_search_model_catalog_redhat_ai_validated",
            ),
            pytest.param(
                {"source": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
                REDHAT_AI_FILTER,
                id="test_search_model_catalog_redhat_ai_default",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_search_model_catalog_match(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
        source_filter: str,
    ):
        """
        RHOAIENG-33656: Validate search model catalog by match
        """
        random_model, random_model_name, _ = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(f"random_model_name: {random_model_name}")
        result = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=source_filter,
            additional_params=f"&filterQuery=name='{random_model_name}'",
        )
        assert random_model_name == result["items"][0]["name"]
        assert result["size"] == 1

        differences = list(diff(random_model, result["items"][0]))
        assert not differences, f"Expected no differences in model information for {random_model_name}: {differences}"
        LOGGER.info("Model information matches")


# All the tests in this class are failing for RHOAIENG-36938, there are two problems:
# 1. The filter parameter is setup to use artifact_type instead of artifactType
# 2. The filter with multiple artifact types is not working as expected
@pytest.mark.xfail(
    strict=True,
    reason="RHOAIENG-36938: artifact_type is usedinstead of artifactType, multiple artifact types are not working",
)
class TestSearchModelArtifact:
    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source, artifact_type",
        [
            pytest.param(
                {"catalog_id": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
                MODEL_ARTIFACT_TYPE,
                id="redhat_ai_model_artifact",
            ),
            pytest.param(
                {"catalog_id": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
                METRICS_ARTIFACT_TYPE,
                id="redhat_ai_metrics_artifact",
            ),
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                MODEL_ARTIFACT_TYPE,
                id="validated_model_artifact",
            ),
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                METRICS_ARTIFACT_TYPE,
                id="validated_metrics_artifact",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_validate_model_artifacts_by_artifact_type(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
        artifact_type: str,
    ):
        """
        RHOAIENG-33659: Validates that the model artifacts returned by the artifactType filter
        match the complete set of artifacts for a random model.
        """
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source
        LOGGER.info(f"Artifact type: '{artifact_type}'")

        # Fetch all artifacts with dynamic page size adjustment
        all_model_artifacts = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?pageSize",
            headers=model_registry_rest_headers,
            page_size=100,
        )["items"]

        # Fetch filtered artifacts by type with dynamic page size adjustment
        artifact_type_artifacts = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=(
                f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?"
                f"artifactType={artifact_type}&pageSize"
            ),
            headers=model_registry_rest_headers,
            page_size=50,
        )["items"]

        # Create lookup for validation
        all_artifacts_by_id = {artifact["id"]: artifact for artifact in all_model_artifacts}

        # Verify all filtered artifacts exist
        for artifact in artifact_type_artifacts:
            artifact_id = artifact["id"]
            assert artifact_id in all_artifacts_by_id, (
                f"Filtered artifact {artifact_id} not found in complete artifact list for {model_name}"
            )

            differences = list(diff(artifact, all_artifacts_by_id[artifact_id]))
            assert not differences, f"Artifact {artifact_id} mismatch for {model_name}: {differences}"

        # Verify the filter didn't miss any artifacts of the type
        artifacts_of_type_in_all = [
            artifact for artifact in all_model_artifacts if artifact.get("artifactType") == artifact_type
        ]
        assert len(artifact_type_artifacts) == len(artifacts_of_type_in_all), (
            f"Filter returned {len(artifact_type_artifacts)} {artifact_type} artifacts, "
            f"but found {len(artifacts_of_type_in_all)} in complete list for {model_name}"
        )

        LOGGER.info(f"Validated {len(artifact_type_artifacts)} {artifact_type} artifacts for {model_name}")

    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source",
        [
            pytest.param(
                {"header_type": "registry"},
                id="invalid_artifact_type",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_error_handled_for_invalid_artifact_type(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """
        RHOAIENG-33659: Validates that the API returns the correct error when an invalid artifactType
        is provided regardless of catalog or model.
        """
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source

        invalid_artifact_type = "invalid"
        LOGGER.info(f"Testing invalid artifact type: '{invalid_artifact_type}'")

        with pytest.raises(ResourceNotFoundError, match=f"unsupported catalog artifact type: {invalid_artifact_type}"):
            fetch_all_artifacts_with_dynamic_paging(
                url_with_pagesize=(
                    f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?"
                    f"artifactType={invalid_artifact_type}&pageSize"
                ),
                headers=model_registry_rest_headers,
                page_size=1,
            )

        LOGGER.info(f"Successfully validated that invalid artifact type '{invalid_artifact_type}' raises an error")

    @pytest.mark.parametrize(
        "randomly_picked_model_from_catalog_api_by_source",
        [
            pytest.param(
                {"catalog_id": REDHAT_AI_CATALOG_ID, "header_type": "registry"},
                id="redhat_ai_catalog",
            ),
            pytest.param(
                {"catalog_id": VALIDATED_CATALOG_ID, "header_type": "registry"},
                id="validated_catalog",
            ),
        ],
        indirect=["randomly_picked_model_from_catalog_api_by_source"],
    )
    def test_multiple_artifact_type_filtering(
        self: Self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        randomly_picked_model_from_catalog_api_by_source: tuple[dict[Any, Any], str, str],
    ):
        """
        RHOAIENG-33659: Validates that the API returns all artifacts of a random model
        when filtering by multiple artifact types.
        """
        _, model_name, catalog_id = randomly_picked_model_from_catalog_api_by_source
        artifact_types = f"{METRICS_ARTIFACT_TYPE},{MODEL_ARTIFACT_TYPE}"
        LOGGER.info(f"Testing multiple artifact types: '{artifact_types}'")
        # Fetch all artifacts with dynamic page size adjustment
        all_model_artifacts = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?pageSize",
            headers=model_registry_rest_headers,
            page_size=100,
        )["items"]

        # Fetch filtered artifacts by type with dynamic page size adjustment
        artifact_type_artifacts = fetch_all_artifacts_with_dynamic_paging(
            url_with_pagesize=(
                f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}/artifacts?"
                f"artifactType={artifact_types}&pageSize"
            ),
            headers=model_registry_rest_headers,
            page_size=100,
        )["items"]

        assert len(artifact_type_artifacts) == len(all_model_artifacts), (
            f"Filter returned {len(artifact_type_artifacts)} artifacts, "
            f"but found {len(all_model_artifacts)} in complete list for {model_name}"
        )
