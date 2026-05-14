import pytest
import structlog

from tests.model_registry.model_catalog.constants import (
    MODEL_ARTIFACT_TYPE,
    VALIDATED_CATALOG_ID,
)
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)

EXPECTED_REGISTRY_PREFIX = "registry.redhat.io"

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


@pytest.mark.skip_must_gather
class TestValidatedModelArtifactURI:
    """Tests for validated model artifact URI compliance (RHOAIENG-61451)."""

    @pytest.mark.tier1
    def test_validated_model_artifacts_use_redhat_registry(
        self,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Given all models in the validated catalog
        When fetching model artifacts for each model
        Then every model-artifact URI should contain registry.redhat.io
        """
        models_response = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={VALIDATED_CATALOG_ID}&pageSize=1000",
            headers=model_registry_rest_headers,
        )
        models = models_response.get("items", [])
        assert models, f"No models found in {VALIDATED_CATALOG_ID} catalog"
        LOGGER.info(f"Validating artifact URIs for {len(models)} validated models")

        validation_errors = []

        for model in models:
            model_name = model["name"]
            artifacts_url = (
                f"{model_catalog_rest_url[0]}sources/{VALIDATED_CATALOG_ID}/models/{model_name}/artifacts?pageSize=100"
            )
            artifacts_response = execute_get_command(url=artifacts_url, headers=model_registry_rest_headers)
            model_artifacts = [
                artifact
                for artifact in artifacts_response.get("items", [])
                if artifact.get("artifactType") == MODEL_ARTIFACT_TYPE
            ]

            if not model_artifacts:
                validation_errors.append(f"Model '{model_name}' has no model-artifact entries")
                continue

            for artifact in model_artifacts:
                uri = artifact.get("uri", "")
                if EXPECTED_REGISTRY_PREFIX not in uri:
                    validation_errors.append(f"Model '{model_name}' has non-Red Hat registry URI: '{uri}'")

        assert not validation_errors, (
            f"Artifact URI validation failed for {len(validation_errors)} model(s):\n" + "\n".join(validation_errors)
        )

        LOGGER.info(f"All {len(models)} validated models have artifact URIs containing '{EXPECTED_REGISTRY_PREFIX}'")
