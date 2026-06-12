import pytest
import structlog

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
        validated_model_artifact_uris: dict[str, list[str]],
    ):
        """Given all models in the validated catalog
        When fetching model artifacts for each model
        Then every model-artifact URI should contain registry.redhat.io
        """
        validation_errors = []

        for model_name, uris in validated_model_artifact_uris.items():
            if not uris:
                validation_errors.append(f"Model '{model_name}' has no model-artifact entries")
                continue

            for uri in uris:
                if EXPECTED_REGISTRY_PREFIX not in uri:
                    validation_errors.append(f"Model '{model_name}' has non-Red Hat registry URI: '{uri}'")

        assert not validation_errors, (
            f"Artifact URI validation failed for {len(validation_errors)} model(s):\n" + "\n".join(validation_errors)
        )

        LOGGER.info(
            f"All {len(validated_model_artifact_uris)} validated models have artifact URIs"
            f" containing '{EXPECTED_REGISTRY_PREFIX}'"
        )

    @pytest.mark.tier1
    def test_validated_models_have_single_model_artifact(
        self,
        validated_model_artifact_uris: dict[str, list[str]],
    ):
        """Given all models in the validated catalog
        When fetching model artifacts for each model
        Then every model should have exactly one model-artifact
        """
        validation_errors = []

        for model_name, uris in validated_model_artifact_uris.items():
            if len(uris) != 1:
                validation_errors.append(f"Model '{model_name}' has {len(uris)} model-artifact(s), expected 1: {uris}")

        assert not validation_errors, (
            f"Unexpected model-artifact count for {len(validation_errors)} model(s):\n" + "\n".join(validation_errors)
        )

        LOGGER.info(f"All {len(validated_model_artifact_uris)} validated models have exactly 1 model-artifact")
