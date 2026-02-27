import pytest
from kubernetes.dynamic.client import DynamicClient
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError

from tests.model_registry.model_catalog.catalog_config.utils import (
    filter_models_by_pattern,
    get_models_from_database_by_source,
    modify_catalog_source,
    validate_cleanup_logging,
    validate_filter_test_result,
    validate_source_disabling_result,
    wait_for_catalog_source_restore,
    wait_for_model_set_match,
)
from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_CATALOG_ID,
    REDHAT_AI_CATALOG_NAME,
)
from tests.model_registry.model_catalog.utils import wait_for_model_catalog_api

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


class TestModelInclusionFiltering:
    """Test inclusion filtering functionality (RHOAIENG-41841)"""

    @pytest.mark.parametrize(
        "redhat_ai_models_with_filter",
        [
            pytest.param(
                {"filter_type": "inclusion", "pattern": "granite", "filter_value": "*granite*"},
                marks=pytest.mark.tier2,
                id="test_include_granite_models_only",
            ),
            pytest.param(
                {"filter_type": "inclusion", "pattern": "prometheus", "filter_value": "*prometheus*"},
                marks=pytest.mark.tier1,
                id="test_include_prometheus_models_only",
            ),
            pytest.param(
                {"filter_type": "inclusion", "pattern": "-8b-", "filter_value": "*-8b-*"},
                marks=pytest.mark.tier2,
                id="test_include_eight_b_models_only",
            ),
            pytest.param(
                {"filter_type": "inclusion", "pattern": "code", "filter_value": "*code*"},
                marks=pytest.mark.tier2,
                id="test_include_code_models_only",
            ),
        ],
        indirect=True,
    )
    def test_include_models_by_pattern(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        redhat_ai_models_with_filter: set[str],
    ):
        """Test that includedModels=[filter_value] shows only models matching pattern."""
        validate_filter_test_result(
            admin_client=admin_client,
            expected_models=redhat_ai_models_with_filter,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
            filter_type="inclusion",
        )


class TestModelExclusionFiltering:
    """Test exclusion filtering functionality (RHOAIENG-41841 part 2)"""

    @pytest.mark.parametrize(
        "redhat_ai_models_with_filter",
        [
            pytest.param(
                {"filter_type": "exclusion", "pattern": "granite", "filter_value": "*granite*"},
                marks=pytest.mark.tier2,
                id="test_exclude_granite_models",
            ),
            pytest.param(
                {"filter_type": "exclusion", "pattern": "prometheus", "filter_value": "*prometheus*"},
                marks=pytest.mark.tier1,
                id="test_exclude_prometheus_models",
            ),
            pytest.param(
                {"filter_type": "exclusion", "pattern": "lab", "filter_value": "*lab*"},
                marks=pytest.mark.tier2,
                id="test_exclude_lab_models",
            ),
        ],
        indirect=True,
    )
    def test_exclude_models_by_pattern(
        self,
        admin_client: DynamicClient,
        redhat_ai_models_with_filter: set[str],
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test that excludedModels=[filter_value] removes models matching pattern."""
        validate_filter_test_result(
            admin_client=admin_client,
            expected_models=redhat_ai_models_with_filter,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
            filter_type="exclusion",
        )


class TestCombinedIncludeExcludeFiltering:
    """Test combined include+exclude filtering (RHOAIENG-41841 part 3)"""

    @pytest.mark.parametrize(
        "redhat_ai_models_with_filter",
        [
            pytest.param(
                {
                    "filter_type": "combined",
                    "include_pattern": "granite",
                    "include_filter_value": "*granite*",
                    "exclude_pattern": "lab",
                    "exclude_filter_value": "*lab*",
                },
                marks=pytest.mark.tier2,
                id="include_granite_exclude_lab",
            ),
            pytest.param(
                {
                    "filter_type": "combined",
                    "include_pattern": "-8b-",
                    "include_filter_value": "*-8b-*",
                    "exclude_pattern": "code",
                    "exclude_filter_value": "*code*",
                },
                marks=pytest.mark.tier1,
                id="include_eight_b_exclude_code",
            ),
        ],
        indirect=True,
    )
    def test_combined_include_exclude_filtering(
        self,
        admin_client: DynamicClient,
        redhat_ai_models_with_filter: set[str],
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test includedModels + excludedModels precedence."""
        validate_filter_test_result(
            admin_client=admin_client,
            expected_models=redhat_ai_models_with_filter,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
            filter_type="combined inclusion/exclusion",
        )


class TestModelCleanupLifecycle:
    """Test automatic model cleanup during lifecycle changes (RHOAIENG-41846)"""

    @pytest.mark.tier1
    def test_model_cleanup_on_exclusion_change(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict[str, set[str] | int],
        catalog_pod_model_counts: dict[str, int],
    ):
        """Test that models are cleaned up when filters change to exclude them."""
        LOGGER.info("Testing model cleanup on exclusion filter change")

        granite_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="granite")
        prometheus_models = filter_models_by_pattern(
            all_models=baseline_redhat_ai_models["api_models"], pattern="prometheus"
        )

        # Phase 1: Include only granite models
        phase1_patch = modify_catalog_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*granite*"],
        )

        try:
            with ResourceEditor(patches={phase1_patch["configmap"]: phase1_patch["patch"]}):
                wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

                # Verify granite models are present
                try:
                    phase1_api_models = wait_for_model_set_match(
                        model_catalog_rest_url=model_catalog_rest_url,
                        model_registry_rest_headers=model_registry_rest_headers,
                        source_label=REDHAT_AI_CATALOG_NAME,
                        expected_models=granite_models,
                    )
                except TimeoutExpiredError as e:
                    pytest.fail(f"Phase 1: Timeout waiting for granite models {granite_models}: {e}")

                phase1_db_models = get_models_from_database_by_source(
                    admin_client=admin_client, source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
                )

                assert phase1_api_models == granite_models, (
                    f"Phase 1: Expected granite models {granite_models}, got {phase1_api_models}"
                )
                assert phase1_db_models == granite_models, "Phase 1: DB should match API"

                LOGGER.info(f"Phase 1 SUCCESS: {len(phase1_api_models)} granite models included")

                # Phase 2: Change to exclude granite models (should trigger cleanup)
                phase2_patch = modify_catalog_source(
                    admin_client=admin_client,
                    namespace=model_registry_namespace,
                    source_id=REDHAT_AI_CATALOG_ID,
                    included_models=["*"],  # Include all
                    excluded_models=["*granite*"],  # But exclude granite
                )

                # Apply new filter without exiting context

                phase1_patch["configmap"].update(phase2_patch["patch"])

                wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

                # Verify granite models are removed (cleanup behavior)
                try:
                    phase2_api_models = wait_for_model_set_match(
                        model_catalog_rest_url=model_catalog_rest_url,
                        model_registry_rest_headers=model_registry_rest_headers,
                        source_label=REDHAT_AI_CATALOG_NAME,
                        expected_models=prometheus_models,
                    )
                except TimeoutExpiredError as e:
                    pytest.fail(f"Phase 2: Timeout waiting for prometheus models {prometheus_models}: {e}")

                phase2_db_models = get_models_from_database_by_source(
                    admin_client=admin_client, source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
                )

                # Should only have prometheus models now
                assert phase2_api_models == prometheus_models, (
                    f"Phase 2: Expected only prometheus {prometheus_models}, got {phase2_api_models}"
                )
                assert phase2_db_models == prometheus_models, "Phase 2: DB should match API"

                LOGGER.info(
                    f"Phase 2 SUCCESS: Granite models cleaned up, {len(phase2_api_models)} prometheus models remain"
                )
        finally:
            # Ensure clean up of the configpams
            wait_for_catalog_source_restore(
                model_catalog_rest_url=model_catalog_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                source_label=REDHAT_AI_CATALOG_NAME,
                expected_count=catalog_pod_model_counts[REDHAT_AI_CATALOG_ID],
            )


@pytest.mark.usefixtures("disabled_redhat_ai_source")
class TestSourceLifecycleCleanup:
    """Test source disabling cleanup scenarios (RHOAIENG-41846)"""

    @pytest.mark.tier2
    def test_source_disabling_removes_models(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test that disabling a source removes all its models from the catalog."""
        LOGGER.info("Testing source disabling cleanup")

        validate_source_disabling_result(
            admin_client=admin_client,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
        )

    @pytest.mark.tier2
    def test_source_disabling_logging(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test that source disabling operations are properly logged."""
        # Validate logging occurred
        expected_log_patterns = [rf"Removing models from source {REDHAT_AI_CATALOG_ID}"]

        try:
            found_patterns = validate_cleanup_logging(
                client=admin_client, namespace=model_registry_namespace, expected_log_patterns=expected_log_patterns
            )
            LOGGER.info(f"SUCCESS: Found expected source disabling log patterns: {found_patterns}")
        except TimeoutExpiredError as e:
            pytest.fail(f"Expected source disabling log patterns not found: {e}")


class TestLoggingValidation:
    """Test cleanup operation logging (RHOAIENG-41846)"""

    @pytest.mark.parametrize(
        "redhat_ai_models_with_filter",
        [
            pytest.param(
                {"filter_type": "exclusion", "pattern": "granite", "filter_value": "*granite*", "log_cleanup": True},
                marks=pytest.mark.tier2,
                id="test_exclude_granite_models_for_logging",
            )
        ],
        indirect=True,
    )
    def test_model_removal_logging(
        self,
        redhat_ai_models_with_filter: set[str],
        admin_client: DynamicClient,
        model_registry_namespace: str,
    ):
        """Test that model removal operations are properly logged."""
        LOGGER.info("Testing model removal logging")

        # Validate logging occurred for granite model removals
        expected_log_patterns = [
            rf"Removing {REDHAT_AI_CATALOG_ID} model .*granite.*",
        ]

        try:
            found_patterns = validate_cleanup_logging(
                client=admin_client, namespace=model_registry_namespace, expected_log_patterns=expected_log_patterns
            )
            LOGGER.info(f"SUCCESS: Found expected log patterns: {found_patterns}")
        except TimeoutExpiredError as e:
            pytest.fail(f"Expected log patterns not found: {e}")
