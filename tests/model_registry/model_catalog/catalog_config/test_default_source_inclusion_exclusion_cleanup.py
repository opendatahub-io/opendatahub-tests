import pytest
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError
from ocp_resources.resource import ResourceEditor
from kubernetes.dynamic.client import DynamicClient
from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_CATALOG_ID,
    REDHAT_AI_CATALOG_NAME,
)
from tests.model_registry.model_catalog.catalog_config.utils import (
    validate_model_filtering_consistency,
    modify_catalog_source,
    get_models_from_database_by_source,
    wait_for_model_count_change,
    wait_for_model_set_match,
    validate_cleanup_logging,
    filter_models_by_pattern,
    execute_inclusion_exclusion_filter_test,
    ensure_baseline_model_state,
)
from tests.model_registry.utils import wait_for_model_catalog_api

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


class TestModelInclusionFiltering:
    """Test inclusion filtering functionality (RHOAIENG-41841)"""

    @pytest.mark.parametrize(
        "pattern,filter_value,expected_comment",
        [
            pytest.param("granite", "*granite*", "6/7", marks=pytest.mark.smoke, id="include_granite_models_only"),
            pytest.param(
                "prometheus", "*prometheus*", "1/7", marks=pytest.mark.sanity, id="include_prometheus_models_only"
            ),
            pytest.param("-8b-", "*-8b-*", "5/7", marks=pytest.mark.sanity, id="include_eight_b_models_only"),
            pytest.param("code", "*code*", "2/7", marks=pytest.mark.sanity, id="include_code_models_only"),
        ],
    )
    def test_include_models_by_pattern(
        self,
        pattern: str,
        filter_value: str,
        expected_comment: str,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict[str, set[str] | int],
    ):
        """Test that includedModels=[filter_value] shows only models matching pattern ({expected_comment})."""
        execute_inclusion_exclusion_filter_test(
            filter_type="inclusion",
            pattern=pattern,
            filter_value=filter_value,
            baseline_models=baseline_redhat_ai_models["api_models"],
            admin_client=admin_client,
            model_registry_namespace=model_registry_namespace,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
        )

        # Ensure baseline model state is restored for subsequent tests
        ensure_baseline_model_state(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
        )


class TestModelExclusionFiltering:
    """Test exclusion filtering functionality (RHOAIENG-41841 part 2)"""

    @pytest.mark.parametrize(
        "pattern,filter_value,expected_comment",
        [
            pytest.param("granite", "*granite*", "1/7 remaining", marks=pytest.mark.smoke, id="exclude_granite_models"),
            pytest.param(
                "prometheus", "*prometheus*", "6/7 remaining", marks=pytest.mark.sanity, id="exclude_prometheus_models"
            ),
            pytest.param("lab", "*lab*", "4/7 remaining", marks=pytest.mark.sanity, id="exclude_lab_models"),
        ],
    )
    def test_exclude_models_by_pattern(
        self,
        pattern: str,
        filter_value: str,
        expected_comment: str,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict[str, set[str] | int],
    ):
        """Test that excludedModels=[filter_value] removes models matching pattern ({expected_comment})."""
        execute_inclusion_exclusion_filter_test(
            filter_type="exclusion",
            pattern=pattern,
            filter_value=filter_value,
            baseline_models=baseline_redhat_ai_models["api_models"],
            admin_client=admin_client,
            model_registry_namespace=model_registry_namespace,
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
        )

        # Ensure baseline model state is restored for subsequent tests
        ensure_baseline_model_state(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
        )


class TestCombinedIncludeExcludeFiltering:
    """Test combined include+exclude filtering (RHOAIENG-41841 part 3)"""

    @pytest.mark.parametrize(
        "include_pattern,include_filter_value,exclude_pattern,exclude_filter_value,expected_comment",
        [
            pytest.param(
                "granite",
                "*granite*",
                "lab",
                "*lab*",
                "3/7 remaining",
                marks=pytest.mark.smoke,
                id="include_granite_exclude_lab",
            ),
            pytest.param(
                "-8b-",
                "*-8b-*",
                "code",
                "*code*",
                "3/7 remaining",
                marks=pytest.mark.sanity,
                id="include_eight_b_exclude_code",
            ),
        ],
    )
    def test_combined_include_exclude_filtering(
        self,
        include_pattern: str,
        include_filter_value: str,
        exclude_pattern: str,
        exclude_filter_value: str,
        expected_comment: str,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict[str, set[str] | int],
    ):
        """Test includedModels=[include_filter_value] + excludedModels=[exclude_filter_value] precedence."""
        LOGGER.info(f"Testing combined include {include_pattern}, exclude {exclude_pattern}")

        # Get models that match the inclusion pattern
        included_models = filter_models_by_pattern(
            all_models=baseline_redhat_ai_models["api_models"], pattern=include_pattern
        )
        # Remove models that also match the exclusion pattern from the included set
        expected_models = {model for model in included_models if exclude_pattern not in model}

        patch_info = modify_catalog_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=[include_filter_value],
            excluded_models=[exclude_filter_value],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=expected_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(
                    f"Timeout waiting for {include_pattern} minus {exclude_pattern} models. "
                    f"Expected: {expected_models}, {e}"
                )

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == expected_models, (
                f"Expected {include_pattern} minus {exclude_pattern} models {expected_models}, got {api_models}"
            )

            LOGGER.info(
                f"SUCCESS: {len(api_models)} {include_pattern} models after excluding {exclude_pattern} variants"
            )

        # Ensure baseline model state is restored for subsequent tests
        ensure_baseline_model_state(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
        )


class TestModelCleanupLifecycle:
    """Test automatic model cleanup during lifecycle changes (RHOAIENG-41846)"""

    @pytest.mark.sanity
    def test_model_cleanup_on_exclusion_change(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict[str, set[str] | int],
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
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
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
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            # Should only have prometheus models now
            assert phase2_api_models == prometheus_models, (
                f"Phase 2: Expected only prometheus {prometheus_models}, got {phase2_api_models}"
            )
            assert phase2_db_models == prometheus_models, "Phase 2: DB should match API"

            LOGGER.info(
                f"Phase 2 SUCCESS: Granite models cleaned up, {len(phase2_api_models)} prometheus models remain"
            )

        # Ensure baseline model state is restored for subsequent tests
        ensure_baseline_model_state(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
        )


class TestSourceLifecycleCleanup:
    """Test source disabling cleanup scenarios (RHOAIENG-41846)"""

    @pytest.mark.smoke
    def test_source_disabling_removes_models(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test that disabling a source removes all its models from the catalog."""
        LOGGER.info("Testing source disabling cleanup")

        # Disable the source
        disable_patch = modify_catalog_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            enabled=False,
        )

        with ResourceEditor(patches={disable_patch["configmap"]: disable_patch["patch"]}):
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            # Wait for models to be removed
            try:
                wait_for_model_count_change(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_count=0,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Expected all models to be removed when source is disabled: {e}")

            # Verify database is also cleaned
            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )
            assert len(db_models) == 0, f"Database should be clean when source disabled, found: {db_models}"

            LOGGER.info("SUCCESS: Source disabling removed all models")

        # Ensure baseline model state is restored for subsequent tests
        ensure_baseline_model_state(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
        )


class TestLoggingValidation:
    """Test cleanup operation logging (RHOAIENG-41846)"""

    @pytest.mark.sanity
    def test_model_removal_logging(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test that model removal operations are properly logged."""
        LOGGER.info("Testing model removal logging")

        # Apply filter to exclude granite models (should trigger removals)
        patch_info = modify_catalog_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            excluded_models=["*granite*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            # Wait for exclusion to take effect
            try:
                wait_for_model_count_change(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_count=1,  # Only prometheus should remain
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Expected granite models to be excluded: {e}")

            # Validate logging occurred
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

        # Ensure baseline model state is restored for subsequent tests
        ensure_baseline_model_state(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
        )

    @pytest.mark.sanity
    def test_source_disabling_logging(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test that source disabling operations are properly logged."""
        LOGGER.info("Testing source disabling logging")

        # Disable the source
        disable_patch = modify_catalog_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            enabled=False,
        )

        with ResourceEditor(patches={disable_patch["configmap"]: disable_patch["patch"]}):
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            # Wait for disabling to take effect
            try:
                wait_for_model_count_change(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_count=0,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Expected all models to be removed when source disabled: {e}")

            # Validate logging occurred
            expected_log_patterns = [rf"Removing models from source {REDHAT_AI_CATALOG_ID}"]

            try:
                found_patterns = validate_cleanup_logging(
                    client=admin_client, namespace=model_registry_namespace, expected_log_patterns=expected_log_patterns
                )
                LOGGER.info(f"SUCCESS: Found expected source disabling log patterns: {found_patterns}")
            except TimeoutExpiredError as e:
                pytest.fail(f"Expected source disabling log patterns not found: {e}")

        # Ensure baseline model state is restored for subsequent tests
        ensure_baseline_model_state(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            model_registry_namespace=model_registry_namespace,
        )
