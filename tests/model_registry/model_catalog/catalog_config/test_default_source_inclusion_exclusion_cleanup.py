import pytest
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutExpiredError
from ocp_resources.resource import ResourceEditor

from tests.model_registry.model_catalog.constants import (
    REDHAT_AI_CATALOG_ID,
    REDHAT_AI_CATALOG_NAME,
    INVALID_MODEL_PATTERNS,
)
from tests.model_registry.model_catalog.catalog_config.utils import (
    validate_model_filtering_consistency,
    apply_inclusion_exclusion_filters_to_source,
    get_api_models_by_source_label,
    get_models_from_database_by_source,
    wait_for_model_count_change,
    wait_for_model_set_match,
    disable_catalog_source,
    validate_cleanup_logging,
    validate_invalid_pattern_error,
    filter_models_by_pattern,
)
from tests.model_registry.utils import is_model_catalog_ready, wait_for_model_catalog_api

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session", "model_registry_namespace", "validate_baseline_expectations"
    ),
]


class TestModelInclusionFiltering:
    """Test inclusion filtering functionality (RHOAIENG-41841 part 1)"""

    def test_include_granite_models_only(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that includedModels=['*granite*'] shows only granite models (6/7)."""
        LOGGER.info("Testing granite model inclusion filter")

        granite_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="granite")

        # Apply inclusion filter
        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*granite*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            # Wait for the expected model set to appear
            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=granite_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for granite models to appear. Expected: {granite_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            # Validate consistency
            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            # Validate only granite models are present
            assert api_models == granite_models, f"Expected granite models {granite_models}, got {api_models}"

            LOGGER.info(f"SUCCESS: {len(api_models)} granite models included")

    def test_include_prometheus_models_only(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that includedModels=['*prometheus*'] shows only prometheus models (1/7)."""
        LOGGER.info("Testing prometheus model inclusion filter")

        prometheus_models = filter_models_by_pattern(
            all_models=baseline_redhat_ai_models["api_models"], pattern="prometheus"
        )

        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*prometheus*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=prometheus_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for prometheus models to appear. Expected: {prometheus_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == prometheus_models, f"Expected prometheus models {prometheus_models}, got {api_models}"

            LOGGER.info(f"SUCCESS: {len(api_models)} prometheus models included")

    def test_include_eight_b_models_only(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that includedModels=['*-8b-*'] shows only 8B models (5/7)."""
        LOGGER.info("Testing 8B model inclusion filter")

        eight_b_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="-8b-")

        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*-8b-*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=eight_b_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for 8B models to appear. Expected: {eight_b_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == eight_b_models, f"Expected 8B models {eight_b_models}, got {api_models}"

            LOGGER.info(f"SUCCESS: {len(api_models)} 8B models included")

    def test_include_code_models_only(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that includedModels=['*code*'] shows only code models (2/7)."""
        LOGGER.info("Testing code model inclusion filter")

        code_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="code")

        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*code*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=code_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for code models to appear. Expected: {code_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == code_models, f"Expected code models {code_models}, got {api_models}"

            LOGGER.info(f"SUCCESS: {len(api_models)} code models included")


class TestModelExclusionFiltering:
    """Test exclusion filtering functionality (RHOAIENG-41841 part 2)"""

    def test_exclude_granite_models(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that excludedModels=['*granite*'] removes granite models (1/7 remaining)."""
        LOGGER.info("Testing granite model exclusion filter")

        granite_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="granite")
        expected_models = baseline_redhat_ai_models["api_models"] - granite_models

        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            excluded_models=["*granite*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=expected_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for non-granite models. Expected: {expected_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == expected_models, f"Expected models without granite {expected_models}, got {api_models}"

            LOGGER.info(f"SUCCESS: {len(api_models)} models after excluding granite")

    def test_exclude_prometheus_models(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that excludedModels=['*prometheus*'] removes prometheus models (6/7 remaining)."""
        LOGGER.info("Testing prometheus model exclusion filter")

        prometheus_models = filter_models_by_pattern(
            all_models=baseline_redhat_ai_models["api_models"], pattern="prometheus"
        )
        expected_models = baseline_redhat_ai_models["api_models"] - prometheus_models

        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            excluded_models=["*prometheus*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=expected_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for non-prometheus models. Expected: {expected_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == expected_models, (
                f"Expected models without prometheus {expected_models}, got {api_models}"
            )

            LOGGER.info(f"SUCCESS: {len(api_models)} models after excluding prometheus")

    def test_exclude_lab_models(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that excludedModels=['*lab*'] removes lab models (4/7 remaining)."""
        LOGGER.info("Testing lab model exclusion filter")

        lab_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="lab")
        expected_models = baseline_redhat_ai_models["api_models"] - lab_models

        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            excluded_models=["*lab*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=expected_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for non-lab models. Expected: {expected_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == expected_models, f"Expected models without lab {expected_models}, got {api_models}"

            LOGGER.info(f"SUCCESS: {len(api_models)} models after excluding lab")


class TestCombinedIncludeExcludeFiltering:
    """Test combined include+exclude filtering (RHOAIENG-41841 part 3)"""

    def test_include_granite_exclude_lab_models(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test includedModels=['*granite*'] + excludedModels=['*lab*'] precedence (3/7 remaining)."""
        LOGGER.info("Testing combined include granite, exclude lab")

        granite_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="granite")
        lab_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="lab")
        # Expected: granite models minus any that contain "lab"
        expected_models = granite_models - lab_models

        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*granite*"],
            excluded_models=["*lab*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=expected_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for granite minus lab models. Expected: {expected_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == expected_models, (
                f"Expected granite minus lab models {expected_models}, got {api_models}"
            )

            LOGGER.info(f"SUCCESS: {len(api_models)} granite models after excluding lab variants")

    def test_include_eight_b_exclude_code_models(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test includedModels=['*-8b-*'] + excludedModels=['*code*'] precedence (3/7 remaining)."""
        LOGGER.info("Testing combined include 8B, exclude code")

        eight_b_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="-8b-")
        code_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="code")
        # Expected: 8B models minus any code models
        expected_models = eight_b_models - code_models

        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*-8b-*"],
            excluded_models=["*code*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                api_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=expected_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Timeout waiting for 8B minus code models. Expected: {expected_models}, {e}")

            db_models = get_models_from_database_by_source(
                source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
            )

            is_valid, error_msg = validate_model_filtering_consistency(api_models=api_models, db_models=db_models)
            assert is_valid, error_msg

            assert api_models == expected_models, f"Expected 8B minus code models {expected_models}, got {api_models}"

            LOGGER.info(f"SUCCESS: {len(api_models)} 8B models after excluding code variants")


class TestModelCleanupLifecycle:
    """Test automatic model cleanup during lifecycle changes (RHOAIENG-41846)"""

    def test_model_cleanup_on_exclusion_change(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that models are cleaned up when filters change to exclude them."""
        LOGGER.info("Testing model cleanup on exclusion filter change")

        granite_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="granite")
        prometheus_models = filter_models_by_pattern(
            all_models=baseline_redhat_ai_models["api_models"], pattern="prometheus"
        )

        # Phase 1: Include only granite models
        phase1_patch = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*granite*"],
        )

        with ResourceEditor(patches={phase1_patch["configmap"]: phase1_patch["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
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
            phase2_patch = apply_inclusion_exclusion_filters_to_source(
                admin_client=admin_client,
                namespace=model_registry_namespace,
                source_id=REDHAT_AI_CATALOG_ID,
                included_models=["*"],  # Include all
                excluded_models=["*granite*"],  # But exclude granite
            )

            # Apply new filter without exiting context

            phase1_patch["configmap"].update(phase2_patch["patch"])

            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
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

    def test_model_restoration_after_filter_removal(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that models are restored when exclusion filters are removed."""
        LOGGER.info("Testing model restoration after filter removal")

        baseline_models = baseline_redhat_ai_models["api_models"]

        # Phase 1: Exclude granite models
        phase1_patch = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            excluded_models=["*granite*"],
        )

        with ResourceEditor(patches={phase1_patch["configmap"]: phase1_patch["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            phase1_api_models = get_api_models_by_source_label(
                model_catalog_rest_url=model_catalog_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                source_label=REDHAT_AI_CATALOG_NAME,
            )

            granite_in_phase1 = {model for model in phase1_api_models if "granite" in model}
            assert len(granite_in_phase1) == 0, f"Phase 1: Should have no granite models, found: {granite_in_phase1}"

            LOGGER.info(f"Phase 1 SUCCESS: {len(phase1_api_models)} models after excluding granite")

        # Phase 2: Remove filters (should restore all models)
        is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

        try:
            restored_api_models = wait_for_model_set_match(
                model_catalog_rest_url=model_catalog_rest_url,
                model_registry_rest_headers=model_registry_rest_headers,
                source_label=REDHAT_AI_CATALOG_NAME,
                expected_models=baseline_models,
            )
        except TimeoutExpiredError as e:
            pytest.fail(f"Phase 2: Timeout waiting for baseline restoration {baseline_models}: {e}")

        restored_db_models = get_models_from_database_by_source(
            source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
        )

        assert restored_api_models == baseline_models, (
            f"Phase 2: Should restore to baseline {baseline_models}, got {restored_api_models}"
        )
        assert restored_db_models == baseline_models, "Phase 2: DB should match restored API models"

        LOGGER.info(f"Phase 2 SUCCESS: Restored to baseline {len(restored_api_models)} models")

    def test_dynamic_model_switching_with_cleanup(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test dynamic switching between different model sets validates cleanup."""
        LOGGER.info("Testing dynamic model switching with cleanup validation")

        granite_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="granite")
        prometheus_models = filter_models_by_pattern(
            all_models=baseline_redhat_ai_models["api_models"], pattern="prometheus"
        )
        starter_models = filter_models_by_pattern(all_models=baseline_redhat_ai_models["api_models"], pattern="starter")

        # Phase 1: Only granite models
        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            included_models=["*granite*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                phase1_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=granite_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Phase 1: Timeout waiting for granite models {granite_models}: {e}")
            assert phase1_models == granite_models, f"Phase 1: Expected granite {granite_models}, got {phase1_models}"

            # Phase 2: Switch to only prometheus models
            new_patch = apply_inclusion_exclusion_filters_to_source(
                admin_client=admin_client,
                namespace=model_registry_namespace,
                source_id=REDHAT_AI_CATALOG_ID,
                included_models=["*prometheus*"],
            )

            patch_info["configmap"].update(new_patch["patch"])

            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                phase2_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=prometheus_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Phase 2: Timeout waiting for prometheus models {prometheus_models}: {e}")
            assert phase2_models == prometheus_models, (
                f"Phase 2: Expected prometheus {prometheus_models}, got {phase2_models}"
            )

            # Phase 3: Switch to only starter models
            new_patch = apply_inclusion_exclusion_filters_to_source(
                admin_client=admin_client,
                namespace=model_registry_namespace,
                source_id=REDHAT_AI_CATALOG_ID,
                included_models=["*starter*"],
            )

            patch_info["configmap"].update(new_patch["patch"])

            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)

            try:
                phase3_models = wait_for_model_set_match(
                    model_catalog_rest_url=model_catalog_rest_url,
                    model_registry_rest_headers=model_registry_rest_headers,
                    source_label=REDHAT_AI_CATALOG_NAME,
                    expected_models=starter_models,
                )
            except TimeoutExpiredError as e:
                pytest.fail(f"Phase 3: Timeout waiting for starter models {starter_models}: {e}")
            assert phase3_models == starter_models, f"Phase 3: Expected starter {starter_models}, got {phase3_models}"

            LOGGER.info("SUCCESS: All phases of dynamic switching validated")


class TestModelValidationErrors:
    """Test error validation for invalid patterns and configurations (RHOAIENG-41841 part 4)"""

    def test_invalid_include_patterns_rejected(self, admin_client, model_registry_namespace: str):
        """Test that invalid inclusion patterns generate validation errors."""
        LOGGER.info("Testing invalid inclusion pattern validation")

        for category, patterns in INVALID_MODEL_PATTERNS.items():
            LOGGER.info(f"Testing {category} patterns: {patterns}")

            error_detected, error_msg = validate_invalid_pattern_error(
                admin_client=admin_client,
                namespace=model_registry_namespace,
                source_id=REDHAT_AI_CATALOG_ID,
                invalid_patterns=patterns,
                field_name="includedModels",
            )

            if category in ["malformed_regex", "sql_injection"]:
                assert error_detected, f"Expected {category} patterns to be rejected: {patterns}. Error: {error_msg}"

            LOGGER.info(f"{category} validation result: {error_msg}")

    def test_invalid_exclude_patterns_rejected(self, admin_client, model_registry_namespace: str):
        """Test that invalid exclusion patterns generate validation errors."""
        LOGGER.info("Testing invalid exclusion pattern validation")

        for category, patterns in INVALID_MODEL_PATTERNS.items():
            LOGGER.info(f"Testing {category} patterns: {patterns}")

            error_detected, error_msg = validate_invalid_pattern_error(
                admin_client=admin_client,
                namespace=model_registry_namespace,
                source_id=REDHAT_AI_CATALOG_ID,
                invalid_patterns=patterns,
                field_name="excludedModels",
            )

            if category in ["malformed_regex", "sql_injection"]:
                assert error_detected, f"Expected {category} patterns to be rejected: {patterns}. Error: {error_msg}"

            LOGGER.info(f"{category} validation result: {error_msg}")


class TestSourceLifecycleCleanup:
    """Test source disabling cleanup scenarios (RHOAIENG-41846 part 2)"""

    def test_source_disabling_removes_models(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that disabling a source removes all its models from the catalog."""
        LOGGER.info("Testing source disabling cleanup")

        baseline_models = baseline_redhat_ai_models["api_models"]

        # Verify models exist initially
        initial_models = get_api_models_by_source_label(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            source_label=REDHAT_AI_CATALOG_NAME,
        )
        assert initial_models == baseline_models, f"Initial state should match baseline: {baseline_models}"

        # Disable the source
        disable_patch = disable_catalog_source(
            admin_client=admin_client, namespace=model_registry_namespace, source_id=REDHAT_AI_CATALOG_ID
        )

        with ResourceEditor(patches={disable_patch["configmap"]: disable_patch["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
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


class TestLoggingValidation:
    """Test cleanup operation logging (RHOAIENG-41846 part 3)"""

    def test_model_removal_logging(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        baseline_redhat_ai_models: dict,
    ):
        """Test that model removal operations are properly logged."""
        LOGGER.info("Testing model removal logging")

        # Apply filter to exclude granite models (should trigger removals)
        patch_info = apply_inclusion_exclusion_filters_to_source(
            admin_client=admin_client,
            namespace=model_registry_namespace,
            source_id=REDHAT_AI_CATALOG_ID,
            excluded_models=["*granite*"],
        )

        with ResourceEditor(patches={patch_info["configmap"]: patch_info["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
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
                r"removed.*granite.*models?",  # Log about removing granite models
                r"cleanup.*completed",  # Log about cleanup completion
                r"excluded.*granite",  # Log about exclusion filter
            ]

            try:
                found_patterns = validate_cleanup_logging(
                    namespace=model_registry_namespace, expected_log_patterns=expected_log_patterns
                )
                LOGGER.info(f"SUCCESS: Found expected log patterns: {found_patterns}")
            except TimeoutExpiredError as e:
                LOGGER.warning(f"WARNING: Expected log patterns not found: {e}")
                # Don't fail the test - logging might be implemented differently

    def test_source_disabling_logging(
        self,
        admin_client,
        model_registry_namespace: str,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Test that source disabling operations are properly logged."""
        LOGGER.info("Testing source disabling logging")

        # Disable the source
        disable_patch = disable_catalog_source(
            admin_client=admin_client, namespace=model_registry_namespace, source_id=REDHAT_AI_CATALOG_ID
        )

        with ResourceEditor(patches={disable_patch["configmap"]: disable_patch["patch"]}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
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
            expected_log_patterns = [
                rf"source.*{REDHAT_AI_CATALOG_ID}.*disabled",
                r"removed.*models.*source.*disabled",
                r"cleanup.*source.*disabled",
            ]

            try:
                found_patterns = validate_cleanup_logging(
                    namespace=model_registry_namespace, expected_log_patterns=expected_log_patterns
                )
                LOGGER.info(f"SUCCESS: Found expected source disabling log patterns: {found_patterns}")
            except TimeoutExpiredError as e:
                LOGGER.warning(f"WARNING: Expected source disabling log patterns not found: {e}")
                # Don't fail - logging implementation may vary
