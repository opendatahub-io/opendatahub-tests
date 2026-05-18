import time
from typing import Self

import pytest
import structlog
from ocp_resources.config_map import ConfigMap

from tests.model_registry.constants import CUSTOM_CATALOG_ID1
from tests.model_registry.model_catalog.utils import get_catalog_str, get_models_from_catalog_api
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)
pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


MODEL_WITH_SERVING_CONFIG = "test/model-with-serving-config"
MODEL_WITHOUT_SERVING_CONFIG = "test/model-without-serving-config"

EXPECTED_VALIDATED_TASKS = ["text-generation", "tool-calling"]
EXPECTED_TOOL_CALL_PARSER = "granite"
EXPECTED_CHAT_TEMPLATE = "opt/app-root/template/tool_chat_template_granite.jinja"
EXPECTED_REQUIRED_ARGS = ["--config_format granite"]

VALIDATED_TASKS_UNIQUE_TERM = "tool-calling"

_CURRENT_TIME = int(time.time() * 1000)
CUSTOM_YAML_WITH_SERVING_CONFIG: str = f"""source: Custom Test
models:
- name: {MODEL_WITH_SERVING_CONFIG}
  description: Test model with servingConfig and validatedTasks.
  readme: |-
    # Test model with serving config
  provider: Test Provider
  logo: placeholder
  license: apache-2.0
  licenseLink: https://www.apache.org/licenses/LICENSE-2.0.txt
  tasks:
    - text-generation
  validatedTasks:
    - text-generation
    - tool-calling
  servingConfig:
    toolCalling:
      toolCallParser: {EXPECTED_TOOL_CALL_PARSER}
      chatTemplate: {EXPECTED_CHAT_TEMPLATE}
      enableAutoToolChoice: true
      requiredArgs:
        - "--config_format granite"
  artifacts:
    - uri: oci://registry.example.io/test-serving:1.0
  createTimeSinceEpoch: "{_CURRENT_TIME - 10000!s}"
  lastUpdateTimeSinceEpoch: "{_CURRENT_TIME!s}"

- name: {MODEL_WITHOUT_SERVING_CONFIG}
  description: Plain test model without servingConfig or validatedTasks.
  readme: |-
    # Plain test model
  provider: Test Provider
  logo: placeholder
  license: apache-2.0
  licenseLink: https://www.apache.org/licenses/LICENSE-2.0.txt
  tasks:
    - text-generation
  artifacts:
    - uri: oci://registry.example.io/test-plain:1.0
  createTimeSinceEpoch: "{_CURRENT_TIME - 10000!s}"
  lastUpdateTimeSinceEpoch: "{_CURRENT_TIME!s}"
"""


@pytest.mark.parametrize(
    "updated_catalog_config_map",
    [
        pytest.param(
            {
                "sources_yaml": get_catalog_str(ids=[CUSTOM_CATALOG_ID1]),
                "sample_yaml": {
                    "sample-custom-catalog1.yaml": CUSTOM_YAML_WITH_SERVING_CONFIG,
                },
            },
            id="test_serving_config_validated_tasks",
        ),
    ],
    indirect=["updated_catalog_config_map"],
)
@pytest.mark.usefixtures("model_registry_namespace")
@pytest.mark.jira("RHOAIENG-60668")
@pytest.mark.tier1
class TestServingConfigAndValidatedTasks:
    """Tests for validatedTasks and servingConfig fields in custom catalog models (RHOAIENG-60668)."""

    def test_model_with_serving_config_and_validated_tasks(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Given a custom catalog model with validatedTasks and servingConfig
        When querying the model via the catalog API
        Then the response includes both fields with correct values
        """
        model = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{CUSTOM_CATALOG_ID1}/models/{MODEL_WITH_SERVING_CONFIG}",
            headers=model_registry_rest_headers,
        )

        validated_tasks = model.get("validatedTasks", [])
        assert sorted(validated_tasks) == sorted(EXPECTED_VALIDATED_TASKS), (
            f"validatedTasks mismatch: expected {EXPECTED_VALIDATED_TASKS}, got {validated_tasks}"
        )

        serving_config = model.get("servingConfig")
        assert serving_config, f"servingConfig should be present for model '{MODEL_WITH_SERVING_CONFIG}'"

        tool_calling = serving_config.get("toolCalling")
        assert tool_calling, "servingConfig.toolCalling should be present"
        assert tool_calling.get("toolCallParser") == EXPECTED_TOOL_CALL_PARSER
        assert tool_calling.get("chatTemplate") == EXPECTED_CHAT_TEMPLATE
        assert tool_calling.get("enableAutoToolChoice") is True

        required_args = tool_calling.get("requiredArgs")
        assert isinstance(required_args, list), f"requiredArgs should be a list, got: {type(required_args)}"
        assert required_args == EXPECTED_REQUIRED_ARGS

    def test_model_without_serving_config_and_validated_tasks(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Given a custom catalog model without validatedTasks or servingConfig
        When querying the model via the catalog API
        Then the response omits servingConfig and has empty validatedTasks
        """
        model = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{CUSTOM_CATALOG_ID1}/models/{MODEL_WITHOUT_SERVING_CONFIG}",
            headers=model_registry_rest_headers,
        )

        assert model.get("servingConfig") is None, (
            f"servingConfig should be absent for model '{MODEL_WITHOUT_SERVING_CONFIG}'"
        )

        validated_tasks = model.get("validatedTasks", [])
        assert not validated_tasks, (
            f"validatedTasks should be empty for model '{MODEL_WITHOUT_SERVING_CONFIG}', got {validated_tasks}"
        )

    def test_serving_config_excluded_from_filter_options(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Given the model catalog API with a custom catalog loaded
        When querying the filter_options endpoint
        Then serving_config should not appear in the filterable properties
        """
        sources = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources",
            headers=model_registry_rest_headers,
        )
        source_ids = {source["id"] for source in sources.get("items", [])}
        assert CUSTOM_CATALOG_ID1 in source_ids, (
            f"Custom catalog source '{CUSTOM_CATALOG_ID1}' not found — catalog may not have loaded yet"
        )

        response = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models/filter_options",
            headers=model_registry_rest_headers,
        )
        filter_keys = set(response.get("filters", {}).keys())
        assert "serving_config" not in filter_keys, "serving_config should be excluded from filter_options"

    def test_validated_tasks_searchable_via_q_parameter(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """Given a custom catalog model with validatedTasks containing 'tool-calling' but tasks without it
        When searching with q=tool-calling scoped to the custom source
        Then the model is returned because validated_tasks is included in the search index
        """
        response = get_models_from_catalog_api(
            model_catalog_rest_url=model_catalog_rest_url,
            model_registry_rest_headers=model_registry_rest_headers,
            q=VALIDATED_TASKS_UNIQUE_TERM,
            additional_params=f"&source={CUSTOM_CATALOG_ID1}",
        )
        model_names = [model["name"] for model in response.get("items", [])]
        assert MODEL_WITH_SERVING_CONFIG in model_names, (
            f"Expected '{MODEL_WITH_SERVING_CONFIG}' in search results for '{VALIDATED_TASKS_UNIQUE_TERM}'. "
            f"The backend search should match validated_tasks (not just tasks). Got: {model_names}"
        )
