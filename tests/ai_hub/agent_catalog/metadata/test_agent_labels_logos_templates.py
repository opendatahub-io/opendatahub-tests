from typing import Any, Self

import pytest
import structlog

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


class TestDefaultAgentFieldValidation:
    """Tests for default agent catalog field validation."""

    @pytest.mark.parametrize(
        "field",
        [
            pytest.param("labels", id="test_all_agents_have_labels", marks=pytest.mark.smoke),
            pytest.param("logo", id="test_all_agents_have_logos"),
        ],
    )
    def test_all_agents_have_field(
        self: Self,
        default_agents: list[dict[str, Any]],
        field: str,
    ) -> None:
        """Given agents exist in the default catalog When inspecting each agent
        Then every agent has a non-empty value for the given field
        """
        missing = [agent.get("name", "<unknown>") for agent in default_agents if not agent.get(field)]
        assert not missing, f"Agents missing '{field}': {missing}"

    def test_agent_labels_match_yaml(
        self: Self,
        default_agents: list[dict[str, Any]],
        default_agents_yaml_content: dict[str, Any],
    ) -> None:
        """Given agents are defined in the catalog YAML on the pod
        When comparing each agent's labels with the API response
        Then every agent's labels match the YAML source
        """
        yaml_labels = {
            agent["name"]: set(agent.get("labels", [])) for agent in default_agents_yaml_content.get("agents", [])
        }
        errors: list[str] = []
        for agent in default_agents:
            name = agent.get("name", "<unknown>")
            expected = yaml_labels.get(name)
            if expected is None:
                continue
            actual = set(agent.get("labels", []))
            if actual != expected:
                errors.append(f"Agent '{name}': expected={expected}, actual={actual}")
        assert not errors, "Label validation against YAML failed:\n" + "\n".join(errors)
