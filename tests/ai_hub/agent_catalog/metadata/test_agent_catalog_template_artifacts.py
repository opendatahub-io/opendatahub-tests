from typing import Any, Self

import pytest
import structlog

LOGGER = structlog.get_logger(name=__name__)

REQUIRED_TEMPLATE_FIELDS: set[str] = {"name", "displayName", "description", "env", "framework", "labels", "logo"}
TEMPLATE_AGENT_MATCHED_FIELDS: tuple[str, ...] = ("description", "displayName", "framework", "labels", "logo")

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


class TestAgentTemplates:
    """Tests for agent catalog template artifacts."""

    def test_agents_have_template_artifacts(
        self: Self,
        default_agents: list[dict[str, Any]],
        agent_template_artifacts: dict[str, dict],
    ) -> None:
        """Given agents exist in the default catalog
        When fetching template artifacts for each agent
        Then every agent has a template-artifact named agent.yaml
        """
        missing = [
            agent.get("name", "<unknown>")
            for agent in default_agents
            if agent.get("name") not in agent_template_artifacts
        ]
        assert not missing, f"Agents missing 'agent.yaml' template artifact: {missing}"

    def test_agent_template_content_has_required_fields(
        self: Self,
        agent_template_artifacts: dict[str, dict],
    ) -> None:
        """Given agents have template artifacts
        When inspecting the parsed template content
        Then every template has all required fields
        """
        errors: list[str] = []
        for name, content in agent_template_artifacts.items():
            missing = REQUIRED_TEMPLATE_FIELDS - set(content.keys())
            if missing:
                errors.append(f"Agent '{name}' template content missing fields: {missing}")
        assert not errors, "Template content validation failed:\n" + "\n".join(errors)

    def test_agent_template_content_matches_agent_metadata(
        self: Self,
        default_agents: list[dict[str, Any]],
        agent_template_artifacts: dict[str, dict],
    ) -> None:
        """Given agents have template artifacts
        When comparing template content with agent top-level fields
        Then description, displayName, framework, labels, and logo are consistent
        """
        api_agents = {agent["name"]: agent for agent in default_agents}
        errors: list[str] = []
        for name, content in agent_template_artifacts.items():
            agent = api_agents[name]

            for field in TEMPLATE_AGENT_MATCHED_FIELDS:
                template_value = content.get(field)
                agent_value = agent.get(field)
                if template_value != agent_value:
                    errors.append(
                        f"Agent '{name}' template {field} mismatch: template={template_value!r}, agent={agent_value!r}"
                    )

        assert not errors, "Template/agent metadata mismatch:\n" + "\n".join(errors)
