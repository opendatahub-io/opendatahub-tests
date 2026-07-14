from typing import Self

import pytest
import structlog
import yaml
from ocp_resources.config_map import ConfigMap

from tests.ai_hub.agent_catalog.config.constants import (
    AGENT_SOURCES_CATALOGS_PATH_ARG,
    AGENT_SOURCES_VOLUME_MOUNT_PATH,
    EXPECTED_AGENT_LABEL_DEFINITION,
    EXPECTED_DEFAULT_AGENT_CATALOG,
)
from tests.ai_hub.utils import execute_get_command_with_retry

LOGGER = structlog.get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures("updated_dsc_component_state_scope_session", "model_registry_namespace"),
]


@pytest.mark.tier1
class TestDefaultAgentCatalogConfig:
    """Tests for agent catalog operator wiring (RHOAIENG-70685)."""

    def test_agent_catalog_configmap_exists(
        self: Self,
        agent_catalog_configmap: ConfigMap,
    ) -> None:
        """Given the model registry operator has reconciled
        When checking the model registries namespace
        Then the agent-catalog-sources ConfigMap exists with empty agent_catalogs
        """
        assert agent_catalog_configmap.exists, f"{agent_catalog_configmap.name} does not exist"
        data = yaml.safe_load(agent_catalog_configmap.instance.data.get("sources.yaml", "{}") or "{}")
        agent_catalogs = data.get("agent_catalogs", [])
        assert agent_catalogs == [], f"Expected empty agent_catalogs in user ConfigMap, got {agent_catalogs}"

    def test_default_agent_catalog_entry(
        self: Self,
        default_agent_catalogs: list[dict],
    ) -> None:
        """Given the operator has reconciled
        When reading default-catalog-sources ConfigMap
        Then it contains the expected rh_agents entry
        """
        matching = [
            entry for entry in default_agent_catalogs if entry.get("id") == EXPECTED_DEFAULT_AGENT_CATALOG["id"]
        ]
        assert len(matching) == 1, (
            f"Expected exactly 1 agent_catalogs entry with id '{EXPECTED_DEFAULT_AGENT_CATALOG['id']}', "
            f"found {len(matching)}: {matching}"
        )
        assert matching[0] == EXPECTED_DEFAULT_AGENT_CATALOG, (
            f"Agent catalog entry does not match expected values.\nExpected: {EXPECTED_DEFAULT_AGENT_CATALOG}\n"
            f"Actual: {matching[0]}"
        )

    def test_default_agent_label_definition(
        self: Self,
        default_agent_label_definitions: list[dict],
    ) -> None:
        """Given the operator has reconciled
        When reading labels in default-catalog-sources ConfigMap
        Then an agent label definition with assetType agents exists
        """
        matching = [
            label
            for label in default_agent_label_definitions
            if label.get("name") == EXPECTED_AGENT_LABEL_DEFINITION["name"]
        ]
        assert len(matching) == 1, (
            f"Expected exactly 1 label definition with name '{EXPECTED_AGENT_LABEL_DEFINITION['name']}', "
            f"found {len(matching)}: {matching}"
        )
        assert matching[0] == EXPECTED_AGENT_LABEL_DEFINITION, (
            f"Label definition does not match expected values.\nExpected: {EXPECTED_AGENT_LABEL_DEFINITION}\n"
            f"Actual: {matching[0]}"
        )

    def test_catalog_deployment_has_agent_sources_volume_mount(
        self: Self,
        catalog_container_spec: object,
    ) -> None:
        """Given the catalog deployment is running
        When inspecting the catalog container spec
        Then /data/user-agent-sources is mounted from the agent-catalog-sources ConfigMap
        """
        mount_paths = [mount.mountPath for mount in (catalog_container_spec.volumeMounts or [])]
        assert AGENT_SOURCES_VOLUME_MOUNT_PATH in mount_paths, (
            f"Expected volume mount '{AGENT_SOURCES_VOLUME_MOUNT_PATH}' not found. Mounts: {mount_paths}"
        )

    def test_catalog_deployment_has_agent_catalogs_path_arg(
        self: Self,
        catalog_container_spec: object,
    ) -> None:
        """Given the catalog deployment is running
        When inspecting the catalog container args
        Then --catalogs-path includes the agent sources path
        """
        args = catalog_container_spec.args or []
        assert AGENT_SOURCES_CATALOGS_PATH_ARG in args, (
            f"Expected arg '{AGENT_SOURCES_CATALOGS_PATH_ARG}' not found. Args: {args}"
        )


@pytest.mark.tier1
class TestDefaultAgentCatalogApi:
    """Tests for default agent catalog API behavior (RHOAIENG-70685)."""

    def test_default_agent_source_available(
        self: Self,
        model_catalog_api_url: str,
        model_registry_rest_headers: dict[str, str],
    ) -> None:
        """Given the default agent catalog source is configured
        When requesting GET /sources?assetType=agents
        Then rh_agents source appears with status available
        """
        response = execute_get_command_with_retry(
            url=f"{model_catalog_api_url}sources",
            headers=model_registry_rest_headers,
            params={"assetType": "agents"},
        )
        items = response.get("items", [])
        source_ids = {item["id"] for item in items}
        assert EXPECTED_DEFAULT_AGENT_CATALOG["id"] in source_ids, (
            f"Expected agent source '{EXPECTED_DEFAULT_AGENT_CATALOG['id']}' in {source_ids}"
        )

        rh_agents_source = next(item for item in items if item["id"] == EXPECTED_DEFAULT_AGENT_CATALOG["id"])
        assert rh_agents_source.get("status") == "available", (
            f"Expected agent source status 'available', got '{rh_agents_source.get('status')}'"
        )
