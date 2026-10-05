from typing import Any

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.deployment import Deployment

from tests.ai_hub.agent_catalog.config.constants import AGENT_CATALOG_SOURCES_CM
from tests.ai_hub.constants import CATALOG_CONTAINER, DEFAULT_MODEL_CATALOG_CM
from tests.ai_hub.model_catalog.constants import MODEL_CATALOG_DEPLOYMENT_NAME


@pytest.fixture(scope="class")
def default_agent_catalog_sources_data(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> dict:
    """Return the parsed sources.yaml data from the default catalog sources ConfigMap."""
    configmap = ConfigMap(
        name=DEFAULT_MODEL_CATALOG_CM,
        client=admin_client,
        namespace=model_registry_namespace,
    )
    return yaml.safe_load(configmap.instance.data.get("sources.yaml", "{}") or "{}")


@pytest.fixture(scope="class")
def default_agent_catalogs(default_agent_catalog_sources_data: dict) -> list[dict]:
    """Return the agent_catalogs list from the default catalog sources ConfigMap."""
    return default_agent_catalog_sources_data.get("agent_catalogs", [])


@pytest.fixture(scope="class")
def default_agent_label_definitions(default_agent_catalog_sources_data: dict) -> list[dict]:
    """Return agent label definitions from the default catalog sources ConfigMap."""
    return [
        label for label in default_agent_catalog_sources_data.get("labels", []) if label.get("assetType") == "agents"
    ]


@pytest.fixture(scope="class")
def agent_catalog_configmap(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> ConfigMap:
    """Return the user-editable agent-catalog-sources ConfigMap."""
    return ConfigMap(
        name=AGENT_CATALOG_SOURCES_CM,
        client=admin_client,
        namespace=model_registry_namespace,
    )


@pytest.fixture(scope="class")
def catalog_container_spec(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Any:
    """Return the catalog container spec from the model-catalog deployment."""
    deployment = Deployment(
        name=MODEL_CATALOG_DEPLOYMENT_NAME,
        namespace=model_registry_namespace,
        client=admin_client,
        ensure_exists=True,
    )
    container = next(
        (
            container
            for container in deployment.instance.spec.template.spec.containers
            if container.name == CATALOG_CONTAINER
        ),
        None,
    )
    assert container, f"Container '{CATALOG_CONTAINER}' not found in deployment"
    return container
