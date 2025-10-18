from typing import Any

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger


from ocp_resources.pod import Pod
from tests.model_registry.model_catalog.constants import (
    DEFAULT_CATALOG_NAME,
    DEFAULT_CATALOG_ID,
    CATALOG_TYPE,
    DEFAULT_CATALOG_FILE,
)

LOGGER = get_logger(name=__name__)


def validate_model_catalog_enabled(pod: Pod) -> bool:
    for container in pod.instance.spec.containers:
        for env in container.env:
            if env.name == "ENABLE_MODEL_CATALOG":
                return True
    return False


def validate_model_catalog_resource(kind: Any, admin_client: DynamicClient, namespace: str) -> None:
    resource = list(kind.get(namespace=namespace, label_selector="component=model-catalog", dyn_client=admin_client))
    assert resource
    assert len(resource) == 1, f"Unexpected number of {kind} resources found: {[res.name for res in resource]}"


def validate_default_catalog(default_catalog) -> None:
    assert default_catalog["name"] == DEFAULT_CATALOG_NAME
    assert default_catalog["id"] == DEFAULT_CATALOG_ID
    assert default_catalog["type"] == CATALOG_TYPE
    assert default_catalog["properties"].get("yamlCatalogPath") == DEFAULT_CATALOG_FILE


def get_validate_default_model_catalog_source(catalogs: list[dict[Any, Any]]) -> None:
    assert len(catalogs) == 1, f"Expected no custom models to be present. Actual: {catalogs}"
    assert catalogs[0]["id"] == DEFAULT_CATALOG_ID
    assert catalogs[0]["name"] == DEFAULT_CATALOG_NAME
    assert str(catalogs[0]["enabled"]) == "True", catalogs[0]["enabled"]
