"""Operator and packaged metadata integration coverage for RHOAIENG-97046."""

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod

from tests.ai_hub.constants import CATALOG_CONTAINER, DEFAULT_MODEL_CATALOG_CM
from tests.ai_hub.model_catalog.utils import get_shipped_catalog
from tests.ai_hub.serving_runtime_catalog.constants import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_LABEL,
    DEFAULT_SOURCE_ID,
    USER_CONFIGMAP,
    USER_MOUNT_PATH,
)
from tests.ai_hub.serving_runtime_catalog.utils import (
    list_runtime_sources,
    wait_for_runtime_data,
    wait_for_source_status,
)

pytestmark = [pytest.mark.tier1, pytest.mark.usefixtures("updated_dsc_component_state_scope_session")]


class TestDefaultRuntimeCatalog:
    """Verify metadata image, operator configuration, and public API integration (RHOAIENG-96066)."""

    def test_operator_source_configuration(
        self,
        admin_client: DynamicClient,
        model_registry_namespace: str,
        runtime_source_configmap: ConfigMap,
        model_catalog_pod: Pod,
    ) -> None:
        """Given the installed catalog, when its configuration is read, then runtime sources are wired correctly."""
        configmap = ConfigMap(client=admin_client, namespace=model_registry_namespace, name=DEFAULT_MODEL_CATALOG_CM)
        config = yaml.safe_load(configmap.instance.data["sources.yaml"])
        source = next(source for source in config["serving_runtime_catalogs"] if source["id"] == DEFAULT_SOURCE_ID)
        assert source["properties"]["yamlCatalogPath"] == DEFAULT_CATALOG_PATH
        assert source["labels"] == [DEFAULT_LABEL]
        assert source.get("enabled", True)
        assert any(
            label["name"] == DEFAULT_LABEL and label["assetType"] == "serving_runtimes" for label in config["labels"]
        )
        user_config = yaml.safe_load(runtime_source_configmap.instance.data["sources.yaml"])
        assert isinstance(user_config["serving_runtime_catalogs"], list)
        assert not runtime_source_configmap.instance.metadata.get("ownerReferences")
        spec = model_catalog_pod.instance.to_dict()["spec"]
        volume = next(volume for volume in spec["volumes"] if volume["name"] == "user-serving-runtime-sources")
        assert volume["configMap"]["name"] == USER_CONFIGMAP
        container = next(container for container in spec["containers"] if container["name"] == CATALOG_CONTAINER)
        assert {"name": volume["name"], "mountPath": USER_MOUNT_PATH}.items() <= next(
            mount for mount in container["volumeMounts"] if mount["name"] == volume["name"]
        ).items()
        assert f"--catalogs-path={USER_MOUNT_PATH}/sources.yaml" in container["args"]

    def test_packaged_runtimes_and_versions(
        self,
        model_catalog_pod: Pod,
        runtime_api_url: str,
        runtime_source_url: str,
        runtime_rest_headers: dict[str, str],
    ) -> None:
        """Given packaged metadata, when queried, then source metadata, runtimes, and versions are preserved."""
        document = get_shipped_catalog(pod=model_catalog_pod, catalog_file=DEFAULT_CATALOG_PATH)
        assert document["serving_runtimes"], "Packaged serving runtime catalog is empty"
        assert all(runtime["versions"] for runtime in document["serving_runtimes"]), "Packaged runtime has no versions"
        wait_for_source_status(
            source_url=runtime_source_url,
            headers=runtime_rest_headers,
            source_id=DEFAULT_SOURCE_ID,
            expected="available",
        )
        sources = list_runtime_sources(source_url=runtime_source_url, headers=runtime_rest_headers)
        source = next(source for source in sources if source["id"] == DEFAULT_SOURCE_ID)
        assert source["assetType"] == "serving_runtimes"
        assert source["enabled"] is True
        assert source["labels"] == [DEFAULT_LABEL]
        assert source["status"] == "available"
        assert not source.get("error"), "Default runtime source listing has a load error"
        wait_for_runtime_data(url=runtime_api_url, headers=runtime_rest_headers, catalogs={DEFAULT_SOURCE_ID: document})
