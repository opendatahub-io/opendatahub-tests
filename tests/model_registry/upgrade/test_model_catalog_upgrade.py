import pytest
import yaml
from simple_logger.logger import get_logger
from typing import Self, Generator
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from tests.model_registry.constants import SAMPLE_MODEL_NAME1, CUSTOM_CATALOG_ID1
from tests.model_registry.utils import (
    is_model_catalog_ready,
    wait_for_model_catalog_api,
    get_catalog_str,
    get_sample_yaml_str,
)

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


@pytest.fixture(scope="class")
def pre_upgrade_config_map_update(
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> ConfigMap:
    """Fixture for updating catalog config map before upgrade"""
    patches = {"data": {"sources.yaml": request.param["sources_yaml"]}}
    if "sample_yaml" in request.param:
        for key in request.param["sample_yaml"]:
            patches["data"][key] = request.param["sample_yaml"][key]

    ResourceEditor(patches=patches).update()
    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
    return catalog_config_map


@pytest.fixture(scope="class")
def post_upgrade_config_map_update(
    catalog_config_map: ConfigMap,
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Generator[ConfigMap, None, None]:
    """Fixture for updating catalog config map after post upgrade testing is done"""
    yield catalog_config_map
    # Only teardown is needed
    catalog_config_map.delete()
    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)


@pytest.mark.parametrize(
    "pre_upgrade_config_map_update",
    [
        pytest.param(
            {
                "sources_yaml": get_catalog_str(ids=[CUSTOM_CATALOG_ID1]),
                "sample_yaml": {"sample-custom-catalog1.yaml": get_sample_yaml_str(models=[SAMPLE_MODEL_NAME1])},
            },
            id="test_file_test_catalog",
        ),
    ],
    indirect=["pre_upgrade_config_map_update"],
)
class TestPreUpgradeModelCatalog:
    """Test class for model catalog functionality before upgrade"""

    @pytest.mark.order("first")
    @pytest.mark.pre_upgrade
    def test_validate_sources(
        self: Self,
        pre_upgrade_config_map_update: ConfigMap,
    ):
        # check that the configmap was updated:
        assert len(yaml.safe_load(pre_upgrade_config_map_update.instance.data["sources.yaml"])["catalogs"]) == 2
        LOGGER.info("Testing model catalog validation")


@pytest.mark.usefixtures("post_upgrade_config_map_update")
class TestPostUpgradeModelCatalog:
    @pytest.mark.order("last")
    @pytest.mark.post_upgrade
    def test_validate_sources(
        self: Self,
        post_upgrade_config_map_update: ConfigMap,
    ):
        # check that the configmap was updated:
        assert len(yaml.safe_load(post_upgrade_config_map_update.instance.data["sources.yaml"])["catalogs"]) == 1
        LOGGER.info("Testing model catalog validation")
