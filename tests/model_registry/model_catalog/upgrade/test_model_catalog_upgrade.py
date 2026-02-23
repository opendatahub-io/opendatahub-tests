from collections.abc import Generator
from typing import Self

import pytest
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger

from tests.model_registry.constants import CUSTOM_CATALOG_ID1, SAMPLE_MODEL_NAME1
from tests.model_registry.utils import (
    get_catalog_str,
    get_sample_yaml_str,
    wait_for_model_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
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

    ResourceEditor(patches={catalog_config_map: patches}).update()
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
    return catalog_config_map


@pytest.fixture(scope="class")
def post_upgrade_config_map_update(
    catalog_config_map: ConfigMap,
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Generator[ConfigMap]:
    """Fixture for updating catalog config map after post upgrade testing is done"""
    yield catalog_config_map
    # Only teardown is needed
    catalog_config_map.delete()
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )


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
        # check that the custom source configmap was updated:
        assert len(yaml.safe_load(pre_upgrade_config_map_update.instance.data["sources.yaml"])["catalogs"]) == 1
        LOGGER.info("Testing model catalog validation")


@pytest.mark.usefixtures("post_upgrade_config_map_update")
class TestPostUpgradeModelCatalog:
    @pytest.mark.order("last")
    @pytest.mark.post_upgrade
    def test_validate_sources(
        self: Self,
        post_upgrade_config_map_update: ConfigMap,
    ):
        # check that the configmap was still updated:
        assert len(yaml.safe_load(post_upgrade_config_map_update.instance.data["sources.yaml"])["catalogs"]) == 1
        LOGGER.info("Testing model catalog validation")
