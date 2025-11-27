import pytest
import yaml
from simple_logger.logger import get_logger
from typing import Self

from ocp_resources.config_map import ConfigMap
from tests.model_registry.constants import CUSTOM_CATALOG_ID1, SAMPLE_MODEL_NAME1
from tests.model_registry.utils import get_sample_yaml_str, get_catalog_str

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


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
