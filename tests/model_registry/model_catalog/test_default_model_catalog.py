import json

import pytest
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.constants import DEFAULT_MODEL_CATALOG, CUSTOM_MODEL_CATALOG
from ocp_resources.config_map import ConfigMap

LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "catalog_config_map, expected_models",
    [
        pytest.param(DEFAULT_MODEL_CATALOG, True),
        pytest.param(CUSTOM_MODEL_CATALOG, False),
    ],
    indirect=["catalog_config_map"],
)
def test_config_map(catalog_config_map: ConfigMap, expected_models):
    # Check that the default configmaps exists in the application namespace.
    # Managed configmap contains default model sources while the unmanaged one is empty by default
    assert catalog_config_map.exists, f"{catalog_config_map.name} does not exist"
    models = json.loads(catalog_config_map.instance.data.modelCatalogSources)["sources"]
    assert bool(models) == expected_models, f"Expected models presence: {expected_models}, actual: {models}"
