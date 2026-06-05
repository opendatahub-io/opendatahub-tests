from typing import Self

import pytest
import structlog
from ocp_resources.config_map import ConfigMap

from tests.ai_hub.constants import CUSTOM_CATALOG_ID1
from tests.ai_hub.model_catalog.catalog_config.constants import (
    CUSTOM_YAML_WITH_HARDWARE_TAGS,
    MODEL_WITH_EMPTY_TAG,
    MODEL_WITH_SINGLE_TAG,
    MODEL_WITHOUT_TAG,
    SINGLE_HW_TAG,
)
from tests.ai_hub.model_catalog.utils import get_catalog_str
from tests.ai_hub.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)


@pytest.mark.parametrize(
    "updated_catalog_config_map",
    [
        pytest.param(
            {
                "sources_yaml": get_catalog_str(ids=[CUSTOM_CATALOG_ID1]),
                "sample_yaml": {
                    "sample-custom-catalog1.yaml": CUSTOM_YAML_WITH_HARDWARE_TAGS,
                },
            },
            id="test_hardware_tag_custom_catalog",
        ),
    ],
    indirect=["updated_catalog_config_map"],
)
@pytest.mark.usefixtures("model_registry_namespace")
@pytest.mark.tier1
class TestHardwareTag:
    """Tests for hardware_tag custom property on model cards (RHOAIENG-65783)."""

    def test_single_hardware_tag(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
    ):
        """
        Given a custom catalog model with a single hardware_tag
        When querying the model via the catalog API
        Then the hardware_tag custom property is present with the correct value
        """
        model = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{CUSTOM_CATALOG_ID1}/models/{MODEL_WITH_SINGLE_TAG}",
            headers=model_registry_rest_headers,
        )
        custom_props = model.get("customProperties", {})
        assert "hardware_tag" in custom_props, f"hardware_tag missing from customProperties: {custom_props}"
        assert custom_props["hardware_tag"]["string_value"] == SINGLE_HW_TAG

    @pytest.mark.parametrize(
        "model_name",
        [
            pytest.param(MODEL_WITH_EMPTY_TAG, id="test_empty_hardware_tag_is_dropped"),
            pytest.param(MODEL_WITHOUT_TAG, id="test_missing_hardware_tag"),
        ],
    )
    def test_hardware_tag_absent(
        self: Self,
        updated_catalog_config_map: ConfigMap,
        model_catalog_rest_url: list[str],
        model_registry_rest_headers: dict[str, str],
        model_name: str,
    ):
        """
        Given a custom catalog model with an empty or missing hardware_tag
        When querying the model via the catalog API
        Then the hardware_tag custom property is absent
        """
        model = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{CUSTOM_CATALOG_ID1}/models/{model_name}",
            headers=model_registry_rest_headers,
        )
        custom_props = model.get("customProperties", {})
        assert "hardware_tag" not in custom_props, (
            f"hardware_tag should not exist for '{model_name}' but found: {custom_props.get('hardware_tag')}"
        )
