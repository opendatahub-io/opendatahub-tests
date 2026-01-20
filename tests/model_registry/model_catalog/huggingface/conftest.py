import pytest
import time
from huggingface_hub import HfApi
from simple_logger.logger import get_logger

from tests.model_registry.model_catalog.huggingface.utils import get_huggingface_model_from_api

LOGGER = get_logger(name=__name__)


@pytest.fixture()
def huggingface_api():
    return HfApi()


@pytest.fixture()
def num_models_from_hf_api_with_matching_criteria(request: pytest.FixtureRequest, huggingface_api: HfApi) -> int:
    excluded_str = request.param.get("excluded_str")
    included_str = request.param.get("included_str")
    models = huggingface_api.list_models(author=request.param["org_name"], limit=10000)
    model_list = []
    for model in models:
        if excluded_str:
            if model.id.endswith(excluded_str):
                LOGGER.info(f"Skipping {model.id} due to {excluded_str}")
                continue
            else:
                LOGGER.info(f"Adding {model.id}")
                model_list.append(model.id)
        elif included_str:
            if model.id.startswith(included_str):
                LOGGER.info(f"Adding {model.id}")
                model_list.append(model.id)
            else:
                LOGGER.info(f"Skipping {model.id} due to {included_str}")
                continue
        else:
            model_list.append(model.id)
    return len(model_list)


@pytest.fixture(scope="module")
def epoch_time_before_config_map_update() -> float:
    """
    Return the current epoch time in milliseconds when the test class starts.
    Useful for comparing against timestamps created during test execution.
    """
    return float(time.time() * 1000)


@pytest.fixture(scope="function")
def initial_last_synced_values(
    request: pytest.FixtureRequest,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> str:
    """
    Collect initial last_synced values for a given model.
    """
    result = get_huggingface_model_from_api(
        model_registry_rest_headers=model_registry_rest_headers,
        model_catalog_rest_url=model_catalog_rest_url,
        model_name=request.param,
        source_id="hf_id",
    )

    return result["customProperties"]["last_synced"]["string_value"]
