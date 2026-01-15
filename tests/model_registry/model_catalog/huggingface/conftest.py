import pytest
from huggingface_hub import HfApi
from simple_logger.logger import get_logger

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
