import pytest
import structlog

from tests.ai_hub.model_catalog.constants import REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME
from tests.ai_hub.model_catalog.sorting.utils import RecommendedBaseline
from tests.ai_hub.model_catalog.utils import get_models_from_catalog_api

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(
    params=[
        "code_fixing",
        pytest.param("chatbot", marks=pytest.mark.tier1),
        "long_rag",
        "rag",
    ],
)
def recommended_baseline(
    request: pytest.FixtureRequest,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> RecommendedBaseline:
    """Baseline model ordering from legacy recommendations=true, parametrized by use case."""
    use_case: str = request.param
    artifact_filter = f"use_case.string_value='{use_case}'"

    response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label=REDHAT_AI_VALIDATED_UNESCAPED_CATALOG_NAME,
        page_size=1000,
        additional_params=f"&filterQuery=artifacts.{artifact_filter}&recommendations=true",
    )

    model_names = [model["name"] for model in response["items"]]
    assert model_names, f"Should have models with recommendations=true for use_case={use_case}"
    LOGGER.info(f"Captured {len(model_names)} models with recommendations=true for use_case={use_case}")

    return RecommendedBaseline(model_names=model_names, artifact_filter=artifact_filter, use_case=use_case)
