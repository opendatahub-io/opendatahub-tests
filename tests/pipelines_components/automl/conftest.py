import pytest

from tests.pipelines_components.constants import AUTOML_PIPELINE_YAML
from tests.pipelines_components.utils import resolve_pipeline_yaml


@pytest.fixture(scope="session", autouse=True)
def _validate_automl_env() -> None:
    if not AUTOML_PIPELINE_YAML:
        pytest.skip(
            "AutoML smoke test requires AUTOML_PIPELINE_YAML environment variable.\n"
            "Set it to a local path or URL of a compiled AutoGluon Tabular Training pipeline YAML."
        )

    import tests.pipelines_components.constants as _constants

    _constants.AUTOML_PIPELINE_YAML = resolve_pipeline_yaml(value=AUTOML_PIPELINE_YAML)
