import pytest

from simple_logger.logger import get_logger


LOGGER = get_logger(name=__name__)


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, serving_runtime",
    [
        pytest.param(
            {"name": "prxw-access"},
            {"model-dir": "granite-2b-instruct-preview-4k-r240917a"},
            {"deployment_type": "Serverless"},
        ),
        pytest.param(
            {"name": "pvc-rxw-access"},
            {"model-dir": "granite-2b-instruct-preview-4k-r240917a"},
            {"deployment_type": "Serverless-grpc"},
        ),
        pytest.param(
            {"name": "pvc-rxw-access"},
            {"model-dir": "granite-2b-instruct-preview-4k-r240917a"},
            {"deployment_type": "RAW"},
        ),
    ],
    indirect=True,
)
class TestGranite2BModel:
    def test_deploy_model_state_loaded(self):
        print("skelton")
