from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor

from tests.model_serving.model_server.serverless.utils import wait_for_canary_rollout


@pytest.fixture
def inference_service_updated_canary_config(
    request: FixtureRequest, s3_models_inference_service: InferenceService
) -> Generator[InferenceService, Any, Any]:
    percent = request.param("canary-traffic-percent")
    predictor_config = {
        "spec": {
            "predictor": {"canaryTrafficPercent": percent},
        }
    }

    if model_path := request.param.get("model-path"):
        predictor_config["spec"]["predictor"]["model"]["storage_path"] = model_path

    with ResourceEditor(patches={s3_models_inference_service: predictor_config}):
        wait_for_canary_rollout(isvc=s3_models_inference_service, percentage=percent)
        yield s3_models_inference_service
