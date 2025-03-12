from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor

from tests.model_serving.model_server.serverless.utils import wait_for_canary_rollout
from tests.model_serving.model_server.utils import run_inference_multiple_times
from utilities.constants import ModelFormat, Protocols
from utilities.inference_utils import Inference
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG


@pytest.fixture(scope="class")
def inference_service_patched_replicas(
    request: FixtureRequest, ovms_serverless_inference_service: InferenceService
) -> InferenceService:
    ResourceEditor(
        patches={
            ovms_serverless_inference_service: {
                "spec": {
                    "predictor": {"minReplicas": request.param["min-replicas"]},
                }
            }
        }
    ).update()

    return ovms_serverless_inference_service


@pytest.fixture
def inference_service_updated_canary_config(
    request: FixtureRequest, s3_models_inference_service: InferenceService
) -> Generator[InferenceService, Any, Any]:
    canary_percent = request.param["canary-traffic-percent"]
    predictor_config = {
        "spec": {
            "predictor": {"canaryTrafficPercent": canary_percent},
        }
    }

    if model_path := request.param.get("model-path"):
        predictor_config["spec"]["predictor"]["model"] = {"storage": {"path": model_path}}

    with ResourceEditor(patches={s3_models_inference_service: predictor_config}):
        wait_for_canary_rollout(isvc=s3_models_inference_service, percentage=canary_percent)
        yield s3_models_inference_service


@pytest.fixture
def multiple_tgis_inference_requests(s3_models_inference_service: InferenceService) -> None:
    run_inference_multiple_times(
        isvc=s3_models_inference_service,
        inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
        inference_type=Inference.ALL_TOKENS,
        protocol=Protocols.HTTPS,
        model_name=ModelFormat.CAIKIT,
        iterations=50,
        run_in_parallel=True,
    )
