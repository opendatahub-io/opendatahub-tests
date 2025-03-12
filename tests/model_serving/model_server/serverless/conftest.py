import pytest
from _pytest.fixtures import FixtureRequest
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor


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
