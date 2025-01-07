import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor


@pytest.fixture()
def patched_http_s3_caikit_raw_isvc_visibility_label(
    request: FixtureRequest,
    admin_client: DynamicClient,
    s3_models_inference_service: InferenceService,
) -> InferenceService:
    with ResourceEditor(
        patches={
            s3_models_inference_service: {
                "metadata": {
                    "labels": {"networking.kserve.io/visibility": request.param["visibility"]},
                }
            }
        }
    ):
        yield s3_models_inference_service
