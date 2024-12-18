import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor

from tests.model_serving.model_server.utils import get_pods_by_isvc_label


@pytest.fixture()
def patched_isvc_visibility_annotation(
    request: FixtureRequest, admin_client: DynamicClient, http_s3_inference_service: InferenceService
) -> InferenceService:
    with ResourceEditor(
        patches={
            http_s3_inference_service: {
                "metadata": {
                    "annotations": {"networking.kserve.io/visibility": request.param["visibility"]},
                }
            }
        }
    ):
        predictor_pod = get_pods_by_isvc_label(
            client=admin_client,
            isvc=http_s3_inference_service,
        )[0]
        predictor_pod.wait_deleted()

        yield http_s3_inference_service


@pytest.fixture()
def patched_grpc_isvc_visibility_annotation(
    request: FixtureRequest, admin_client: DynamicClient, grpc_s3_inference_service: InferenceService
) -> InferenceService:
    with ResourceEditor(
        patches={
            grpc_s3_inference_service: {
                "metadata": {
                    "annotations": {"networking.kserve.io/visibility": request.param["visibility"]},
                }
            }
        }
    ):
        predictor_pod = get_pods_by_isvc_label(
            client=admin_client,
            isvc=grpc_s3_inference_service,
        )[0]
        predictor_pod.wait_deleted()

        yield grpc_s3_inference_service
