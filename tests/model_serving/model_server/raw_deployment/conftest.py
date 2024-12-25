import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import KServeDeploymentType, ModelFormat, Protocols


@pytest.fixture(scope="class")
def http_s3_caikit_raw_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    http_s3_caikit_serving_runtime: ServingRuntime,
    s3_models_storage_uri: str,
    http_model_service_account: ServiceAccount,
) -> InferenceService:
    isvc_kwargs = {
        "client": admin_client,
        "name": f"{Protocols.HTTP}-{ModelFormat.CAIKIT}",
        "namespace": model_namespace.name,
        "runtime": http_s3_caikit_serving_runtime.name,
        "storage_uri": s3_models_storage_uri,
        "model_format": http_s3_caikit_serving_runtime.instance.spec.supportedModelFormats[0].name,
        "model_service_account": http_model_service_account.name,
        "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
    }

    enable_auth = False

    if hasattr(request, "param"):
        enable_auth = request.param.get("enable-auth")

    isvc_kwargs["enable_auth"] = enable_auth

    with create_isvc(**isvc_kwargs) as isvc:
        yield isvc


@pytest.fixture()
def patched_http_s3_caikit_raw_isvc_visibility_label(
    request: FixtureRequest, admin_client: DynamicClient, http_s3_caikit_raw_inference_service: InferenceService
) -> InferenceService:
    with ResourceEditor(
        patches={
            http_s3_caikit_raw_inference_service: {
                "metadata": {
                    "labels": {"networking.kserve.io/visibility": request.param["visibility"]},
                }
            }
        }
    ):
        yield http_s3_caikit_raw_inference_service
