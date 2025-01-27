import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.service_account import ServiceAccount
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.utils import create_isvc
from utilities.constants import KServeDeploymentType, ModelAndFormat, ModelFormat, ModelVersion, Protocols


@pytest.fixture(scope="class")
def http_s3_openvino_second_model_mesh_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    http_s3_ovms_model_mesh_serving_runtime: ServingRuntime,
    ci_model_mesh_endpoint_s3_secret: Secret,
    model_mesh_model_service_account: ServiceAccount,
) -> InferenceService:
    with create_isvc(
        client=admin_client,
        name=f"{Protocols.HTTP}-{ModelFormat.OPENVINO}-2",
        namespace=model_namespace.name,
        runtime=http_s3_ovms_model_mesh_serving_runtime.name,
        model_service_account=model_mesh_model_service_account.name,
        storage_key=ci_model_mesh_endpoint_s3_secret.name,
        storage_path=request.param["model-path"],
        model_format=ModelAndFormat.OPENVINO_IR,
        deployment_mode=KServeDeploymentType.MODEL_MESH,
        model_version=ModelVersion.OPSET1,
    ) as isvc:
        yield isvc
