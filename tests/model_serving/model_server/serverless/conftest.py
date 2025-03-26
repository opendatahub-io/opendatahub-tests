from typing import Any, Generator

import pytest
from _pytest.fixtures import FixtureRequest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from ocp_resources.resource import ResourceEditor
from ocp_resources.namespace import Namespace
from ocp_resources.secret import Secret
from ocp_resources.serving_runtime import ServingRuntime

from tests.model_serving.model_server.serverless.utils import wait_for_canary_rollout
from tests.model_serving.model_server.utils import run_inference_multiple_times
from utilities.constants import Protocols
from utilities.inference_utils import Inference
from utilities.constants import KServeDeploymentType, ModelStoragePath
from utilities.inference_utils import create_isvc
from utilities.manifests.onnx import ONNX_INFERENCE_CONFIG


@pytest.fixture(scope="class")
def inference_service_patched_replicas(
    request: FixtureRequest, ovms_kserve_inference_service: InferenceService
) -> InferenceService:
    ResourceEditor(
        patches={
            ovms_kserve_inference_service: {
                "spec": {
                    "predictor": {"minReplicas": request.param["min-replicas"]},
                }
            }
        }
    ).update()

    return ovms_kserve_inference_service


@pytest.fixture
def inference_service_updated_canary_config(
    request: FixtureRequest, ovms_kserve_inference_service: InferenceService
) -> Generator[InferenceService, Any, Any]:
    canary_percent = request.param["canary-traffic-percent"]
    predictor_config = {
        "spec": {
            "predictor": {"canaryTrafficPercent": canary_percent},
        }
    }

    if model_path := request.param.get("model-path"):
        predictor_config["spec"]["predictor"]["model"] = {"storage": {"path": model_path}}

    with ResourceEditor(patches={ovms_kserve_inference_service: predictor_config}):
        wait_for_canary_rollout(isvc=ovms_kserve_inference_service, percentage=canary_percent)
        yield ovms_kserve_inference_service


@pytest.fixture
def multiple_onnx_inference_requests(
    ovms_kserve_inference_service: InferenceService,
) -> None:
    run_inference_multiple_times(
        isvc=ovms_kserve_inference_service,
        inference_config=ONNX_INFERENCE_CONFIG,
        inference_type=Inference.INFER,
        protocol=Protocols.HTTPS,
        iterations=50,
        run_in_parallel=True,
    )


@pytest.fixture(scope="class")
def s3_mnist_serverless_inference_service(
    request: FixtureRequest,
    admin_client: DynamicClient,
    model_namespace: Namespace,
    openvino_kserve_serving_runtime: ServingRuntime,
    models_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=admin_client,
        name="mnist-model",
        namespace=model_namespace.name,
        runtime=openvino_kserve_serving_runtime.name,
        storage_key=models_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.MNIST_8_ONNX,
        model_format=openvino_kserve_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def deleted_isvc(ovms_kserve_inference_service: InferenceService) -> None:
    ovms_kserve_inference_service.clean_up()
