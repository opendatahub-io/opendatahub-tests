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
from utilities.constants import ModelFormat, Protocols, Timeout
from utilities.constants import KServeDeploymentType, ModelStoragePath
from utilities.inference_utils import Inference, create_isvc
from utilities.infra import verify_no_failed_pods
from utilities.manifests.caikit_tgis import CAIKIT_TGIS_INFERENCE_CONFIG


@pytest.fixture(scope="class")
def inference_service_patched_replicas(
    request: FixtureRequest, ovms_kserve_inference_service: InferenceService
) -> InferenceService:
    if hasattr(request, "param"):
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
    request: FixtureRequest, unprivileged_client: DynamicClient, ovms_kserve_inference_service: InferenceService
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
        verify_no_failed_pods(
            client=unprivileged_client,
            isvc=ovms_kserve_inference_service,
            timeout=Timeout.TIMEOUT_2MIN,
        )
        yield ovms_kserve_inference_service


@pytest.fixture
def multiple_onnx_inference_requests(
    s3_models_inference_service: InferenceService,
) -> None:
    run_inference_multiple_times(
        isvc=s3_models_inference_service,
        inference_config=CAIKIT_TGIS_INFERENCE_CONFIG,
        inference_type=Inference.ALL_TOKENS,
        protocol=Protocols.HTTPS,
        model_name=ModelFormat.CAIKIT,
        iterations=20,
        run_in_parallel=True,
    )


@pytest.fixture(scope="class")
def s3_mnist_serverless_inference_service(
    request: FixtureRequest,
    unprivileged_client: DynamicClient,
    unprivileged_model_namespace: Namespace,
    ovms_kserve_serving_runtime: ServingRuntime,
    ci_endpoint_s3_secret: Secret,
) -> Generator[InferenceService, Any, Any]:
    with create_isvc(
        client=unprivileged_client,
        name="mnist-model",
        namespace=unprivileged_model_namespace.name,
        runtime=ovms_kserve_serving_runtime.name,
        storage_key=ci_endpoint_s3_secret.name,
        storage_path=ModelStoragePath.MNIST_8_ONNX,
        model_format=ovms_kserve_serving_runtime.instance.spec.supportedModelFormats[0].name,
        deployment_mode=KServeDeploymentType.SERVERLESS,
        external_route=True,
    ) as isvc:
        yield isvc


@pytest.fixture(scope="class")
def deleted_isvc(ovms_kserve_inference_service: InferenceService) -> None:
    ovms_kserve_inference_service.clean_up()
