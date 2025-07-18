import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from utilities.infra import get_pods_by_isvc_label
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from tests.model_serving.model_runtime.utils import skip_if_not_deployment_mode
from utilities.constants import KServeDeploymentType


LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def vllm_skip_if_no_supported_accelerator_type(supported_accelerator_type: str) -> None:
    if not supported_accelerator_type:
        pytest.skip("Accelerator type is not provided,vLLM test cannot be run on CPU")


@pytest.fixture
def vllm_pod_resource(admin_client: DynamicClient, vllm_model_car_inference_service: InferenceService) -> Pod:
    return get_pods_by_isvc_label(client=admin_client, isvc=vllm_model_car_inference_service)[0]


@pytest.fixture
def skip_if_not_serverless_deployment(vllm_model_car_inference_service: InferenceService) -> None:
    skip_if_not_deployment_mode(
        isvc=vllm_model_car_inference_service,
        deployment_type=KServeDeploymentType.SERVERLESS,
        deployment_message="Test is being skipped because model is being deployed in serverless mode",
    )


@pytest.fixture
def skip_if_not_raw_deployment(vllm_model_car_inference_service: InferenceService) -> None:
    skip_if_not_deployment_mode(
        isvc=vllm_model_car_inference_service,
        deployment_type=KServeDeploymentType.RAW_DEPLOYMENT,
        deployment_message="Test is being skipped because model is being deployed in raw mode",
    )
