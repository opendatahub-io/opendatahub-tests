import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.inference_service import InferenceService
from utilities.infra import get_pods_by_isvc_label
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.fixture(scope="session", autouse=True)
def vllm_skip_if_no_supported_accelerator_type(supported_accelerator_type: str) -> None:  # noqa: UFN001
    if not supported_accelerator_type:
        pytest.skip("Accelerator type is not provided,vLLM test can not be run on CPU")


@pytest.fixture
def vllm_pod_resource(admin_client: DynamicClient, vllm_model_car_inference_service: InferenceService) -> Pod:
    return get_pods_by_isvc_label(client=admin_client, isvc=vllm_model_car_inference_service)[0]
