import pytest
from simple_logger.logger import get_logger
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_runtime.vllm.utils import fetch_openai_response

LOGGER = get_logger(name=__name__)

serving_arument = ["--dtype=bfloat16", "--model=/mnt/models", "--max-model-len=2048", "--uvicorn-log-level=debug"]


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-serverless-rest"},
            {"model-dir": "granite-2b-instruct-preview-4k-r240917a"},
            {"deployment_type": "Serverless"},
            {
                "deployment_mode": KServeDeploymentType.SERVERLESS,
                "runtime_argument": serving_arument,
                "gpu_count": 1,
                "name": "granite-rest",
                "min-replicas": 1,
            },
        ),
    ],
    indirect=True,
)
class TestGranite2BModel:
    def test_deploy_model_inference(self, vllm_inference_service, response_snapshot):
        URL = vllm_inference_service.instance.status.url
        if (
            vllm_inference_service.instance.metadata.annotations["serving.kserve.io/deploymentMode"]
            == KServeDeploymentType.SERVERLESS
        ):
            model_info, chat_responses, completion_responses = fetch_openai_response(
                url=URL,
                model_name=vllm_inference_service.instance.metadata.name,
            )
            assert model_info == response_snapshot
            assert chat_responses == response_snapshot
            assert completion_responses == response_snapshot
