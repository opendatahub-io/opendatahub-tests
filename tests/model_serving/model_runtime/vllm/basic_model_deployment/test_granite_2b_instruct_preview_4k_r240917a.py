import pytest
from simple_logger.logger import get_logger
from utilities.plugins.openai_plugin import OpenAIClient
from utilities.constants import KServeDeploymentType
from tests.model_serving.model_runtime.vllm.constant import CHAT_QUERY, COMPLETION_QUERY

LOGGER = get_logger(name=__name__)

serving_arument = ["--dtype=bfloat16", "--model=/mnt/models", "--max-model-len=2048", "--uvicorn-log-level=debug"]


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, serving_runtime, vllm_inference_service",
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
        completion_responses = []
        chat_responses = []
        URL = vllm_inference_service.instance.status.url
        if "rest" in vllm_inference_service.instance.metadata.namespace:
            openai_client = OpenAIClient(
                host=URL, model_name=vllm_inference_service.instance.metadata.name, streaming=True
            )
            for query in COMPLETION_QUERY:
                completion_response = openai_client.request_http(
                    endpoint="/v1/completions", query=query, extra_param={"max_tokens": 100}
                )
                completion_responses.append(completion_response)
            for query in CHAT_QUERY:
                chat_response = openai_client.request_http(endpoint="/v1/chat/completions", query=query)
                chat_responses.append(chat_response)
            model_info = OpenAIClient.get_request_http(host=URL, endpoint="/v1/models")
            assert model_info == response_snapshot
            assert completion_responses == response_snapshot
            assert chat_responses == response_snapshot
