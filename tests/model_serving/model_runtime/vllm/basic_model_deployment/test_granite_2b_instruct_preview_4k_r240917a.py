import pytest
import shlex
from typing import List
from simple_logger.logger import get_logger
from utilities.plugins.openai_plugin import OpenAIClient
from tests.model_serving.model_runtime.vllm.constant import CHAT_QUERY, COMPLETION_QUERY
from tests.model_serving.model_server.storage.constants import (
    KSERVE_CONTAINER_NAME,
)

POD_LS_SPLIT_COMMAND: List[str] = shlex.split("ls /mnt/models")

LOGGER = get_logger(name=__name__)

serving_arument = ["--dtype=bfloat16", "--model=/mnt/models", "--max-model-len=2048", "--uvicorn-log-level=debug"]


pytestmark = pytest.mark.usefixtures("skip_if_no_supported_accelerator_type", "valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, ci_s3_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "prxw-access"},
            {"model-dir": "granite-2b-instruct-preview-4k-r240917a"},
            {"deployment_type": "Serverless"},
            {"runtime_argument": serving_arument, "gpu_count": 1, "name": "granite-rest", "min-replicas": 1},
        ),
    ],
    indirect=True,
)
class TestGranite2BModel:
    def test_deploy_model_state_loaded(self, predictor_pods_scope_class, vllm_inference_service, response_snapshot):
        predictor_pods_scope_class[0].execute(
            container=KSERVE_CONTAINER_NAME,
            command=POD_LS_SPLIT_COMMAND,
        )
        URL = vllm_inference_service.instance.status.url
        model_info = OpenAIClient.get_request_http(host=URL, endpoint="/v1/models")
        completion_response = []
        openai_client = OpenAIClient(host=URL, model_name=vllm_inference_service.instance.metadata.name, streaming=True)
        for query in COMPLETION_QUERY:
            completion_responses = openai_client.request_http(endpoint="/v1/completions", query=query)
            completion_response.append(completion_responses)
        chat_response = []
        for query in CHAT_QUERY:
            chat_responses = openai_client.request_http(endpoint="/v1/chat/completions", query=query)
            chat_response.append(chat_responses)
        assert model_info == response_snapshot
        assert completion_response == response_snapshot
        assert chat_response == response_snapshot
