import pytest
import structlog
import time

from tests.model_serving.model_runtime.vllm.utils import (
        run_raw_inference,
        get_vllm_version,
        get_vllm_throughput_logs,
        parse_vllm_logs,
        save_performance_report,
)
from utilities.constants import KServeDeploymentType, Ports

from tests.model_serving.model_runtime.vllm.constant import (
        CHAT_QUERY,
        COMPLETION_QUERY,
)

LOGGER = structlog.get_logger(name=__name__)

serving_argument = [
    "--dtype=bfloat16",
    "--model=/mnt/models",
    "--max-model-len=256",
    "--max-num-seqs=1",
    "--max-num-batched-tokens=256",
    "--uvicorn-log-level=debug",
]

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, serving_runtime, vllm_inference_service",
    [
        pytest.param(
            {"name": "granite-raw-cpu"},
            {"model-dir": "models/granite-3.1-8b-instruct"},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": serving_argument,
                "name": "granite-raw-cpu",
                "min-replicas": 1,
            },
        ),
    ],
    indirect=True,
)
class TestGranite8BModelCPU:
    def test_deploy_model_inference(
        self,
        vllm_inference_service,
        vllm_pod_resource,
        response_snapshot,
    ):
        pod = vllm_pod_resource.name
        namespace_name = vllm_inference_service.namespace
        model_name = "granite-3.1-8b-instruct"


        start_time = time.strftime("%H:%M:%S")

        model_details, grpc_chat_response, grpc_chat_stream_responses = run_raw_inference(
            pod_name=pod,
            isvc=vllm_inference_service,
            port=Ports.REST_PORT,
            endpoint="openai"
        )
        
        time.sleep(2)
        vllm_version = get_vllm_version(namespace_name, pod)
        used_entries_chat = set()
        chat_logs = get_vllm_throughput_logs(namespace_name, pod)
        save_performance_report(model_name=model_name, version=vllm_version, logs=chat_logs, request_type="chat", input_prompt=CHAT_QUERY[0][0]["content"], start_time=start_time, used_entries=used_entries_chat)

        assert model_details == response_snapshot
        assert grpc_chat_response == response_snapshot
        assert grpc_chat_stream_responses == response_snapshot
