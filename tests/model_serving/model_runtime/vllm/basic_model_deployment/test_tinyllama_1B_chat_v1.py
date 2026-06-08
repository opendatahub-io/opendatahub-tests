import pytest
import structlog
from timeout_sampler import TimeoutSampler

from tests.model_serving.model_runtime.vllm.constant import (
    CHAT_QUERY,
)
from tests.model_serving.model_runtime.vllm.utils import (
    get_vllm_throughput_logs,
    get_vllm_version,
    run_raw_inference,
    save_performance_report,
)
from utilities.constants import KServeDeploymentType, Ports

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
            {"name": "tinyllama-raw-cpu"},
            {"model-dir": "models/tinyllama-1.1b-chat-v1.0"},
            {"deployment_type": KServeDeploymentType.RAW_DEPLOYMENT},
            {
                "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
                "runtime_argument": serving_argument,
                "name": "tinyllama-raw-cpu",
                "min-replicas": 1,
            },
        ),
    ],
    indirect=True,
)
class TestTinyLlama1BChatV1ModelCPU:
    def test_deploy_model_inference(
        self,
        vllm_inference_service,
        vllm_pod_resource,
        response_snapshot,
    ):
        pod_name = vllm_pod_resource.name
        namespace_name = vllm_inference_service.namespace
        model_name = "tinyllama-1B-chat-v1"

        model_details, grpc_chat_response, grpc_chat_stream_responses = run_raw_inference(
            pod_name=pod_name,
            isvc=vllm_inference_service,
            port=Ports.REST_PORT,
            endpoint="openai",
        )

        used_entries_chat = set()
        chat_logs = ""
        for sample in TimeoutSampler(
            wait_timeout=30,
            sleep=2,
            func=get_vllm_throughput_logs,
            namespace=namespace_name,
            pod_name=pod_name,
        ):
            chat_logs = sample
            if chat_logs:
                break

        vllm_version = get_vllm_version(namespace_name, pod_name)
        save_performance_report(
            model_name=model_name,
            version=vllm_version,
            logs=chat_logs,
            request_type="chat",
            input_prompt=CHAT_QUERY[0][0]["content"],
            used_entries=used_entries_chat,
        )

        assert model_details == response_snapshot
        assert grpc_chat_response == response_snapshot
        assert grpc_chat_stream_responses == response_snapshot
