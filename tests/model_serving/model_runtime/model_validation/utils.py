from contextlib import contextmanager
from typing import Generator, Any
from kubernetes.dynamic import DynamicClient
from ocp_resources.secret import Secret
from ocp_resources.inference_service import InferenceService
from simple_logger.logger import get_logger
from tests.model_serving.model_runtime.vllm.constant import CHAT_QUERY, COMPLETION_QUERY
from tenacity import retry, stop_after_attempt, wait_exponential

from utilities.constants import Ports
from utilities.exceptions import NotSupportedError
from utilities.plugins.constant import OpenAIEnpoints
from utilities.plugins.openai_plugin import OpenAIClient
from utilities.plugins.tgis_grpc_plugin import TGISGRPCPlugin
from tests.model_serving.model_runtime.vllm.constant import VLLM_SUPPORTED_QUANTIZATION
from tests.model_serving.model_runtime.vllm.constant import (
    OPENAI_ENDPOINT_NAME,
    TGIS_ENDPOINT_NAME,
)
import portforward
import pytest

LOGGER = get_logger(name=__name__)


@contextmanager
def kserve_s3_endpoint_secret(
    admin_client: DynamicClient,
    name: str,
    namespace: str,
    aws_access_key: str,
    aws_secret_access_key: str,
    aws_s3_endpoint: str,
    aws_s3_region: str,
) -> Generator[Secret, Any, Any]:
    with Secret(
        client=admin_client,
        name=name,
        namespace=namespace,
        annotations={
            "serving.kserve.io/s3-endpoint": aws_s3_endpoint.replace("https://", ""),
            "serving.kserve.io/s3-region": aws_s3_region,
            "serving.kserve.io/s3-useanoncredential": "false",
            "serving.kserve.io/s3-verifyssl": "0",
            "serving.kserve.io/s3-usehttps": "1",
        },
        string_data={
            "AWS_ACCESS_KEY_ID": aws_access_key,
            "AWS_SECRET_ACCESS_KEY": aws_secret_access_key,
        },
        wait_for_resource=True,
    ) as secret:
        yield secret

def validate_supported_quantization_schema(q_type: str) -> None:
    if q_type not in VLLM_SUPPORTED_QUANTIZATION:
        raise ValueError(f"Unsupported quantization type: {q_type}")

def skip_if_deployment_mode(isvc: InferenceService, deployment_type: str, deployment_message: str) -> None:
    if isvc.instance.metadata.annotations["serving.kserve.io/deploymentMode"] == deployment_type:
        pytest.skip(deployment_message)
