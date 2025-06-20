"""
Test module for huggingface model deployment using MLServer runtime.

This module contains parameterized tests that validate huggingface model inference
across different protocols (REST/gRPC) and deployment types (raw/serverless).
"""

from typing import Any

import pytest
from simple_logger.logger import get_logger

from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from utilities.constants import Protocols

from tests.model_serving.model_runtime.mlserver.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    BASE_SERVERLESS_DEPLOYMENT_CONFIG,
    MODEL_PATH_PREFIX,
    HUGGING_FACE_GRPC_INPUT_QUERY,
    HUGGING_FACE_REST_INPUT_QUERY,
    HUGGING_FACE_FRAMEWORK,
    NON_DETERMINISTIC_OUTPUT,
)
from tests.model_serving.model_runtime.mlserver.utils import validate_inference_request


LOGGER = get_logger(name=__name__)

MODEL_NAME: str = "distilgpt2"

MODEL_VERSION: str = "v0.1.0"

MODEL_NAME_DICT: dict[str, str] = {"name": MODEL_NAME}

MODEL_STORAGE_URI_DICT: dict[str, str] = {"model-dir": f"{MODEL_PATH_PREFIX}/huggingface"}


pytestmark = pytest.mark.usefixtures(
    "root_dir", "valid_aws_config", "mlserver_rest_serving_runtime_template", "mlserver_grpc_serving_runtime_template"
)


@pytest.mark.parametrize(
    ("protocol", "model_namespace", "mlserver_inference_service", "s3_models_storage_uri", "mlserver_serving_runtime"),
    [
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "huggingface-raw-rest"},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                **MODEL_NAME_DICT,
            },
            MODEL_STORAGE_URI_DICT,
            BASE_RAW_DEPLOYMENT_CONFIG,
            id="huggingface-raw-rest-deployment",
        ),
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "huggingface-raw-grpc"},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                **MODEL_NAME_DICT,
            },
            MODEL_STORAGE_URI_DICT,
            BASE_RAW_DEPLOYMENT_CONFIG,
            id="huggingface-raw-grpc-deployment",
        ),
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "huggingface-serverless-rest"},
            {**BASE_SERVERLESS_DEPLOYMENT_CONFIG, **MODEL_NAME_DICT},
            MODEL_STORAGE_URI_DICT,
            BASE_SERVERLESS_DEPLOYMENT_CONFIG,
            id="huggingface-serverless-rest-deployment",
        ),
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "huggingface-serverless-grpc"},
            {**BASE_SERVERLESS_DEPLOYMENT_CONFIG, **MODEL_NAME_DICT},
            MODEL_STORAGE_URI_DICT,
            BASE_SERVERLESS_DEPLOYMENT_CONFIG,
            id="huggingface-serverless-grpc-deployment",
        ),
    ],
    indirect=True,
)
class TestHuggingFaceModel:
    """
    Test class for huggingface model inference with MLServer runtime.

    Covers multiple deployment scenarios:
    - REST and gRPC protocols
    - Raw and serverless deployment modes
    - Response validation against snapshots
    """

    def test_huggingface_model_inference(
        self,
        mlserver_inference_service: InferenceService,
        mlserver_pod_resource: Pod,
        mlserver_response_snapshot: Any,
        protocol: str,
        root_dir: str,
    ) -> None:
        """
        Test huggingface model inference across different protocols and deployment types.

        Args:
            mlserver_inference_service (InferenceService): The deployed inference service.
            mlserver_pod_resource (Pod): Pod running the model server.
            mlserver_response_snapshot (Any): Expected response for validation.
            protocol (str): Communication protocol (REST or gRPC).
            root_dir (str): Test root directory path.
        """
        input_query = HUGGING_FACE_REST_INPUT_QUERY if protocol == Protocols.REST else HUGGING_FACE_GRPC_INPUT_QUERY

        validate_inference_request(
            pod_name=mlserver_pod_resource.name,
            isvc=mlserver_inference_service,
            response_snapshot=mlserver_response_snapshot,
            input_query=input_query,
            model_version=MODEL_VERSION,
            model_framework=HUGGING_FACE_FRAMEWORK,
            model_output_type=NON_DETERMINISTIC_OUTPUT,
            protocol=protocol,
            root_dir=root_dir,
        )
