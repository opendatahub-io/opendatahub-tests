"""
Test module for lightgbm model deployment using MLServer runtime.

This module contains parameterized tests that validate lightgbm model inference
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
    LIGHTGBM_GRPC_INPUT_QUERY,
    LIGHTGBM_REST_INPUT_QUERY,
    LIGHTGBM_FRAMEWORK,
    DETERMINISTIC_OUTPUT,
)
from tests.model_serving.model_runtime.mlserver.utils import validate_inference_request


LOGGER = get_logger(name=__name__)

MODEL_NAME: str = "lightgbm-iris"

MODEL_VERSION: str = "v0.1.0"

MODEL_NAME_DICT: dict[str, str] = {"name": MODEL_NAME}

MODEL_STORAGE_URI_DICT: dict[str, str] = {"model-dir": f"{MODEL_PATH_PREFIX}/lightgbm"}


pytestmark = pytest.mark.usefixtures(
    "root_dir", "valid_aws_config", "mlserver_rest_serving_runtime_template", "mlserver_grpc_serving_runtime_template"
)


@pytest.mark.parametrize(
    ("protocol", "model_namespace", "mlserver_inference_service", "s3_models_storage_uri", "mlserver_serving_runtime"),
    [
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "lightgbm-iris-raw-rest"},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                **MODEL_NAME_DICT,
            },
            MODEL_STORAGE_URI_DICT,
            BASE_RAW_DEPLOYMENT_CONFIG,
            id="lightgbm-iris-raw-rest-deployment",
        ),
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "lightgbm-iris-raw-grpc"},
            {
                **BASE_RAW_DEPLOYMENT_CONFIG,
                **MODEL_NAME_DICT,
            },
            MODEL_STORAGE_URI_DICT,
            BASE_RAW_DEPLOYMENT_CONFIG,
            id="lightgbm-iris-raw-grpc-deployment",
        ),
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "lightgbm-iris-serverless-rest"},
            {**BASE_SERVERLESS_DEPLOYMENT_CONFIG, **MODEL_NAME_DICT},
            MODEL_STORAGE_URI_DICT,
            BASE_SERVERLESS_DEPLOYMENT_CONFIG,
            id="lightgbm-iris-serverless-rest-deployment",
        ),
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "lightgbm-iris-serverless-grpc"},
            {**BASE_SERVERLESS_DEPLOYMENT_CONFIG, **MODEL_NAME_DICT},
            MODEL_STORAGE_URI_DICT,
            BASE_SERVERLESS_DEPLOYMENT_CONFIG,
            id="lightgbm-iris-serverless-grpc-deployment",
        ),
    ],
    indirect=True,
)
class TestLightGBMModel:
    """
    Test class for lightgbm model inference with MLServer runtime.

    Covers multiple deployment scenarios:
    - REST and gRPC protocols
    - Raw and serverless deployment modes
    - Response validation against snapshots
    """

    def test_lightgbm_model_inference(
        self,
        mlserver_inference_service: InferenceService,
        mlserver_pod_resource: Pod,
        mlserver_response_snapshot: Any,
        protocol: str,
        root_dir: str,
    ) -> None:
        """
        Test lightgbm model inference across different protocols and deployment types.

        Args:
            mlserver_inference_service (InferenceService): The deployed inference service.
            mlserver_pod_resource (Pod): Pod running the model server.
            mlserver_response_snapshot (Any): Expected response for validation.
            protocol (str): Communication protocol (REST or gRPC).
            root_dir (str): Test root directory path.
        """
        input_query = LIGHTGBM_REST_INPUT_QUERY if protocol == Protocols.REST else LIGHTGBM_GRPC_INPUT_QUERY

        validate_inference_request(
            pod_name=mlserver_pod_resource.name,
            isvc=mlserver_inference_service,
            response_snapshot=mlserver_response_snapshot,
            input_query=input_query,
            model_version=MODEL_VERSION,
            model_framework=LIGHTGBM_FRAMEWORK,
            model_output_type=DETERMINISTIC_OUTPUT,
            protocol=protocol,
            root_dir=root_dir,
        )
