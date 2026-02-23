"""
Test module for Dali  model served by Triton via KServe.

Validates inference using REST and gRPC protocols with raw deployment mode.
"""

from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.triton.basic_model_deployment.utils import load_json, validate_inference_request
from tests.model_serving.model_runtime.triton.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    MODEL_PATH_PREFIX_DALI,
    TRITON_GRPC_DALI_INPUT_PATH,
    TRITON_REST_DALI_INPUT_PATH,
)
from utilities.constants import Protocols

LOGGER = get_logger(name=__name__)

DALI_MODEL_NAME = "daligpu"
MODEL_STORAGE_URI_DICT = {"model-dir": f"{MODEL_PATH_PREFIX_DALI}"}

pytestmark = pytest.mark.usefixtures(
    "root_dir", "valid_aws_config", "triton_rest_serving_runtime_template", "triton_grpc_serving_runtime_template"
)


@pytest.mark.gpu
@pytest.mark.parametrize(
    ("protocol", "model_namespace", "s3_models_storage_uri", "triton_serving_runtime", "triton_inference_service"),
    [
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "dali-raw"},
            MODEL_STORAGE_URI_DICT,
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "dali-raw-rest",
                **BASE_RAW_DEPLOYMENT_CONFIG,
            },
            id="dali-raw-rest-deployment",
        ),
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "dali-raw"},
            MODEL_STORAGE_URI_DICT,
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {
                "name": "dali-raw-grpc",
                **BASE_RAW_DEPLOYMENT_CONFIG,
            },
            id="dali-raw-grpc-deployment",
        ),
    ],
    indirect=True,
)
class TestDaliModel:
    """
    Test class for Dali inference using Triton on KServe.

    Covers:
    - REST and gRPC protocols
    - Raw deployment mode
    - Snapshot validation of inference results
    """

    def test_dali_inference(
        self,
        triton_inference_service: InferenceService,
        triton_pod_resource: Pod,
        triton_response_snapshot: Any,
        protocol: str,
        root_dir: str,
    ) -> None:
        """
        Run inference and validate against snapshot.

        Args:
            triton_inference_service: The deployed InferenceService object
            triton_pod_resource: The pod running the model server
            triton_response_snapshot: Expected response snapshot
            protocol: REST or gRPC
            root_dir: Root directory for test execution
        """
        input_path = TRITON_GRPC_DALI_INPUT_PATH if protocol == Protocols.GRPC else TRITON_REST_DALI_INPUT_PATH
        input_query = load_json(path=input_path)

        validate_inference_request(
            pod_name=triton_pod_resource.name,
            isvc=triton_inference_service,
            response_snapshot=triton_response_snapshot,
            input_query=input_query,
            model_name=DALI_MODEL_NAME,
            protocol=protocol,
            root_dir=root_dir,
        )
