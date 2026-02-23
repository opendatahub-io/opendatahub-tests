"""
Test module for model deployment using the OpenVINO runtime.

This module contains parameterized tests to validate model inference
across REST and Grpc protocol and raw deployment type.
"""

from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger

from tests.model_serving.model_runtime.openvino.constant import (
    MODEL_CONFIGS,
    RAW_DEPLOYMENT_TYPE,
    REST_PROTOCOL_TYPE_DICT,
)
from tests.model_serving.model_runtime.openvino.utils import (
    get_deployment_config_dict,
    get_input_query,
    get_model_namespace_dict,
    get_model_storage_uri_dict,
    get_test_case_id,
    validate_inference_request,
)
from utilities.constants import ModelFormat, Protocols

LOGGER = get_logger(name=__name__)

pytestmark = pytest.mark.usefixtures("valid_aws_config")


@pytest.mark.smoke
@pytest.mark.parametrize(
    (
        "protocol",
        "model_namespace",
        "openvino_inference_service",
        "s3_models_storage_uri",
        "openvino_serving_runtime",
        "model_format",
    ),
    [
        pytest.param(
            REST_PROTOCOL_TYPE_DICT,
            get_model_namespace_dict(
                model_format_name=ModelFormat.ONNX,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
            get_deployment_config_dict(model_format_name=ModelFormat.ONNX, deployment_type=RAW_DEPLOYMENT_TYPE),
            get_model_storage_uri_dict(model_format_name=ModelFormat.ONNX),
            get_deployment_config_dict(model_format_name=ModelFormat.ONNX, deployment_type=RAW_DEPLOYMENT_TYPE),
            ModelFormat.ONNX,
            id=get_test_case_id(
                model_format_name=ModelFormat.ONNX,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
        ),
        pytest.param(
            REST_PROTOCOL_TYPE_DICT,
            get_model_namespace_dict(
                model_format_name=ModelFormat.TENSORFLOW,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
            get_deployment_config_dict(model_format_name=ModelFormat.TENSORFLOW, deployment_type=RAW_DEPLOYMENT_TYPE),
            get_model_storage_uri_dict(model_format_name=ModelFormat.TENSORFLOW),
            get_deployment_config_dict(model_format_name=ModelFormat.TENSORFLOW, deployment_type=RAW_DEPLOYMENT_TYPE),
            ModelFormat.TENSORFLOW,
            id=get_test_case_id(
                model_format_name=ModelFormat.TENSORFLOW,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
        ),
        pytest.param(
            REST_PROTOCOL_TYPE_DICT,
            get_model_namespace_dict(
                model_format_name=ModelFormat.OPENVINO,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
            get_deployment_config_dict(model_format_name=ModelFormat.OPENVINO, deployment_type=RAW_DEPLOYMENT_TYPE),
            get_model_storage_uri_dict(model_format_name=ModelFormat.OPENVINO),
            get_deployment_config_dict(model_format_name=ModelFormat.OPENVINO, deployment_type=RAW_DEPLOYMENT_TYPE),
            ModelFormat.OPENVINO,
            id=get_test_case_id(
                model_format_name=ModelFormat.OPENVINO,
                deployment_type=RAW_DEPLOYMENT_TYPE,
                protocol_type=Protocols.REST,
            ),
        ),
    ],
    indirect=[
        "protocol",
        "model_namespace",
        "openvino_inference_service",
        "s3_models_storage_uri",
        "openvino_serving_runtime",
    ],
)
class TestOpenVINOModels:
    """
    Test class for model inference using the OpenVINO runtime.

    This class validates inference functionality across multiple configurations:
    - Protocols: REST and Grpc
    - Deployment modes: Raw
    - Response validation against predefined snapshots
    """

    def test_openvino_model_inference(
        self,
        openvino_inference_service: InferenceService,
        openvino_pod_resource: Pod,
        openvino_response_snapshot: Any,
        protocol: str,
        model_format: str,
    ) -> None:
        """
        Test model inference using OpenVINO across REST protocol and raw deployment type.

        This test sends inference requests using REST protocol and compares
        the actual response with the expected snapshot for validation.

        Args:
            openvino_inference_service (InferenceService): The deployed inference service instance.
            openvino_pod_resource (Pod): The Kubernetes pod running the OpenVINO.
            openvino_response_snapshot (Any): The expected model response for snapshot-based validation.
            protocol (str): The communication protocol to use ("rest").
            model_format (str): Identifier for the model framework (e.g., "tensorflow", "onnx").
        """

        if model_format not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model format: {model_format}")

        model_format_config = MODEL_CONFIGS[model_format]

        model_version = model_format_config.get("model_version", "")

        model_input_query = get_input_query(model_format_config=model_format_config, protocol=protocol)

        validate_inference_request(
            pod_name=openvino_pod_resource.name,
            isvc=openvino_inference_service,
            response_snapshot=openvino_response_snapshot,
            input_query=model_input_query,
            model_version=model_version,
        )
