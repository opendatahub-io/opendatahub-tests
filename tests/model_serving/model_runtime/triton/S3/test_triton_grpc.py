"""
Test module for Triton model formats using gRPC protocol.

Validates inference for all 7 Triton model formats using gRPC protocol:
    - ONNX (densenetonnx)
    - TensorFlow (inceptiongraphdef)
    - Keras (resnet50)
    - PyTorch (resnet50)
    - Python (custom backend)
    - FIL (Forest Inference Library)
    - DALI (Data Loading Library - GPU required)
"""

from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.triton.constant import (
    BASE_RAW_DEPLOYMENT_CONFIG,
    MODEL_PATH_PREFIX,
    MODEL_PATH_PREFIX_DALI,
    MODEL_PATH_PREFIX_KERAS,
    TRITON_GRPC_DALI_INPUT_PATH,
    TRITON_GRPC_FIL_INPUT_PATH,
    TRITON_GRPC_KERAS_INPUT_PATH,
    TRITON_GRPC_ONNX_INPUT_PATH,
    TRITON_GRPC_PYTHON_INPUT_PATH,
    TRITON_GRPC_PYTORCH_INPUT_PATH,
    TRITON_GRPC_TF_INPUT_PATH,
)
from tests.model_serving.model_runtime.triton.S3.utils import load_json, validate_inference_request
from utilities.constants import Protocols
from utilities.path_utils import resolve_repo_path

pytestmark = pytest.mark.usefixtures("root_dir", "valid_aws_config", "triton_grpc_serving_runtime_template")


@pytest.mark.parametrize(
    (
        "protocol",
        "model_namespace",
        "s3_models_storage_uri",
        "triton_serving_runtime",
        "triton_inference_service",
        "model_name",
        "input_path",
    ),
    [
        # ONNX Model - gRPC
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "onnx-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "densenetonnx-standard-grpc", **BASE_RAW_DEPLOYMENT_CONFIG},
            "densenetonnx",
            TRITON_GRPC_ONNX_INPUT_PATH,
            id="onnx-grpc",
            marks=pytest.mark.tier1,
        ),
        # TensorFlow Model - gRPC
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "tensorflow-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "tensorflow-standard-grpc", **BASE_RAW_DEPLOYMENT_CONFIG},
            "inceptiongraphdef",
            TRITON_GRPC_TF_INPUT_PATH,
            id="tensorflow-grpc",
            marks=pytest.mark.smoke,
        ),
        # Keras Model - gRPC
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "keras-standard"},
            {"model-dir": MODEL_PATH_PREFIX_KERAS},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "keras-standard-grpc", **BASE_RAW_DEPLOYMENT_CONFIG},
            "resnet50",
            TRITON_GRPC_KERAS_INPUT_PATH,
            id="keras-grpc",
            marks=pytest.mark.tier1,
        ),
        # PyTorch Model - gRPC
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "pytorch-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "pytorch-standard-grpc", **BASE_RAW_DEPLOYMENT_CONFIG},
            "resnet50",
            TRITON_GRPC_PYTORCH_INPUT_PATH,
            id="pytorch-grpc",
            marks=pytest.mark.tier1,
        ),
        # Python Model - gRPC
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "python-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "python-standard-grpc", **BASE_RAW_DEPLOYMENT_CONFIG},
            "python",
            TRITON_GRPC_PYTHON_INPUT_PATH,
            id="python-grpc",
            marks=pytest.mark.tier1,
        ),
        # FIL Model - gRPC
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "fil-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "fil-standard-grpc", **BASE_RAW_DEPLOYMENT_CONFIG},
            "fil",
            TRITON_GRPC_FIL_INPUT_PATH,
            id="fil-grpc",
            marks=pytest.mark.tier1,
        ),
        # DALI Model - gRPC (GPU required)
        pytest.param(
            {"protocol_type": Protocols.GRPC},
            {"name": "dali-standard"},
            {"model-dir": MODEL_PATH_PREFIX_DALI},
            {**BASE_RAW_DEPLOYMENT_CONFIG, "gpu_count": 1},
            {"name": "daligpu-standard-grpc", **BASE_RAW_DEPLOYMENT_CONFIG, "gpu_count": 1},
            "daligpu",
            TRITON_GRPC_DALI_INPUT_PATH,
            id="dali-grpc",
            marks=pytest.mark.gpu,
        ),
    ],
    indirect=[
        "protocol",
        "model_namespace",
        "s3_models_storage_uri",
        "triton_serving_runtime",
        "triton_inference_service",
    ],
)
class TestTritonGRPC:
    """
    Test class for all Triton model formats using gRPC protocol.

    Tests 7 Triton model formats with gRPC protocol:
    - ONNX, TensorFlow, Keras, PyTorch, Python, FIL, DALI

    Each test case:
    1. Deploys the model using gRPC ServingRuntime
    2. Loads the gRPC input JSON from S3/grpc-input/
    3. Sends an inference request via gRPC
    4. Validates the response against expected output structure
    """

    def test_triton_grpc_inference(
        self,
        triton_inference_service: InferenceService,
        triton_pod_resource: Pod,
        triton_response_snapshot: Any,
        protocol: str,
        root_dir: str,
        model_name: str,
        input_path: str,
    ) -> None:
        """
        Run gRPC inference and validate response for Triton model formats.

        Given: A deployed Triton InferenceService with a specific model format
        When: An inference request is sent via gRPC protocol
        Then: The response contains valid outputs matching the expected structure

        Args:
            triton_inference_service: The deployed InferenceService object
            triton_pod_resource: The pod running the Triton model server
            triton_response_snapshot: Expected response snapshot
            protocol: Protocol type (gRPC)
            root_dir: Root directory for test execution
            model_name: Name of the model being tested
            input_path: Path to gRPC protocol input JSON file
        """
        resolved_input_path = resolve_repo_path(source=input_path)
        input_query = load_json(path=resolved_input_path)

        validate_inference_request(
            pod_name=triton_pod_resource.name,
            isvc=triton_inference_service,
            response_snapshot=triton_response_snapshot,
            input_query=input_query,
            model_name=model_name,
            protocol=protocol,
            root_dir=root_dir,
        )
