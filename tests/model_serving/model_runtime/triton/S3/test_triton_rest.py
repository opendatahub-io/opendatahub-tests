"""
Test module for Triton model formats using REST protocol.

Validates inference for all 7 Triton model formats using REST protocol:
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
    TRITON_REST_DALI_INPUT_PATH,
    TRITON_REST_FIL_INPUT_PATH,
    TRITON_REST_KERAS_INPUT_PATH,
    TRITON_REST_ONNX_INPUT_PATH,
    TRITON_REST_PYTHON_INPUT_PATH,
    TRITON_REST_PYTORCH_INPUT_PATH,
    TRITON_REST_TF_INPUT_PATH,
)
from tests.model_serving.model_runtime.triton.S3.utils import load_json, validate_inference_request
from utilities.constants import Protocols
from utilities.path_utils import resolve_repo_path

pytestmark = pytest.mark.usefixtures("root_dir", "valid_aws_config", "triton_rest_serving_runtime_template")


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
        # ONNX Model - REST
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "onnx-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "densenetonnx-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "densenetonnx",
            TRITON_REST_ONNX_INPUT_PATH,
            id="onnx-rest",
            marks=pytest.mark.tier1,
        ),
        # TensorFlow Model - REST
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "tensorflow-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "tensorflow-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "inceptiongraphdef",
            TRITON_REST_TF_INPUT_PATH,
            id="tensorflow-rest",
            marks=pytest.mark.smoke,
        ),
        # Keras Model - REST
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "keras-standard"},
            {"model-dir": MODEL_PATH_PREFIX_KERAS},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "keras-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "resnet50",
            TRITON_REST_KERAS_INPUT_PATH,
            id="keras-rest",
            marks=pytest.mark.tier1,
        ),
        # PyTorch Model - REST
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "pytorch-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "pytorch-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "resnet50",
            TRITON_REST_PYTORCH_INPUT_PATH,
            id="pytorch-rest",
            marks=pytest.mark.tier1,
        ),
        # Python Model - REST
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "python-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "python-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "python",
            TRITON_REST_PYTHON_INPUT_PATH,
            id="python-rest",
            marks=pytest.mark.tier1,
        ),
        # FIL Model - REST
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "fil-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "fil-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "fil",
            TRITON_REST_FIL_INPUT_PATH,
            id="fil-rest",
            marks=pytest.mark.tier1,
        ),
        # DALI Model - REST (GPU required)
        pytest.param(
            {"protocol_type": Protocols.REST},
            {"name": "dali-standard"},
            {"model-dir": MODEL_PATH_PREFIX_DALI},
            {**BASE_RAW_DEPLOYMENT_CONFIG, "gpu_count": 1},
            {"name": "daligpu-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG, "gpu_count": 1},
            "daligpu",
            TRITON_REST_DALI_INPUT_PATH,
            id="dali-rest",
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
class TestTritonREST:
    """
    Test class for all Triton model formats using REST protocol.

    Tests 7 Triton model formats with REST protocol:
    - ONNX, TensorFlow, Keras, PyTorch, Python, FIL, DALI

    Each test case:
    1. Deploys the model using REST ServingRuntime
    2. Loads the REST input JSON from S3/rest-input/
    3. Sends an inference request via REST
    4. Validates the response against expected output structure
    """

    def test_triton_rest_inference(
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
        Run REST inference and validate response for Triton model formats.

        Given: A deployed Triton InferenceService with a specific model format
        When: An inference request is sent via REST protocol
        Then: The response contains valid outputs matching the expected structure

        Args:
            triton_inference_service: The deployed InferenceService object
            triton_pod_resource: The pod running the Triton model server
            triton_response_snapshot: Expected response snapshot
            protocol: Protocol type (REST)
            root_dir: Root directory for test execution
            model_name: Name of the model being tested
            input_path: Path to REST protocol input JSON file
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
