"""Triton model deployment tests — parametrized across all supported model formats.

Each model format (ONNX, TensorFlow, Keras, PyTorch, Python, FIL, DALI) uses
the same deployment and inference validation flow, differing only in model
name, S3 path, input data, and pytest marks.
"""

from typing import Any

import pytest
from ocp_resources.inference_service import InferenceService
from ocp_resources.pod import Pod

from tests.model_serving.model_runtime.triton.basic_model_deployment.utils import load_json, validate_inference_request
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

pytestmark = pytest.mark.usefixtures("valid_aws_config", "triton_rest_serving_runtime_template")


@pytest.mark.tested_verified
@pytest.mark.parametrize(
    "model_namespace, s3_models_storage_uri, triton_serving_runtime, triton_inference_service, model_name, input_path",
    [
        pytest.param(
            {"name": "onnx-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "densenetonnx-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "densenetonnx",
            TRITON_REST_ONNX_INPUT_PATH,
            id="densenetonnx-standard-rest-deployment",
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            {"name": "tf-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "inceptiongraphdef-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "inceptiongraphdef",
            TRITON_REST_TF_INPUT_PATH,
            id="inceptiongraphdef-standard-rest-deployment",
            marks=pytest.mark.smoke,
        ),
        pytest.param(
            {"name": "keras-standard"},
            {"model-dir": MODEL_PATH_PREFIX_KERAS},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "resnet50-keras-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "resnet50",
            TRITON_REST_KERAS_INPUT_PATH,
            id="resnet50-keras-standard-rest-deployment",
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            {"name": "pytorch-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "resnet50-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "resnet50",
            TRITON_REST_PYTORCH_INPUT_PATH,
            id="resnet50-standard-rest-deployment",
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            {"name": "python-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "python-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "python",
            TRITON_REST_PYTHON_INPUT_PATH,
            id="python-standard-rest-deployment",
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            {"name": "fil-standard"},
            {"model-dir": MODEL_PATH_PREFIX},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "fil-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "fil",
            TRITON_REST_FIL_INPUT_PATH,
            id="fil-standard-rest-deployment",
            marks=pytest.mark.tier1,
        ),
        pytest.param(
            {"name": "dali-standard"},
            {"model-dir": MODEL_PATH_PREFIX_DALI},
            {**BASE_RAW_DEPLOYMENT_CONFIG},
            {"name": "daligpu-standard-rest", **BASE_RAW_DEPLOYMENT_CONFIG},
            "daligpu",
            TRITON_REST_DALI_INPUT_PATH,
            id="daligpu-standard-rest-deployment",
            marks=[pytest.mark.tier1, pytest.mark.gpu],
        ),
    ],
    indirect=["model_namespace", "s3_models_storage_uri", "triton_serving_runtime", "triton_inference_service"],
)
class TestTritonModelDeployment:
    def test_inference(
        self,
        triton_inference_service: InferenceService,
        triton_pod_resource: Pod,
        triton_response_snapshot: Any,
        model_name: str,
        input_path: str,
    ) -> None:
        input_query = load_json(path=input_path)

        validate_inference_request(
            pod_name=triton_pod_resource.name,
            isvc=triton_inference_service,
            response_snapshot=triton_response_snapshot,
            input_query=input_query,
            model_name=model_name,
        )
