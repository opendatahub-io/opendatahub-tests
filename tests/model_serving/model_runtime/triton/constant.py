import os
import json
from typing import Any, Union

from utilities.constants import (
    KServeDeploymentType,
    Protocols,
    RuntimeTemplates,
)


TRITON_REST_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-resnet-rest-input.json"
)
TRITON_GRPC_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-resnet-gRPC-input.json"
)
TRITON_REST_ONNX_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-onnx-rest-input.json"
)
TRITON_GRPC_ONNX_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-onnx-gRPC-input.json"
)
TRITON_REST_TF_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-tensorflow-rest-input.json"
)
TRITON_GRPC_TF_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-tensorflow-gRPC-input.json"
)
TRITON_REST_KERAS_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-keras-triton-resnet-rest-input.json"
)
TRITON_GRPC_KERAS_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-keras-triton-resnet-grpc-input.json"
)
TRITON_REST_PYTHON_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-python-rest-input.json"
)
TRITON_GRPC_PYTHON_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-python-grpc-input.json"
)
TRITON_REST_FIL_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-fil_rest-input.json"
)
TRITON_GRPC_FIL_INPUT_PATH = (
    "tests/model_serving/model_runtime/triton/basic_model_deployment/kserve-triton-fil-gRPC-input.json"
)


def load_json(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


TRITON_REST_INPUT_QUERY = load_json(path=TRITON_REST_INPUT_PATH)
TRITON_GRPC_INPUT_QUERY = load_json(path=TRITON_GRPC_INPUT_PATH)
TRITON_REST_ONNX_INPUT_QUERY = load_json(path=TRITON_REST_ONNX_INPUT_PATH)
TRITON_GRPC_ONNX_INPUT_QUERY = load_json(path=TRITON_GRPC_ONNX_INPUT_PATH)
TRITON_REST_TF_INPUT_QUERY = load_json(path=TRITON_REST_TF_INPUT_PATH)
TRITON_GRPC_TF_INPUT_QUERY = load_json(path=TRITON_GRPC_TF_INPUT_PATH)
TRITON_REST_KERAS_INPUT_QUERY = load_json(path=TRITON_REST_KERAS_INPUT_PATH)
TRITON_GRPC_KERAS_INPUT_QUERY = load_json(path=TRITON_GRPC_KERAS_INPUT_PATH)
TRITON_REST_PYTHON_INPUT_QUERY = load_json(path=TRITON_REST_PYTHON_INPUT_PATH)
TRITON_GRPC_PYTHON_INPUT_QUERY = load_json(path=TRITON_GRPC_PYTHON_INPUT_PATH)
TRITON_REST_FIL_INPUT_QUERY = load_json(path=TRITON_REST_FIL_INPUT_PATH)
TRITON_GRPC_FIL_INPUT_QUERY = load_json(path=TRITON_GRPC_FIL_INPUT_PATH)


LOCAL_HOST_URL: str = "http://localhost"

TRITON_REST_PORT: int = 8080

TRITON_GRPC_PORT: int = 9000

TRITON_GRPC_REMOTE_PORT: int = 443

MODEL_PATH_PREFIX_KERAS: str = "triton_resnet/model_repository"

MODEL_PATH_PREFIX: str = "triton/model_repository"

PROTO_FILE_PATH: str = "utilities/manifests/common/grpc_predict_v2.proto"

BASE_DIR = os.path.dirname(__file__)
TEMPLATE_FILE_PATH: dict[str, str] = {
    Protocols.REST: os.path.join(BASE_DIR, "basic_model_deployment", "triton_rest_serving_template.yaml"),
    Protocols.GRPC: os.path.join(BASE_DIR, "basic_model_deployment", "triton_grpc_serving_template.yaml"),
}

TEMPLATE_MAP: dict[str, str] = {
    Protocols.REST: RuntimeTemplates.TRITON_REST,
    Protocols.GRPC: RuntimeTemplates.TRITON_GRPC,
}

RUNTIME_MAP: dict[str, str] = {
    Protocols.REST: "triton-kserve-rest",
    Protocols.GRPC: "triton-grpc-runtime",
}

PREDICT_RESOURCES: dict[str, Union[list[dict[str, Union[str, dict[str, str]]]], dict[str, dict[str, str]]]] = {
    "volumes": [
        {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}},
        {"name": "tmp", "emptyDir": {}},
        {"name": "home", "emptyDir": {}},
    ],
    "volume_mounts": [
        {"name": "shared-memory", "mountPath": "/dev/shm"},
        {"name": "tmp", "mountPath": "/tmp"},
        {"name": "home", "mountPath": "/home/mlserver"},
    ],
    "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}},
}

BASE_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_type": KServeDeploymentType.RAW_DEPLOYMENT,
    "min-replicas": 1,
    "enable_external_route": False,
}

BASE_SERVERLESS_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_type": KServeDeploymentType.SERVERLESS,
    "min-replicas": 1,
    "enable_external_route": True,
}
