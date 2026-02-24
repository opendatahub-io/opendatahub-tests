import os
from typing import Any

from utilities.constants import (
    KServeDeploymentType,
    Labels,
    Protocols,
    RuntimeTemplates,
    Timeout,
)

TRITON_INPUT_BASE_PATH = "tests/model_serving/model_runtime/triton/basic_model_deployment"

TRITON_REST_ONNX_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-onnx-rest-input.json")
TRITON_GRPC_ONNX_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-onnx-gRPC-input.json")
TRITON_REST_PYTHON_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-python-rest-input.json")
TRITON_GRPC_PYTHON_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-python-gRPC-input.json")
TRITON_REST_KERAS_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-keras-triton-resnet-rest-input.json")
TRITON_GRPC_KERAS_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-keras-triton-resnet-gRPC-input.json")
TRITON_REST_TF_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-tensorflow-rest-input.json")
TRITON_GRPC_TF_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-tensorflow-gRPC-input.json")
TRITON_REST_PYTORCH_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-resnet-rest-input.json")
TRITON_GRPC_PYTORCH_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-resnet-gRPC-input.json")
TRITON_REST_DALI_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-dali-rest-input.json")
TRITON_GRPC_DALI_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-dali-gRPC-input.json")
TRITON_REST_FIL_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-fil-rest-input.json")
TRITON_GRPC_FIL_INPUT_PATH = os.path.join(TRITON_INPUT_BASE_PATH, "kserve-triton-fil-gRPC-input.json")

LOCAL_HOST_URL: str = "http://localhost"
TRITON_REST_PORT: int = 8080
TRITON_GRPC_PORT: int = 9000


TRITON_IMAGE: str = "nvcr.io/nvidia/tritonserver:24.10-py3"


MODEL_PATH_PREFIX_KERAS: str = "triton_resnet/model_repository"
MODEL_PATH_PREFIX: str = "triton/model_repository"
MODEL_PATH_PREFIX_DALI: str = "triton_gpu/model_repository"

PROTO_FILE_PATH: str = "utilities/manifests/common/grpc_predict_v2.proto"

TEMPLATE_MAP: dict[str, str] = {
    "rest_nvidia": RuntimeTemplates.TRITON_REST,
    "grpc_nvidia": RuntimeTemplates.TRITON_GRPC,
}

RUNTIME_MAP: dict[str, str] = {
    Protocols.REST: "triton-rest-runtime",
    Protocols.GRPC: "triton-grpc-runtime",
}

PREDICT_RESOURCES: dict[str, list[dict[str, str | dict[str, str]]] | dict[str, dict[str, str]]] = {
    "volumes": [
        {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
        {"name": "tmp", "emptyDir": {}},
        {"name": "home", "emptyDir": {}},
    ],
    "volume_mounts": [
        {"name": "shared-memory", "mountPath": "/dev/shm"},
        {"name": "tmp", "mountPath": "/tmp"},
        {"name": "home", "mountPath": "/home/triton"},
    ],
    "resources": {
        "requests": {"cpu": "1", "memory": "2Gi"},
        "limits": {"cpu": "2", "memory": "4Gi"},
    },
}

BASE_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_type": KServeDeploymentType.RAW_DEPLOYMENT,
    "min-replicas": 1,
    "enable_external_route": False,
    "timeout": Timeout.TIMEOUT_10MIN,
}

ACCELERATOR_IDENTIFIER: dict[str, str] = {
    "nvidia": Labels.Nvidia.NVIDIA_COM_GPU,
    "amd": Labels.ROCm.ROCM_GPU,
}
