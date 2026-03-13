"""
Constants for OpenVINO model serving tests.

This module defines configuration values, resource specifications, deployment configs,
and input queries used across OpenVINO runtime tests for different frameworks.
"""

from pathlib import Path
from typing import Any

from utilities.constants import (
    KServeDeploymentType,
    ModelFormat,
    Protocols,
)

MODEL_PATH_PREFIX: str = "openvino/model_repository"

LOCAL_HOST_URL: str = "http://localhost"

OPENVINO_REST_PORT: int = 8888

RAW_DEPLOYMENT_TYPE: str = "raw"

REST_PROTOCOL_TYPE_DICT: dict[str, str] = {"protocol_type": Protocols.REST}

PREDICT_RESOURCES: dict[str, list[dict[str, str | dict[str, str]]] | dict[str, dict[str, str]]] = {
    "volumes": [
        {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}},
        {"name": "tmp", "emptyDir": {}},
        {"name": "home", "emptyDir": {}},
    ],
    "volume_mounts": [
        {"name": "shared-memory", "mountPath": "/dev/shm"},
        {"name": "tmp", "mountPath": "/tmp"},
        {"name": "home", "mountPath": "/home/openvino"},
    ],
    "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "limits": {"cpu": "2", "memory": "4Gi"}},
}

BASE_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_type": KServeDeploymentType.RAW_DEPLOYMENT,
    "min-replicas": 1,
    "enable_external_route": False,
}

OPENVINO_INPUT_BASE_PATH: str = Path(__file__).parent

ONNX_REST_INPUT_QUERY_PATH: str = str(OPENVINO_INPUT_BASE_PATH / "onnx_input.json")

TENSORFLOW_REST_INPUT_QUERY_PATH: str = str(OPENVINO_INPUT_BASE_PATH / "tensorflow_input.json")

OPENVINO_IR_REST_INPUT_QUERY_PATH: str = str(OPENVINO_INPUT_BASE_PATH / "openvino_input.json")

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    ModelFormat.ONNX: {
        "rest_query_or_path": ONNX_REST_INPUT_QUERY_PATH,
    },
    ModelFormat.TENSORFLOW: {
        "rest_query_or_path": TENSORFLOW_REST_INPUT_QUERY_PATH,
    },
    ModelFormat.OPENVINO: {
        "rest_query_or_path": OPENVINO_IR_REST_INPUT_QUERY_PATH,
    },
}
