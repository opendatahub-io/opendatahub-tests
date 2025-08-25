"""
Constants for OpenVINO model serving tests.

This module defines configuration values, resource specifications, deployment configs,
and input queries used across OpenVINO runtime tests for different frameworks.
"""

from typing import Any, Union

import os

from utilities.constants import (
    KServeDeploymentType,
    Protocols,
    Timeout,
)

ONNX_MODEL_FORMAT_NAME: str = "onnx"

# TODO provide correct path before code push.
MODEL_PATH_PREFIX: str = "test-ovms/model_repository"

LOCAL_HOST_URL: str = "http://localhost"

OPENVINO_REST_PORT: int = 8888

RAW_DEPLOYMENT_TYPE: str = "raw"

SERVERLESS_DEPLOYMENT_TYPE: str = "serverless"

REST_PROTOCOL_TYPE_DICT: dict[str, str] = {"protocol_type": Protocols.REST}

PREDICT_RESOURCES: dict[str, Union[list[dict[str, Union[str, dict[str, str]]]], dict[str, dict[str, str]]]] = {
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
    "timeout": Timeout.TIMEOUT_5MIN,
}

BASE_SERVERLESS_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_type": KServeDeploymentType.SERVERLESS,
    "min-replicas": 1,
    "enable_external_route": True,
    "timeout": Timeout.TIMEOUT_5MIN,
}

OPENVINO_INPUT_BASE_PATH = "tests/model_serving/model_runtime/openvino"

ONNX_REST_INPUT_QUERY_PATH = os.path.join(OPENVINO_INPUT_BASE_PATH, "onnx_mnist_input.json")

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    ONNX_MODEL_FORMAT_NAME: {
        "model_name": "example-onnx-mnist",
        "rest_query_or_path": ONNX_REST_INPUT_QUERY_PATH,
    },
}
