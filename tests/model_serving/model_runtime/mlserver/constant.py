"""
Constants for MLServer model serving tests.

This module defines configuration values, resource specifications, deployment configs,
and input queries used across MLServer runtime tests for different frameworks.
"""

from typing import Any, Union

from utilities.constants import KServeDeploymentType, ModelFormat


class OutputType:
    """Model output types for response validation."""

    DETERMINISTIC: str = "deterministic"
    NON_DETERMINISTIC: str = "non_deterministic"


LOCALHOST_URL: str = "http://localhost"
RAW_DEPLOYMENT_TYPE: str = "raw"
MODEL_PATH_PREFIX: str = "mlserver/model_repository"

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

SKLEARN_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "sklearn",
    "inputs": [
        {
            "name": "sklearn-input-0",
            "shape": [2, 4],
            "datatype": "FP32",
            "data": [[6.8, 2.8, 4.8, 1.4], [6, 3.4, 4.5, 1.6]],
        }
    ],
}

XGBOOST_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "xgboost",
    "inputs": [
        {
            "name": "xgboost-input-0",
            "shape": [2, 4],
            "datatype": "FP32",
            "data": [[6.8, 2.8, 4.8, 1.4], [6, 3.4, 4.5, 1.6]],
        }
    ],
}

LIGHTGBM_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "lightgbm",
    "inputs": [
        {
            "name": "lightgbm-input-0",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [[6.7, 3.3, 5.7, 2.1]],
        }
    ],
}

ONNX_REST_INPUT_QUERY = {
    "id": "onnx",
    "inputs": [
        {
            "name": "input",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [[-1.44964521969853, -0.6500239344068982, -0.08343796979036086, -1.496529255090079]],
        }
    ],
}

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    ModelFormat.LIGHTGBM: {
        "model_name": ModelFormat.LIGHTGBM,
        "model_version": "v0.1.0",
        "rest_query": LIGHTGBM_REST_INPUT_QUERY,
        "output_type": OutputType.DETERMINISTIC,
    },
    ModelFormat.ONNX: {
        "model_name": ModelFormat.ONNX,
        "model_version": "v1.0.0",
        "rest_query": ONNX_REST_INPUT_QUERY,
        "output_type": OutputType.DETERMINISTIC,
    },
    ModelFormat.SKLEARN: {
        "model_name": ModelFormat.SKLEARN,
        "model_version": "v1.0.0",
        "rest_query": SKLEARN_REST_INPUT_QUERY,
        "output_type": OutputType.DETERMINISTIC,
    },
    ModelFormat.XGBOOST: {
        "model_name": ModelFormat.XGBOOST,
        "model_version": "v1.0.0",
        "rest_query": XGBOOST_REST_INPUT_QUERY,
        "output_type": OutputType.DETERMINISTIC,
    },
}
