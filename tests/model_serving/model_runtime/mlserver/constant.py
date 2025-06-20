"""
Constants for MLServer model serving tests.

This module defines configuration values, resource specifications, deployment configs,
and input queries used across MLServer runtime tests for different frameworks.
"""

import os
from typing import Any, Union

from utilities.constants import (
    KServeDeploymentType,
    Protocols,
    RuntimeTemplates,
)

SKLEARN_FRAMEWORK: str = "sklearn"

XGBOOST_FRAMEWORK: str = "xgboost"

LIGHTGBM_FRAMEWORK: str = "lightgbm"

CATBOOST_FRAMEWORK: str = "catboost"

MLFLOW_FRAMEWORK: str = "mlflow"

HUGGING_FACE_FRAMEWORK: str = "huggingface"

DETERMINISTIC_OUTPUT: str = "deterministic"

NON_DETERMINISTIC_OUTPUT: str = "non_deterministic"

LOCAL_HOST_URL: str = "http://localhost"

MLSERVER_REST_PORT: int = 8080

MLSERVER_GRPC_PORT: int = 9000

MLSERVER_GRPC_REMOTE_PORT: int = 443

MODEL_PATH_PREFIX: str = "mlserver/model_repository"

PROTO_FILE_PATH: str = "utilities/manifests/common/grpc_predict_v2.proto"

MLSERVER_TESTS_DIR: str = os.path.dirname(__file__)

TEMPLATE_FILE_PATH: dict[str, str] = {
    Protocols.REST: os.path.join(MLSERVER_TESTS_DIR, "mlserver_rest_serving_template.yaml"),
    Protocols.GRPC: os.path.join(MLSERVER_TESTS_DIR, "mlserver_grpc_serving_template.yaml"),
}

TEMPLATE_MAP: dict[str, str] = {
    Protocols.REST: RuntimeTemplates.MLSERVER_REST,
    Protocols.GRPC: RuntimeTemplates.MLSERVER_GRPC,
}

RUNTIME_MAP: dict[str, str] = {
    Protocols.REST: "mlserver-rest-runtime",
    Protocols.GRPC: "mlserver-grpc-runtime",
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

SKLEARN_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "sklearn-iris",
    "inputs": [
        {
            "name": "sklearn-iris-input-0",
            "shape": [2, 4],
            "datatype": "FP32",
            "data": [[6.8, 2.8, 4.8, 1.4], [6, 3.4, 4.5, 1.6]],
        }
    ],
}

SKLEARN_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "sklearn-iris",
    "model_version": "v1.0.0",
    "inputs": [
        {
            "name": "sklearn-iris-input-0",
            "datatype": "FP32",
            "shape": [2, 4],
            "contents": {"fp32_contents": [6.8, 2.8, 4.8, 1.4, 6, 3.4, 4.5, 1.6]},
        }
    ],
}

XGBOOST_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "xgboost-iris",
    "inputs": [
        {
            "name": "xgboost-iris-input-0",
            "shape": [2, 4],
            "datatype": "FP32",
            "data": [[6.8, 2.8, 4.8, 1.4], [6, 3.4, 4.5, 1.6]],
        }
    ],
}

XGBOOST_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "xgboost-iris",
    "model_version": "v1.0.0",
    "inputs": [
        {
            "name": "xgboost-iris-input-0",
            "datatype": "FP32",
            "shape": [2, 4],
            "contents": {"fp32_contents": [6.8, 2.8, 4.8, 1.4, 6, 3.4, 4.5, 1.6]},
        }
    ],
}

LIGHTGBM_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "lightgbm-iris",
    "inputs": [
        {
            "name": "lightgbm-iris-input-0",
            "shape": [1, 4],
            "datatype": "FP32",
            "data": [[6.7, 3.3, 5.7, 2.1]],
        }
    ],
}

LIGHTGBM_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "lightgbm-iris",
    "model_version": "v0.1.0",
    "inputs": [
        {
            "name": "lightgbm-iris-input-0",
            "datatype": "FP32",
            "shape": [1, 4],
            "contents": {"fp32_contents": [6.7, 3.3, 5.7, 2.1]},
        }
    ],
}

CATBOOST_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "catboost",
    "inputs": [
        {
            "name": "catboost-input-0",
            "shape": [1, 10],
            "datatype": "FP32",
            "data": [[96, 84, 10, 16, 91, 57, 68, 77, 61, 81]],
        }
    ],
}

CATBOOST_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "catboost",
    "model_version": "v0.1.0",
    "inputs": [
        {
            "name": "catboost-input-0",
            "datatype": "FP32",
            "shape": [1, 10],
            "contents": {"fp32_contents": [96, 84, 10, 16, 91, 57, 68, 77, 61, 81]},
        }
    ],
}

MLFLOW_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "mlflow-wine-classifier",
    "inputs": [
        {"name": "fixed acidity", "shape": [1], "datatype": "FP32", "data": [7.4]},
        {"name": "volatile acidity", "shape": [1], "datatype": "FP32", "data": [0.7000]},
        {"name": "citric acid", "shape": [1], "datatype": "FP32", "data": [0]},
        {"name": "residual sugar", "shape": [1], "datatype": "FP32", "data": [1.9]},
        {"name": "chlorides", "shape": [1], "datatype": "FP32", "data": [0.076]},
        {"name": "free sulfur dioxide", "shape": [1], "datatype": "FP32", "data": [11]},
        {"name": "total sulfur dioxide", "shape": [1], "datatype": "FP32", "data": [34]},
        {"name": "density", "shape": [1], "datatype": "FP32", "data": [0.9978]},
        {"name": "pH", "shape": [1], "datatype": "FP32", "data": [3.51]},
        {"name": "sulphates", "shape": [1], "datatype": "FP32", "data": [0.56]},
        {"name": "alcohol", "shape": [1], "datatype": "FP32", "data": [9.4]},
    ],
}

MLFLOW_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "mlflow-wine-classifier",
    "model_version": "v0.1.0",
    "inputs": [
        {"name": "fixed acidity", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [7.4]}},
        {"name": "volatile acidity", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [0.7]}},
        {"name": "citric acid", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [0]}},
        {"name": "residual sugar", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [1.9]}},
        {"name": "chlorides", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [0.076]}},
        {"name": "free sulfur dioxide", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [11]}},
        {"name": "total sulfur dioxide", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [34]}},
        {"name": "density", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [0.9978]}},
        {"name": "pH", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [3.51]}},
        {"name": "sulphates", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [0.56]}},
        {"name": "alcohol", "shape": [1], "datatype": "FP32", "contents": {"fp32_contents": [9.4]}},
    ],
}

HUGGING_FACE_REST_INPUT_QUERY: dict[str, Any] = {
    "id": "huggingface",
    "inputs": [
        {
            "name": "text_inputs",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["This is a test"],
        }
    ],
}

HUGGING_FACE_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "distilgpt2",
    "model_version": "v0.1.0",
    "inputs": [
        {
            "name": "text_inputs",
            "datatype": "BYTES",
            "shape": [1],
            "contents": {"bytes_contents": ["VGhpcyBpcyBhIHRlc3QK"]},
        }
    ],
}
