"""
Constants for MLServer model serving tests.

This module defines configuration values, resource specifications, deployment configs,
and input queries used across MLServer runtime tests for different frameworks.
"""

from typing import Any, Union, List, Dict

from utilities.constants import (
    KServeDeploymentType,
    Protocols,
    RuntimeTemplates,
)

LOCAL_HOST_URL: str = "http://localhost"

MLSERVER_REST_PORT: int = 8080

MLSERVER_GRPC_PORT: int = 9000

MLSERVER_GRPC_REMOTE_PORT: int = 443

RAW_DEPLOYMENT_TYPE: str = "raw"

SERVERLESS_DEPLOYMENT_TYPE: str = "serverless"

CATBOOST_MODEL_FORMAT_NAME: str = "catboost"

HUGGING_FACE_MODEL_FORMAT_NAME: str = "huggingface"

MLFLOW_MODEL_FORMAT_NAME: str = "mlflow"

LIGHTGBM_MODEL_FORMAT_NAME: str = "lightgbm"

SKLEARN_MODEL_FORMAT_NAME: str = "sklearn"

XGBOOST_MODEL_FORMAT_NAME: str = "xgboost"

DETERMINISTIC_OUTPUT: str = "deterministic"

NON_DETERMINISTIC_OUTPUT: str = "non_deterministic"

MODEL_PATH_PREFIX: str = "mlserver/model_repository"

PROTO_FILE_PATH: str = "utilities/manifests/common/grpc_predict_v2.proto"

REST_PROTOCOL_TYPE_DICT: dict[str, str] = {"protocol_type": Protocols.REST}

GRPC_PROTOCOL_TYPE_DICT: dict[str, str] = {"protocol_type": Protocols.GRPC}


MLSERVER_IMAGE: str = (
    "docker.io/seldonio/mlserver@sha256:07890828601515d48c0fb73842aaf197cbcf245a5c855c789e890282b15ce390"
)

MLSERVER_RUNTIME_LABELS: Dict[str, str] = {
    "opendatahub.io/dashboard": "true",
}

MLSERVER_RUNTIME_ANNOTATIONS: Dict[str, str] = {
    "openshift.io/display-name": "Seldon MLServer",
    "prometheus.kserve.io/port": "8080",
    "prometheus.kserve.io/path": "/metrics",
}

MLSERVER_SUPPORTED_MODEL_FORMATS: List[Dict[str, Any]] = [
    {"name": "sklearn", "version": "0", "autoSelect": True, "priority": 2},
    {"name": "sklearn", "version": "1", "autoSelect": True, "priority": 2},
    {"name": "xgboost", "version": "1", "autoSelect": True, "priority": 2},
    {"name": "xgboost", "version": "2", "autoSelect": True, "priority": 2},
    {"name": "lightgbm", "version": "3", "autoSelect": True, "priority": 2},
    {"name": "lightgbm", "version": "4", "autoSelect": True, "priority": 2},
    {"name": "mlflow", "version": "1", "autoSelect": True, "priority": 1},
    {"name": "mlflow", "version": "2", "autoSelect": True, "priority": 1},
    {"name": "catboost", "version": "1", "autoSelect": True, "priority": 1},
    {"name": "sparkmlib", "version": "1", "autoSelect": True, "priority": 1},
    {"name": "huggingface", "version": "1", "autoSelect": True, "priority": 1},
]

MLSERVER_CONTAINER_ENV: List[Dict[str, str]] = [
    {"name": "MLSERVER_HTTP_PORT", "value": str(MLSERVER_REST_PORT)},
    {"name": "MLSERVER_GRPC_PORT", "value": str(MLSERVER_GRPC_PORT)},
    {"name": "MODELS_DIR", "value": "/mnt/models"},
]

MLSERVER_CONTAINER_SECURITY_CONTEXT: Dict[str, Any] = {
    "allowPrivilegeEscalation": False,
    "capabilities": {"drop": ["ALL"]},
    "privileged": False,
    "runAsNonRoot": True,
}

MLSERVER_PORTS_MAP = {
    Protocols.REST: [{"containerPort": MLSERVER_REST_PORT, "protocol": "TCP"}],
    Protocols.GRPC: [{"containerPort": MLSERVER_GRPC_PORT, "name": "h2c", "protocol": "TCP"}],
}

TEMPLATE_NAME_MAP: dict[str, str] = {
    Protocols.REST: RuntimeTemplates.MLSERVER_REST,
    Protocols.GRPC: RuntimeTemplates.MLSERVER_GRPC,
}

RUNTIME_NAME_MAP: dict[str, str] = {
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

SKLEARN_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "sklearn",
    "model_version": "v1.0.0",
    "inputs": [
        {
            "name": "sklearn-input-0",
            "datatype": "FP32",
            "shape": [2, 4],
            "contents": {"fp32_contents": [6.8, 2.8, 4.8, 1.4, 6, 3.4, 4.5, 1.6]},
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

XGBOOST_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "xgboost",
    "model_version": "v1.0.0",
    "inputs": [
        {
            "name": "xgboost-input-0",
            "datatype": "FP32",
            "shape": [2, 4],
            "contents": {"fp32_contents": [6.8, 2.8, 4.8, 1.4, 6, 3.4, 4.5, 1.6]},
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

LIGHTGBM_GRPC_INPUT_QUERY: dict[str, Any] = {
    "model_name": "lightgbm",
    "model_version": "v0.1.0",
    "inputs": [
        {
            "name": "lightgbm-input-0",
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
    "id": "mlflow",
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
    "model_name": "mlflow",
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
    "model_name": "huggingface",
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

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    CATBOOST_MODEL_FORMAT_NAME: {
        "model_name": "catboost",
        "model_version": "v0.1.0",
        "rest_query": CATBOOST_REST_INPUT_QUERY,
        "grpc_query": CATBOOST_GRPC_INPUT_QUERY,
        "output_type": DETERMINISTIC_OUTPUT,
    },
    HUGGING_FACE_MODEL_FORMAT_NAME: {
        "model_name": "huggingface",
        "model_version": "v0.1.0",
        "rest_query": HUGGING_FACE_REST_INPUT_QUERY,
        "grpc_query": HUGGING_FACE_GRPC_INPUT_QUERY,
        "output_type": NON_DETERMINISTIC_OUTPUT,
    },
    LIGHTGBM_MODEL_FORMAT_NAME: {
        "model_name": "lightgbm",
        "model_version": "v0.1.0",
        "rest_query": LIGHTGBM_REST_INPUT_QUERY,
        "grpc_query": LIGHTGBM_GRPC_INPUT_QUERY,
        "output_type": DETERMINISTIC_OUTPUT,
    },
    MLFLOW_MODEL_FORMAT_NAME: {
        "model_name": "mlflow",
        "model_version": "v0.1.0",
        "rest_query": MLFLOW_REST_INPUT_QUERY,
        "grpc_query": MLFLOW_GRPC_INPUT_QUERY,
        "output_type": DETERMINISTIC_OUTPUT,
    },
    SKLEARN_MODEL_FORMAT_NAME: {
        "model_name": "sklearn",
        "model_version": "v1.0.0",
        "rest_query": SKLEARN_REST_INPUT_QUERY,
        "grpc_query": SKLEARN_GRPC_INPUT_QUERY,
        "output_type": DETERMINISTIC_OUTPUT,
    },
    XGBOOST_MODEL_FORMAT_NAME: {
        "model_name": "xgboost",
        "model_version": "v1.0.0",
        "rest_query": XGBOOST_REST_INPUT_QUERY,
        "grpc_query": XGBOOST_GRPC_INPUT_QUERY,
        "output_type": DETERMINISTIC_OUTPUT,
    },
}
