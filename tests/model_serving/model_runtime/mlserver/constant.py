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

RAW_DEPLOYMENT_TYPE: str = "raw"

LIGHTGBM_MODEL_FORMAT_NAME: str = "lightgbm"

SKLEARN_MODEL_FORMAT_NAME: str = "sklearn"

XGBOOST_MODEL_FORMAT_NAME: str = "xgboost"

DETERMINISTIC_OUTPUT: str = "deterministic"

NON_DETERMINISTIC_OUTPUT: str = "non_deterministic"

MODEL_PATH_PREFIX: str = "mlserver/model_repository"

REST_PROTOCOL_TYPE_DICT: dict[str, str] = {"protocol_type": Protocols.REST}


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
]

MLSERVER_CONTAINER_ENV: List[Dict[str, str]] = [
    {"name": "MLSERVER_HTTP_PORT", "value": str(MLSERVER_REST_PORT)},
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
}

TEMPLATE_NAME_MAP: dict[str, str] = {
    Protocols.REST: RuntimeTemplates.MLSERVER_REST,
}

RUNTIME_NAME_MAP: dict[str, str] = {
    Protocols.REST: "mlserver-rest-runtime",
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

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    LIGHTGBM_MODEL_FORMAT_NAME: {
        "model_name": "lightgbm",
        "model_version": "v0.1.0",
        "rest_query": LIGHTGBM_REST_INPUT_QUERY,
        "output_type": DETERMINISTIC_OUTPUT,
    },
    SKLEARN_MODEL_FORMAT_NAME: {
        "model_name": "sklearn",
        "model_version": "v1.0.0",
        "rest_query": SKLEARN_REST_INPUT_QUERY,
        "output_type": DETERMINISTIC_OUTPUT,
    },
    XGBOOST_MODEL_FORMAT_NAME: {
        "model_name": "xgboost",
        "model_version": "v1.0.0",
        "rest_query": XGBOOST_REST_INPUT_QUERY,
        "output_type": DETERMINISTIC_OUTPUT,
    },
}
