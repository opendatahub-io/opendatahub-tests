import os
from typing import Any, Union

from utilities.constants import (
    KServeDeploymentType,
    Protocols,
    RuntimeTemplates,
)

LOCAL_HOST_URL: str = "http://localhost"

MLSERVER_REST_PORT: int = 8080

MLSERVER_GRPC_PORT: int = 9000

MLSERVER_GRPC_REMOTE_PORT: int = 443

MODEL_PATH_PREFIX: str = "mlserver/model_repository"

PROTO_FILE_PATH: str = "utilities/manifests/common/grpc_predict_v2.proto"

TEMPLATE_FILE_PATH: dict[str, str] = {
    Protocols.REST: os.path.join(os.path.dirname(__file__), "mlserver_rest_serving_template.yaml"),
    Protocols.GRPC: os.path.join(os.path.dirname(__file__), "mlserver_grpc_serving_template.yaml"),
}

TEMPLATE_MAP: dict[str, str] = {
    Protocols.REST: RuntimeTemplates.MLSERVER_REST,
    Protocols.GRPC: RuntimeTemplates.MLSERVER_GRPC,
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
