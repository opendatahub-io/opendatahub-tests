from typing import Any, Dict

GRPC_PORT = 8033
REST_PORT = 8080
# Configurations
vLLM_CONFIG: Dict[str, Dict[str, Any]] = {
    "port_configurations": {
        "grpc": [{"containerPort": GRPC_PORT, "name": "h2c", "protocol": "TCP"}],
        "raw": [
            {"containerPort": REST_PORT, "name": "http1", "protocol": "TCP"},
            {"containerPort": GRPC_PORT, "name": "h2c", "protocol": "TCP"},
        ],
    },
    "commands": {"GRPC": "vllm_tgis_adapter"},
}
