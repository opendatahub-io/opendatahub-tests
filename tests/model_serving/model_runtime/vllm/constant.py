from typing import Any, Dict
from utilities.constants import AcceleratorType

GRPC_PORT = 8033
REST_PORT = 8080
REST_PORT_NAME = "http1"
GRPC_PORT_NAME = "h2c"
TCP_PROTOCOL_NAME = "TCP"
# Configurations
vLLM_CONFIG: Dict[str, Dict[str, Any]] = {
    "port_configurations": {
        "grpc": [{"containerPort": GRPC_PORT, "name": GRPC_PORT_NAME, "protocol": TCP_PROTOCOL_NAME}],
        "raw": [
            {"containerPort": REST_PORT, "name": REST_PORT_NAME, "protocol": TCP_PROTOCOL_NAME},
            {"containerPort": GRPC_PORT, "name": GRPC_PORT_NAME, "protocol": TCP_PROTOCOL_NAME},
        ],
    },
    "commands": {"GRPC": "vllm_tgis_adapter"},
}

TEMPLATE_MAP = {
    AcceleratorType.NVIDIA: "vllm-runtime-template",
    AcceleratorType.AMD: "vllm-rocm-runtime-template",
    AcceleratorType.GAUDI: "vllm-gaudi-runtime-template",
}

ACCELERATOR_IDENTIFIER = {
    AcceleratorType.NVIDIA: "nvidia.com/gpu",
    AcceleratorType.AMD: "amd.com/gpu",
    AcceleratorType.GAUDI: "habana.ai/gaudi",
}

PREDICT_RESOURCES = {
    "volumes": [
        {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "16Gi"}},
        {"name": "tmp", "emptyDir": {}},
        {"name": "home", "emptyDir": {}},
    ],
    "volume_mounts": [
        {"name": "shared-memory", "mountPath": "/dev/shm"},
        {"name": "tmp", "mountPath": "/tmp"},
        {"name": "home", "mountPath": "/home/vllm"},
    ],
    "resources": {"requests": {"cpu": "2", "memory": "15Gi"}, "limits": {"cpu": "3", "memory": "16Gi"}},
}


COMPLETION_QUERY = [
    {
        "text": "List the top five breeds of dogs and their characteristics.",
    },
    {
        "text": "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches "
        "the worm.'"
    },
    {"text": "Write a short story about a robot that dreams for the first time."},
    {
        "text": "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in "
        "Western versus Eastern societies."
    },
    {
        "text": "Compare and contrast artificial intelligence with human intelligence in terms of "
        "processing information."
    },
    {"text": "Briefly describe the major milestones in the development of artificial intelligence from 1950 to 2020."},
]
CHAT_QUERY = [
    [{"role": "user", "content": "Write python code to find even number"}],
    [
        {
            "role": "system",
            "content": "Given a target sentence, construct the underlying meaning representation of the input "
            "sentence as a single function with attributes and attribute values.",
        },
        {
            "role": "user",
            "content": "SpellForce 3 is a pretty bad game. The developer Grimlore Games is "
            "clearly a bunch of no-talent hacks, and 2017 was a terrible year for games anyway.",
        },
    ],
]
