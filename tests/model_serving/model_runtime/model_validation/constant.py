from typing import Union
from utilities.constants import AcceleratorType, Labels, RuntimeTemplates

# Configurations
ACCELERATOR_IDENTIFIER: dict[str, str] = {
    AcceleratorType.NVIDIA: Labels.Nvidia.NVIDIA_COM_GPU,
    AcceleratorType.AMD: "amd.com/gpu",
    AcceleratorType.GAUDI: "habana.ai/gaudi",
}

TEMPLATE_MAP: dict[str, str] = {
    AcceleratorType.NVIDIA: RuntimeTemplates.VLLM_CUDA,
    AcceleratorType.AMD: RuntimeTemplates.VLLM_ROCM,
    AcceleratorType.GAUDI: RuntimeTemplates.VLLM_GAUDUI,
}


PREDICT_RESOURCES: dict[str, Union[list[dict[str, Union[str, dict[str, str]]]], dict[str, dict[str, str]]]] = {
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

BASE_SEVERRLESS_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.SERVERLESS,
    "runtime_argument": None,
    "min-replicas": 1,
}

COMPLETION_QUERY: list[dict[str, str]] = [
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

CHAT_QUERY: list[list[dict[str, str]]] = [
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


ORIGINAL_PULL_SECRET: str = "conn"  # pragma: allowlist-secret
INFERENCE_SERVICE_PORT: int = 8080
CONTAINER_PORT: int = 8080