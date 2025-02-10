from typing import Any, Union
from utilities.constants import AcceleratorType

GRPC_PORT: int = 8033
REST_PORT: int = 8080
REST_PORT_NAME: str = "http1"
GRPC_PORT_NAME: str = "h2c"
TCP_PROTOCOL_NAME: str = "TCP"
OPENAI_ENDPOINT_NAME: str = "openai"
# Quantization
VLLM_SUPPORTED_QUANTIZATION: list[str] = ["marlin", "awq"]
# Configurations
vLLM_CONFIG: dict[str, dict[str, Any]] = {
    "port_configurations": {
        "grpc": [{"containerPort": GRPC_PORT, "name": GRPC_PORT_NAME, "protocol": TCP_PROTOCOL_NAME}],
        "raw": [
            {"containerPort": REST_PORT, "name": REST_PORT_NAME, "protocol": TCP_PROTOCOL_NAME},
            {"containerPort": GRPC_PORT, "name": GRPC_PORT_NAME, "protocol": TCP_PROTOCOL_NAME},
        ],
    },
    "commands": {"GRPC": "vllm_tgis_adapter"},
}
TEMPLATE_MAP: dict[str, str] = {
    AcceleratorType.NVIDIA: "vllm-runtime-template",
    AcceleratorType.AMD: "vllm-rocm-runtime-template",
    AcceleratorType.GAUDI: "vllm-gaudi-runtime-template",
}

ACCELERATOR_IDENTIFIER: dict[str, str] = {
    AcceleratorType.NVIDIA: "nvidia.com/gpu",
    AcceleratorType.AMD: "amd.com/gpu",
    AcceleratorType.GAUDI: "habana.ai/gaudi",
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

IMAGE_URL_SCENERY: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"  # noqa: E501
IMAGE_URL_DUCK: str = (
    "https://upload.wikimedia.org/wikipedia/commons/d/da/2015_Kaczka_krzy%C5%BCowka_w_wodzie_%28samiec%29.jpg"
)
IMAGE_URL_LION: str = "https://upload.wikimedia.org/wikipedia/commons/7/77/002_The_lion_king_Snyggve_in_the_Serengeti_National_Park_Photo_by_Giles_Laurent.jpg"  # noqa: E501

MULTI_IMAGE_QUERIES: list[list[dict[Any, Any]]] = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_URL_SCENERY},
                },
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What are the animals in these images?"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL_LION}},
                {"type": "image_url", "image_url": {"url": IMAGE_URL_DUCK}},
            ],
        }
    ],
]

THREE_IMAGE_QUERY: list[list[dict[Any, Any]]] = [
    [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Explain these images?"},
                {"type": "image_url", "image_url": {"url": IMAGE_URL_LION}},
                {"type": "image_url", "image_url": {"url": IMAGE_URL_DUCK}},
                {
                    "type": "image_url",
                    "image_url": {"url": IMAGE_URL_SCENERY},
                },
            ],
        }
    ],
]
