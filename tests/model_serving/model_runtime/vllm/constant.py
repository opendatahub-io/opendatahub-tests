from typing import Any

from utilities.constants import AcceleratorType, KServeDeploymentType, Labels, RuntimeTemplates

# Quantization
VLLM_SUPPORTED_QUANTIZATION: list[str] = ["marlin", "awq"]
# Configurations
ACCELERATOR_IDENTIFIER: dict[str, str] = {
    AcceleratorType.NVIDIA: Labels.Nvidia.NVIDIA_COM_GPU,
    AcceleratorType.AMD: "amd.com/gpu",
    AcceleratorType.GAUDI: "habana.ai/gaudi",
    AcceleratorType.SPYRE: Labels.Spyre.SPYRE_COM_GPU,
    AcceleratorType.CPU_x86: Labels.CPU.CPU_x86,
    AcceleratorType.CPU_POWER: Labels.CPU.CPU_x86,
    AcceleratorType.CPU_Z: Labels.CPU.CPU_x86,
}

TEMPLATE_MAP: dict[str, str] = {
    AcceleratorType.NVIDIA: RuntimeTemplates.VLLM_CUDA,
    AcceleratorType.AMD: RuntimeTemplates.VLLM_ROCM,
    AcceleratorType.GAUDI: RuntimeTemplates.VLLM_GAUDI,
    AcceleratorType.SPYRE: RuntimeTemplates.VLLM_SPYRE,
    AcceleratorType.CPU_x86: RuntimeTemplates.VLLM_CPU_x86,
    AcceleratorType.CPU_POWER: RuntimeTemplates.VLLM_CPU_POWER,
    AcceleratorType.CPU_Z: RuntimeTemplates.VLLM_CPU_Z,
}

PREDICT_RESOURCES: dict[str, list[dict[str, str | dict[str, str]]] | dict[str, dict[str, str]]] = {
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

COMPLETION_QUERY: list[dict[str, Any]] = [
    {
        "text": "List the top five breeds of dogs and their characteristics.",
        "keywords": [
            "dog",
            "breed",
            "labrador",
            "german",
            "golden",
            "bulldog",
            "beagle",
            "poodle",
            "characteristic",
            "loyal",
            "friendly",
        ],
    },
    {
        "text": "Translate the following English sentence into Japanese, French, and Swahili: 'The early bird catches "
        "the worm.'",
        "keywords": [
            "japanese",
            "french",
            "swahili",
            "translation",
            "bird",
            "worm",
            "early",
            "tori",
            "oiseau",
            "ndege",
        ],
    },
    {
        "text": "Write a short story about a robot that dreams for the first time.",
        "keywords": [
            "robot",
            "dream",
            "story",
            "first",
            "time",
            "android",
            "machine",
            "sleep",
            "consciousness",
            "awake",
        ],
    },
    {
        "text": "Explain the cultural significance of the Mona Lisa painting, and how its perception might vary in "
        "Western versus Eastern societies.",
        "keywords": [
            "mona lisa",
            "cultural",
            "painting",
            "western",
            "eastern",
            "art",
            "da vinci",
            "leonardo",
            "society",
            "perception",
        ],
    },
    {
        "text": "Compare and contrast artificial intelligence with human intelligence in terms of "
        "processing information.",
        "keywords": [
            "artificial intelligence",
            "human intelligence",
            "ai",
            "compare",
            "contrast",
            "processing",
            "information",
            "machine",
            "brain",
            "learning",
        ],
    },
    {
        "text": "Briefly describe the major milestones in the development of artificial intelligence "
        "from 1950 to 2020.",
        "keywords": [
            "artificial intelligence",
            "milestone",
            "development",
            "1950",
            "2020",
            "history",
            "ai",
            "turing",
            "deep learning",
            "neural",
        ],
    },
]

CHAT_QUERY: list[list[dict[str, Any]]] = [
    [
        {"role": "user", "content": "What is an even number? Answer in one or two sentences."},
        {"keywords": ["even", "number", "divisible", "two", "2", "integer", "half"]},
    ],
    [
        {"role": "user", "content": "Name three common dog breeds and one trait for each."},
        {"keywords": ["dog", "breed", "labrador", "retriever", "poodle", "loyal", "friendly"]},
    ],
]

GRANITE_SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--dtype=float16",
]

GRANITE_CHAT_QUERY: list[list[dict[str, Any]]] = [
    [
        {"role": "user", "content": "Write python code to find even number"},
        {"keywords": ["python", "code", "even", "number", "def", "return", "modulo", "%", "function", "if"]},
    ],
]

BASE_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.STANDARD,
    "runtime_argument": None,
    "min-replicas": 1,
}
