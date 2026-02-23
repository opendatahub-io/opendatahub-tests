from typing import Any

from utilities.constants import AcceleratorType, KServeDeploymentType, Labels, RuntimeTemplates

# Configurations
ACCELERATOR_IDENTIFIER: dict[str, str] = {
    AcceleratorType.NVIDIA: Labels.Nvidia.NVIDIA_COM_GPU,
    AcceleratorType.AMD: "amd.com/gpu",
    AcceleratorType.GAUDI: "habana.ai/gaudi",
    AcceleratorType.SPYRE: Labels.Spyre.SPYRE_COM_GPU,
    AcceleratorType.CPU_x86: Labels.CPU.CPU_x86,
}

TEMPLATE_MAP: dict[str, str] = {
    AcceleratorType.NVIDIA: RuntimeTemplates.VLLM_CUDA,
    AcceleratorType.AMD: RuntimeTemplates.VLLM_ROCM,
    AcceleratorType.GAUDI: RuntimeTemplates.VLLM_GAUDI,
    AcceleratorType.SPYRE: RuntimeTemplates.VLLM_SPYRE,
    AcceleratorType.CPU_x86: RuntimeTemplates.VLLM_CPU_x86,
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

BASE_RAW_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_mode": KServeDeploymentType.RAW_DEPLOYMENT,
    "runtime_argument": None,
    "min-replicas": 1,
}


COMPLETION_QUERY: list[dict[str, str]] = [
    {
        "text": "What are the key benefits of renewable energy sources compared to fossil fuels?",
    },
    {"text": "Translate the following English sentence into Spanish, German, and Mandarin: 'Knowledge is power.'"},
    {"text": "Write a poem about the beauty of the night sky and the mysteries it holds."},
    {"text": "Explain the significance of the Great Wall of China in history and its impact on modern tourism."},
    {"text": "Discuss the ethical implications of using artificial intelligence in healthcare decision-making."},
    {
        "text": "Summarize the main events of the Apollo 11 moon landing and its importance in space exploration history."
    },
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

EMBEDDING_QUERY: list[dict[str, str]] = [
    {
        "text": "What are the key benefits of renewable energy sources compared to fossil fuels?",
    },
    {"text": "Translate the following English sentence into Spanish, German, and Mandarin: 'Knowledge is power.'"},
    {"text": "Write a poem about the beauty of the night sky and the mysteries it holds."},
    {"text": "Explain the significance of the Great Wall of China in history and its impact on modern tourism."},
    {"text": "Discuss the ethical implications of using artificial intelligence in healthcare decision-making."},
    {
        "text": "Summarize the main events of the Apollo 11 moon landing and its importance in space exploration history."
    },
]

PULL_SECRET_ACCESS_TYPE: str = '["Pull"]'
PULL_SECRET_NAME: str = "oci-registry-pull-secret"
SPYRE_INFERENCE_SERVICE_PORT: int = 8000
SPYRE_CONTAINER_PORT: int = 8000
TIMEOUT_20MIN: int = 30 * 60
OPENAI_ENDPOINT_NAME: str = "openai"
TGIS_ENDPOINT_NAME: str = "tgis"
AUDIO_FILE_URL: str = (
    "https://raw.githubusercontent.com/realpython/python-speech-recognition/master/audio_files/harvard.wav"
)
AUDIO_FILE_LOCAL_PATH: str = "/tmp/harvard.wav"
