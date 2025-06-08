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
