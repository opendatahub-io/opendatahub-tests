import os
from typing import Any, Union
from utilities.constants import Protocols, KServeDeploymentType, RuntimeTemplates  ,AcceleratorType,  Labels

ACCELERATOR_IDENTIFIER: dict[str, str] = {
    AcceleratorType.NVIDIA: Labels.Nvidia.NVIDIA_COM_GPU,
    AcceleratorType.AMD: "amd.com/gpu"
}

TEMPLATE_MAP: dict[str, str] = {
    AcceleratorType.NVIDIA: RuntimeTemplates.VLLM_CUDA,
    AcceleratorType.AMD: RuntimeTemplates.VLLM_ROCM
}

TEMPLATE_MAP: dict[str, str] = {
    Protocols.REST: RuntimeTemplates.VLLM_CUDA,
    
}

PULL_SECRET_ACCESS_TYPE: str = "WyJQdWxsIl0="   # PULL_SECRET_ACCESS_TYPE is base64 encoded "Pull" 

COMPLETION_QUERY: list[dict[str, str]] = [
    {
        "text": "What are the advantages of using open source software?",
    }
]

BASE_SEVERRLESS_DEPLOYMENT_CONFIG: dict[str, Any] = {
    "deployment_type": KServeDeploymentType.SERVERLESS,
    "min-replicas": 1,
    "enable_external_route": True
}