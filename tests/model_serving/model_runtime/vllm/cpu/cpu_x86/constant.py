from typing import Any

OPT_125M_MODEL_PATH: str = "opt-125m"

CPU_X86_ENV_VARIABLES: list[dict[str, str]] = [
    {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "4"},
    {"name": "VLLM_WORKER_MULTIPROC_METHOD", "value": "spawn"},
    {"name": "OMP_NUM_THREADS", "value": "8"},
]

CPU_X86_SERVING_ARGUMENT: list[str] = [
    "--model=/mnt/models",
    "--uvicorn-log-level=debug",
    "--enforce-eager",
    "--max-model-len=256",
    "--max-num-seqs=20",
]

CPU_X86_PREDICT_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "8", "memory": "10Gi"},
    "limits": {"cpu": "16", "memory": "16Gi"},
}

CPU_X86_VOLUMES: list[dict[str, str | dict[str, str]]] = [
    {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "32Gi"}},
    {"name": "tmp", "emptyDir": {}},
    {"name": "home", "emptyDir": {}},
]

CPU_X86_VOLUME_MOUNTS: list[dict[str, str]] = [
    {"name": "shared-memory", "mountPath": "/dev/shm"},
    {"name": "tmp", "mountPath": "/tmp"},
    {"name": "home", "mountPath": "/home/vllm"},
]

OPT_125M_COMPLETION_REQUEST: dict[str, Any] = {
    "prompt": "What is Kubernetes?",
    "max_tokens": 50,
}
