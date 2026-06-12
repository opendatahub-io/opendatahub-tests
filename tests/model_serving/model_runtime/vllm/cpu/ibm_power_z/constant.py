from typing import Any

FALCON3_7B_INSTRUCT_MODEL_PATH: str = "models/Falcon3-7B-Instruct"
LLAMA_3_2_1B_INSTRUCT_MODEL_PATH: str = "models/llama-32-1b-instruct"
PHI_4_MODEL_PATH: str = "models/phi-4"
MISTRAL_7B_INSTRUCT_MODEL_PATH: str = "models/Mistral-7B-v0.3"
GRANITE_3_1_8B_INSTRUCT_MODEL_PATH: str = "models/granite-3.1-8b-instruct"

IBM_POWER_Z_PREDICT_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "12", "memory": "64Gi"},
    "limits": {"cpu": "12", "memory": "64Gi"},
}

IBM_POWER_Z_SERVING_ARGUMENT: list[str] = [
    "--dtype=bfloat16",
    "--model=/mnt/models",
    "--max-model-len=256",
    "--max-num-seqs=1",
    "--max-num-batched-tokens=256",
    "--uvicorn-log-level=debug",
]

IBM_POWER_Z_CHAT_INFERENCE_REQUEST: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    "max_tokens": 50,
}
