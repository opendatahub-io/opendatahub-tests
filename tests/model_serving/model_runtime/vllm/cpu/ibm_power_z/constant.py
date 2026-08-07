from typing import Any

GRANITE_4_1_8B_MODEL_PATH: str = "models/granite-4.1-8b"
FALCON3_7B_INSTRUCT_MODEL_PATH: str = "models/Falcon3-7B-Instruct"
LLAMA_3_2_1B_INSTRUCT_MODEL_PATH: str = "models/llama-32-1b-instruct"
PHI_4_MODEL_PATH: str = "models/phi-4"
MISTRAL_7B_INSTRUCT_MODEL_PATH: str = "models/Mistral-7B-v0.3"
GRANITE_3_1_8B_INSTRUCT_MODEL_PATH: str = "models/granite-3.1-8b-instruct"
DEEPSEEK_R1_DISTILL_LLAMA_8B_MODEL_PATH: str = "models/deepseek-r1-distill-llama-8b"
ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT_MODEL_PATH: str = "models/ELYZA-japanese-Llama-2-7b-instruct"
MINISTRAL_3B_INSTRUCT_MODEL_PATH: str = "models/ministral-3b-instruct"
GRANITE_3B_CODE_INSTRUCT_2K_MODEL_PATH: str = "models/granite-3b-code-instruct-2k"

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

# ELYZA is based on Llama-2 which has no chat_template in tokenizer_config.json.
# vLLM rejects /v1/chat/completions with HTTP 400 unless an explicit template is
# supplied.  The vLLM CPU image ships template_chatml.jinja at /app/data/template/.
ELYZA_SERVING_ARGUMENT: list[str] = [
    "--dtype=bfloat16",
    "--model=/mnt/models",
    "--max-model-len=256",
    "--max-num-seqs=1",
    "--max-num-batched-tokens=256",
    "--chat-template=/app/data/template/template_chatml.jinja",
    "--uvicorn-log-level=debug",
]
