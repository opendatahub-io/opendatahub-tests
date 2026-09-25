from typing import Any

GRANITE_4_1_8B_MODEL_PATH: str = "granite-4.1-8b"
FALCON3_7B_INSTRUCT_MODEL_PATH: str = "Falcon3-7B-Instruct"
LLAMA_3_2_1B_INSTRUCT_MODEL_PATH: str = "llama-32-1b-instruct"
PHI_4_MODEL_PATH: str = "phi-4"
MISTRAL_7B_INSTRUCT_MODEL_PATH: str = "mistral_7b"
GRANITE_3_1_8B_INSTRUCT_MODEL_PATH: str = "granite-3.1-8b-instruct-r241212a"
DEEPSEEK_R1_DISTILL_LLAMA_8B_MODEL_PATH: str = "deepseek-ai"
ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT_MODEL_PATH: str = "ELYZA-japanese-Llama-2-7b-instruct"
MINISTRAL_3B_INSTRUCT_MODEL_PATH: str = "ministral"
GRANITE_3B_CODE_INSTRUCT_2K_MODEL_PATH: str = "granite-3b-code-instruct"

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
