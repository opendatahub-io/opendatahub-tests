from typing import Any

DEEPSEEK_R1_DISTILL_LLAMA_8B_MODEL_PATH: str = "models/deepseek-r1-distill-llama-8b"
ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT_MODEL_PATH: str = "models/ELYZA-japanese-Llama-2-7b-instruct"
FALCON3_7B_INSTRUCT_MODEL_PATH: str = "models/Falcon3-7B-Instruct"
GRANITE_3B_CODE_INSTRUCT_2K_MODEL_PATH: str = "models/granite-3b-code-instruct-2k"
GRANITE_3_1_2B_INSTRUCT_MODEL_PATH: str = "models/granite-3.1-2b-instruct"
GRANITE_3_1_8B_INSTRUCT_MODEL_PATH: str = "models/granite-3.1-8b-instruct"
GRANITE_3_3_8B_INSTRUCT_MODEL_PATH: str = "models/granite-3.3-8b-instruct"
LLAMA_3_2_1B_INSTRUCT_MODEL_PATH: str = "models/llama-32-1b-instruct"
LLAMA_3_2_3B_INSTRUCT_MODEL_PATH: str = "models/llama-32-3b-instruct"
META_LLAMA_3_1_8B_INSTRUCT_MODEL_PATH: str = "models/Meta-Llama-3.1-8B-Instruct"
MINISTRAL_3B_INSTRUCT_MODEL_PATH: str = "models/ministral-3b-instruct"
MISTRAL_7B_INSTRUCT_MODEL_PATH: str = "models/Mistral-7B-v0.3"
PHI_3_MINI_4K_INSTRUCT_MODEL_PATH: str = "models/Phi-3-mini-4k-instruct"
PHI_4_MODEL_PATH: str = "models/phi-4"
TINYLLAMA_1_1B_CHAT_V1_0_MODEL_PATH: str = "models/tinyllama-1.1b-chat-v1.0"

IBM_POWER_Z_MODEL_ENV_VARIABLES: list[dict[str, str]] = [
    {"name": "VLLM_CPU_KVCACHE_SPACE", "value": "8"},
    {"name": "VLLM_WORKER_MULTIPROC_METHOD", "value": "spawn"},
    {"name": "OMP_NUM_THREADS", "value": "8"},
]

IBM_POWER_Z_PREDICT_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "16", "memory": "70Gi"},
    "limits": {"cpu": "16", "memory": "70Gi"},
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

# ---------------------------------------------------------------------------
# ELYZA-japanese-Llama-2-7b-instruct specific arguments
# ---------------------------------------------------------------------------
# ELYZA is based on Llama-2 which ships without a chat_template in
# tokenizer_config.json.  vLLM rejects /v1/chat/completions with HTTP 400
# unless an explicit chat template is supplied at startup.  The vLLM CPU
# image ships template_chatml.jinja at /app/data/template/.
ELYZA_SERVING_ARGUMENT: list[str] = [
    "--dtype=bfloat16",
    "--model=/mnt/models",
    "--max-model-len=256",
    "--max-num-seqs=1",
    "--max-num-batched-tokens=256",
    "--chat-template=/app/data/template/template_chatml.jinja",
    "--uvicorn-log-level=debug",
]
