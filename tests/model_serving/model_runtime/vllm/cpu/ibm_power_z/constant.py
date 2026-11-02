from typing import Any

from utilities.constants import AcceleratorType

GRANITE_4_1_8B_MODEL_PATH: str = "granite-4.1-8b"
FALCON3_7B_INSTRUCT_MODEL_PATH: str = "Falcon3-7B-Instruct"
LLAMA_3_2_1B_INSTRUCT_MODEL_PATH: str = "llama-32-1b-instruct"
PHI_4_MODEL_PATH: str = "phi-4"
MISTRAL_7B_INSTRUCT_MODEL_PATH: str = "mistral_7b/Mistral-7B-Instruct-v0.1"
GRANITE_3_1_8B_INSTRUCT_MODEL_PATH: str = "granite-3.1-8b-instruct-r241212a"
DEEPSEEK_R1_DISTILL_LLAMA_8B_MODEL_PATH: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ELYZA_JAPANESE_LLAMA_2_7B_INSTRUCT_MODEL_PATH: str = "ELYZA-japanese-Llama-2-7b-instruct"
MINISTRAL_3B_INSTRUCT_MODEL_PATH: str = "ministral/Ministral-3b-instruct"
GRANITE_3B_CODE_INSTRUCT_2K_MODEL_PATH: str = "granite-3b-code-instruct"

IBM_POWER_Z_PREDICT_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "12", "memory": "64Gi"},
    "limits": {"cpu": "12", "memory": "64Gi"},
}

# IBM Z clusters are typically compact (schedulable control-plane nodes), leaving
# less headroom than the Power sizing above assumes. Requests are kept small so the
# predictor schedules on any node of a 3-node compact cluster, where requests==limits
# at Power sizing leaves the pod Pending on all but the largest node.
#
# The CPU limit must not go below 12. vLLM kills the engine if a worker takes
# longer than VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (300s default) for a single model
# step (EngineDeadError -> HTTP 500 -> container exit), and on 8 cores an 8B model
# took 303s for one prefill step. At 12 it clears prefill and decodes normally.
# Memory is not the binding constraint - a 7-8B model peaks around 16-17Gi - but
# the limit keeps headroom for the larger models in the suite.
IBM_Z_PREDICT_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": "4", "memory": "32Gi"},
    "limits": {"cpu": "12", "memory": "48Gi"},
}

# Power and Z share this suite but not its sizing; accelerators absent from this
# map fall back to the Power defaults.
PREDICT_RESOURCES_MAP: dict[str, dict[str, dict[str, str]]] = {
    AcceleratorType.CPU_Z: IBM_Z_PREDICT_RESOURCES,
}

IBM_POWER_Z_SERVING_ARGUMENT: list[str] = [
    "--dtype=bfloat16",
    "--model=/mnt/models",
    "--max-model-len=256",
    "--max-num-seqs=1",
    "--max-num-batched-tokens=256",
    "--uvicorn-log-level=debug",
]

# Z decode throughput on a compact cluster is a fraction of Power's, so a long
# completion can outrun the 300s Power route timeout. The HAProxy route annotation
# and this client-side budget have to be widened together, or requests.post gives up
# while the route is still open. On Power the 300s route timeout still cuts first,
# so the wider client budget is a no-op there.
IBM_POWER_Z_INFERENCE_TIMEOUT_SECONDS: int = 900

# CPU inference on IBM Power is slow; the default HAProxy route timeout is 30s
# which is not enough for it. The ISVC is annotated so KServe propagates this to
# the Route it creates. Z needs the wider budget above for the same reason.
IBM_POWER_ROUTE_TIMEOUT: str = "300s"
IBM_Z_ROUTE_TIMEOUT: str = f"{IBM_POWER_Z_INFERENCE_TIMEOUT_SECONDS}s"

ROUTE_TIMEOUT_MAP: dict[str, str] = {
    AcceleratorType.CPU_Z: IBM_Z_ROUTE_TIMEOUT,
}

# Readiness budget for create_isvc, which covers pod creation, weight loading and
# vLLM's torch.compile warmup. Warmup scales with model size on Z - roughly 13min
# for a 7B and 22min for an 8B - so a 14B model can still be compiling when the
# 1800s Power default expires.
IBM_POWER_Z_ISVC_TIMEOUT: int = 1800
IBM_Z_ISVC_TIMEOUT: int = 3600

ISVC_TIMEOUT_MAP: dict[str, int] = {
    AcceleratorType.CPU_Z: IBM_Z_ISVC_TIMEOUT,
}

# These tests assert only that the completion is non-empty, so the token count is
# pure cost. Decode measures 0.1-0.3 tok/s on s390x, where 50 tokens could outrun
# the budget above; 10 keeps a 7B model's response near 80s instead of 240s.
IBM_POWER_Z_CHAT_MAX_TOKENS: int = 10

IBM_POWER_Z_CHAT_INFERENCE_REQUEST: dict[str, Any] = {
    "messages": [{"role": "user", "content": "What is Kubernetes?"}],
    "max_tokens": IBM_POWER_Z_CHAT_MAX_TOKENS,
    "request_timeout": IBM_POWER_Z_INFERENCE_TIMEOUT_SECONDS,
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
