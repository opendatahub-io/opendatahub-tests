# Shared by every ai_safety component's session-scoped model resources — must stay a single
# global name. See the shared_models_namespace fixture docstring in tests/ai_safety/conftest.py.
AI_SAFETY_SHARED_MODELS_NAMESPACE: str = "ai-safety-models"

VLLM_EMULATOR: str = "vllm-emulator"
VLLM_EMULATOR_PORT: int = 8000
