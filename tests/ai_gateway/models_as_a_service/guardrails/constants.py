"""Constants for MaaS guardrails (AIGuardrail -> Praxis ai_guardrails -> NeMo) e2e tests."""

ECHO_SIM_PORT: int = 8000

NEMO_PII_CONFIG_ID: str = "pii-detect"
NEMO_PHASE_CONFIG_ID: str = "phase-detect"

AI_GUARDRAIL_TIMEOUT: str = "5s"
INPUT_CHECK_NAME: str = "pii-input"
OUTPUT_CHECK_NAME: str = "pii-output"

GUARDRAIL_CHECK_HEADER_PREFIX: str = "x-aigateway-guardrail-"

# NeMo reports the blocking flow name as the rail; Praxis returns it as the 403 body (input)
# or inside the replaced error message (output).
INPUT_RAIL_FLOW: str = "detect sensitive data on input"
OUTPUT_RAIL_FLOW: str = "detect sensitive data on output"

GUARDRAIL_VIOLATION_ERROR_CODE: str = "content_blocked"
STREAMING_WITH_OUTPUT_GUARDRAIL_STATUS: int = 422
STREAMING_WITH_OUTPUT_GUARDRAIL_MESSAGE: str = (
    "ai_guardrails: streaming responses are not supported while a response guardrail is active"
)

PII_EMAIL: str = "john.doe@example.com"
PII_PROMPT: str = f"Please send the quarterly report to {PII_EMAIL} today."
CLEAN_PROMPT: str = "Please summarize the attached report briefly."

GUARDRAIL_ENFORCEMENT_TIMEOUT_SECONDS: int = 600

MAAS_API_SELECTION_PATH: str = "/internal/v1/subscriptions/select"
