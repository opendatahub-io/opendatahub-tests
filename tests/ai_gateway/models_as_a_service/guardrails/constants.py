"""Constants for MaaS guardrails (AIGuardrail -> Praxis ai_guardrails -> NeMo) e2e tests."""

ECHO_SIM_PORT: int = 8000

NEMO_PII_CONFIG_ID: str = "pii-detect"
NEMO_PHASE_CONFIG_ID: str = "phase-detect"
NEMO_AI_GATEWAY_MODE_LABEL = "aiGatewayMode"

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

NEMO_AUTH_ANNOTATION: str = "security.opendatahub.io/enable-auth"
NEMO_CHECKS_PATH: str = "/v1/checks"

# ai-gateway-controller only discovers a NeMo Service in one of these namespaces.
AI_GATEWAY_INFRA_NAMESPACES: tuple[str, ...] = ("odh-ai-gateway-infra", "redhat-ai-gateway-infra")
ACCEPTED_CONDITION: str = "Accepted"
PROVIDER_READY_CONDITION: str = "ProviderReady"
PROVIDER_AVAILABLE_REASON: str = "ProviderAvailable"

# With API translation the post-auth ExtProc streams responses, which Praxis cannot screen,
# so ai-gateway-controller refuses AIGuardrails that offer an Output-phase check.
OUTPUT_CHECKS_UNSUPPORTED_REASON: str = "OutputChecksUnsupported"
OUTPUT_CHECKS_UNSUPPORTED_MESSAGE: str = "API translation is enabled for this tenant"

# An authenticated callout mounts the post-auth pod's ServiceAccount token and the OpenShift service CA here.
NEMO_AUTH_VOLUME_NAME: str = "nemo-auth"
NEMO_CALLER_TOKEN_FILE: str = "/var/run/secrets/nemo/token"  # pragma: allowlist secret
NEMO_CALLER_SERVICE_CA_FILE: str = "/var/run/secrets/nemo/service-ca.crt"
