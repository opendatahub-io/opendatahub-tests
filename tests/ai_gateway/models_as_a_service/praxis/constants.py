"""Constants for Praxis per-tenant payload-processing tests."""

PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION: str = "maas.opendatahub.io/payload-processing-type"
PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE: str = "praxis"
PRAXIS_AITENANT_CLEANUP_FINALIZER: str = "ai-gateway-controller.opendatahub.io/praxis-cleanup"

# Legacy IPP resource base names (aligned with maas-controller tenantreconcile constants).
LEGACY_IPP_POST_PROCESSING_NAME_BASE: str = "payload-processing"
LEGACY_IPP_PRE_PROCESSING_NAME_BASE: str = "payload-pre-processing"
LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE: str = "payload-processing-plugins"
LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY: str = "custom-ipp-config.yaml"

DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS: int = 300
LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT_SECONDS: int = 600
LEGACY_IPP_POLL_INTERVAL_SECONDS: int = 5
DEFAULT_PRAXIS_FINALIZER_WAIT_TIMEOUT_SECONDS: int = 120
