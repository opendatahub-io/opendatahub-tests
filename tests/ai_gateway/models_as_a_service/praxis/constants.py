"""Constants for Praxis per-tenant payload-processing tests."""

PRAXIS_PAYLOAD_PROCESSING_TYPE_ANNOTATION: str = "maas.opendatahub.io/payload-processing-type"
PRAXIS_PAYLOAD_PROCESSING_TYPE_VALUE: str = "praxis"
PRAXIS_AITENANT_CLEANUP_FINALIZER: str = "ai-gateway-controller.opendatahub.io/praxis-cleanup"

# Legacy IPP resource base names (aligned with maas-controller tenantreconcile constants).
LEGACY_IPP_POST_PROCESSING_NAME_BASE: str = "payload-processing"
LEGACY_IPP_PRE_PROCESSING_NAME_BASE: str = "payload-pre-processing"
LEGACY_IPP_PLUGINS_CONFIGMAP_NAME_BASE: str = "payload-processing-plugins"
LEGACY_IPP_CUSTOM_CONFIG_DATA_KEY: str = "custom-ipp-config.yaml"
LEGACY_IPP_CUSTOM_PRE_CONFIG_DATA_KEY: str = "custom-pre-processing-ipp-config.yaml"
LEGACY_POST_PROCESSING_CONTAINER_CONFIG_ARG: str = "/config/custom-ipp-config.yaml"
LEGACY_PRE_PROCESSING_CONTAINER_CONFIG_ARG: str = "/config/custom-pre-processing-ipp-config.yaml"

# Praxis extproc plugins (ai-gateway-controller praxis-extproc overlay; see pkg/tenant/constants.go).
PRAXIS_EXTPROC_CONFIG_DATA_KEY: str = "extproc.yaml"
PRAXIS_PRE_EXTPROC_CONFIG_DATA_KEY: str = "pre-extproc.yaml"
PRAXIS_POST_PROCESSING_CONTAINER_CONFIG_ARG: str = "/etc/praxis/extproc.yaml"
PRAXIS_PRE_PROCESSING_CONTAINER_CONFIG_ARG: str = "/etc/praxis/pre-extproc.yaml"

# MaaS → ai-gateway handoff before Praxis SSA apply (MaasTenantConfig / AITenant annotations).
MAAS_IPP_RESOURCES_RELEASED_CONDITION: str = "IPPResourcesReleased"
MAAS_IPP_MIGRATION_CLEANUP_COMPLETE_ANNOTATION: str = "maas.opendatahub.io/ipp-migration-cleanup-complete"

DEFAULT_LEGACY_IPP_WAIT_TIMEOUT_SECONDS: int = 300
LEGACY_IPP_SWITCH_BACK_WAIT_TIMEOUT_SECONDS: int = 600
LEGACY_IPP_POLL_INTERVAL_SECONDS: int = 5
DEFAULT_PRAXIS_FINALIZER_WAIT_TIMEOUT_SECONDS: int = 120
