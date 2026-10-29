TRUSTYAI_MODULE_NAME: str = "default-trustyai"
TRUSTYAI_MODULE_CRD_NAME: str = "trustyais.components.platform.opendatahub.io"
TRUSTYAI_MODULE_OPERATOR_DEPLOYMENT: str = "trustyai-operator-module-controller-manager"
TRUSTYAI_WORKLOAD_OPERATOR_DEPLOYMENT: str = "trustyai-service-operator-controller-manager"
TRUSTYAI_MODULE_OPERATOR_NETWORK_POLICY: str = "trustyai-operator-module-controller-manager-allow-traffic"
TRUSTYAI_MODULE_OPERATOR_IDENTITY_LABEL: dict[str, str] = {"app.kubernetes.io/name": "trustyai-operator-module"}
TRUSTYAI_MODULE_OPERATOR_PROBE_PORT: int = 8081
TRUSTYAI_PLATFORM_CONFIGMAP: str = "odh-trustyai-config"
TRUSTYAI_DSC_CONFIGMAP: str = "trustyai-dsc-config"

CONDITION_READY: str = "Ready"
CONDITION_PROVISIONING_SUCCEEDED: str = "ProvisioningSucceeded"
