"""Constants for LLMD tests."""

# DSC status condition that gates LLMD test execution
LLMD_DSC_CONDITION: str = "KserveLLMInferenceServiceDependencies"

# Operator CSV prefixes and their install namespaces for health checks
LLMD_REQUIRED_OPERATORS: dict[str, str] = {
    "cert-manager-operator": "cert-manager-operator",
    "authorino-operator": "kuadrant-system",
    "rhcl-operator": "openshift-operators",
}

# Deployments that must be Available for LLMD infrastructure to be healthy
LLMD_REQUIRED_DEPLOYMENTS: dict[str, str] = {
    "cert-manager-operator-controller-manager": "cert-manager-operator",
    "cert-manager": "cert-manager",
    "cert-manager-webhook": "cert-manager",
    "authorino-operator": "kuadrant-system",
    "kuadrant-operator-controller-manager": "kuadrant-system",
}

# KServe + LLMISVC controller deployments checked in the applications namespace
LLMD_KSERVE_CONTROLLER_DEPLOYMENTS: list[str] = [
    "kserve-controller-manager",
    "odh-model-controller",
    "llmisvc-controller-manager",
]
