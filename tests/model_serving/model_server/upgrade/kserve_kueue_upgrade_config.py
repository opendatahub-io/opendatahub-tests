"""Upgrade test configuration for KServe raw deployment with Kueue integration."""

from utilities.constants import Timeout

KSERVE_KUEUE_QUEUE_LABEL: str = "kueue.x-k8s.io/queue-name"

KSERVE_KUEUE_UPGRADE_NAMESPACE: str = "upgrade-kserve-kueue-raw"
KSERVE_KUEUE_UPGRADE_ISVC_NAME: str = "upgrade-kserve-kueue-isvc"
KSERVE_KUEUE_UPGRADE_RUNTIME_NAME: str = "upgrade-kserve-kueue-runtime"
KSERVE_KUEUE_UPGRADE_S3_SECRET: str = "upgrade-kserve-kueue-connection"

KSERVE_KUEUE_LOCAL_QUEUE: str = "upgrade-kserve-local-queue"
KSERVE_KUEUE_CLUSTER_QUEUE: str = "upgrade-kserve-cluster-queue"
KSERVE_KUEUE_RESOURCE_FLAVOR: str = "upgrade-kserve-flavor"

# Pod resources sized for minimal OVMS raw deployment footprint.
KSERVE_KUEUE_POD_CPU_REQUEST: str = "100m"
KSERVE_KUEUE_POD_MEMORY_REQUEST: str = "1Gi"
KSERVE_KUEUE_POD_CPU_LIMIT: int = 1
KSERVE_KUEUE_POD_MEMORY_LIMIT: str = "2Gi"

# Quota sized so 2 replicas (100m + 1Gi each) exceed memory quota → 1 running, 1 gated
KSERVE_KUEUE_CPU_QUOTA: int = 2
KSERVE_KUEUE_MEMORY_QUOTA: str = "1Gi"

KSERVE_KUEUE_ISVC_RESOURCES: dict[str, dict[str, str | int]] = {
    "requests": {"cpu": KSERVE_KUEUE_POD_CPU_REQUEST, "memory": KSERVE_KUEUE_POD_MEMORY_REQUEST},
    "limits": {"cpu": KSERVE_KUEUE_POD_CPU_LIMIT, "memory": KSERVE_KUEUE_POD_MEMORY_LIMIT},
}

KSERVE_KUEUE_MIN_REPLICAS: int = 1
KSERVE_KUEUE_MAX_REPLICAS: int = 2
KSERVE_KUEUE_SCALED_REPLICAS: int = 2
KSERVE_KUEUE_EXPECTED_RUNNING_PODS: int = 1
KSERVE_KUEUE_EXPECTED_GATED_PODS: int = 1

KSERVE_KUEUE_ISVC_LABELS: dict[str, str] = {KSERVE_KUEUE_QUEUE_LABEL: KSERVE_KUEUE_LOCAL_QUEUE}

UPGRADE_DSCI_SERVICEMESH_STATE_CM_NAME: str = "upgrade-dsci-servicemesh-state"

# Post-upgrade inference uses port-forward; allow extra time after cluster upgrade.
KSERVE_KUEUE_INFERENCE_TIMEOUT: int = Timeout.TIMEOUT_30SEC + Timeout.TIMEOUT_1MIN
