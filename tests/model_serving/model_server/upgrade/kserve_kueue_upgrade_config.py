"""Upgrade test configuration for KServe raw deployment with Kueue integration."""

from utilities.constants import Labels, Timeout

KSERVE_KUEUE_UPGRADE_NAMESPACE: str = "upgrade-kserve-kueue-raw"
KSERVE_KUEUE_UPGRADE_ISVC_NAME: str = "upgrade-kserve-kueue-isvc"
KSERVE_KUEUE_UPGRADE_RUNTIME_NAME: str = "upgrade-kserve-kueue-runtime"
KSERVE_KUEUE_UPGRADE_S3_SECRET: str = "upgrade-kserve-kueue-connection"

KSERVE_KUEUE_LOCAL_QUEUE: str = "upgrade-kserve-local-queue"
KSERVE_KUEUE_CLUSTER_QUEUE: str = "upgrade-kserve-cluster-queue"
KSERVE_KUEUE_RESOURCE_FLAVOR: str = "upgrade-kserve-flavor"

# Pod resources sized for minimal OVMS raw deployment footprint.
# Kueue admits on the sum of ALL container requests in the pod. Current raw
# predictor pods charge only the model container (100m / 1Gi); do not assume
# kube-rbac-proxy overhead when sizing quota.
KSERVE_KUEUE_POD_CPU_REQUEST: str = "100m"
KSERVE_KUEUE_POD_MEMORY_REQUEST: str = "1Gi"
KSERVE_KUEUE_POD_CPU_LIMIT: str = "1"
KSERVE_KUEUE_POD_MEMORY_LIMIT: str = "2Gi"

# Quota sized so 1 pod fits and 2 pods do not:
#   1 pod  = 1Gi  ≤  1Gi  → admitted / running
#   2 pods = 2Gi  >  1Gi  → second replica stays gated
# (same pattern as test_kueue_isvc_raw.py: request N, quota between N and 2N)
KSERVE_KUEUE_CPU_QUOTA: int = 2
KSERVE_KUEUE_MEMORY_QUOTA: str = "1Gi"

KSERVE_KUEUE_ISVC_RESOURCES: dict[str, dict[str, str]] = {
    "requests": {"cpu": KSERVE_KUEUE_POD_CPU_REQUEST, "memory": KSERVE_KUEUE_POD_MEMORY_REQUEST},
    "limits": {"cpu": KSERVE_KUEUE_POD_CPU_LIMIT, "memory": KSERVE_KUEUE_POD_MEMORY_LIMIT},
}

KSERVE_KUEUE_MIN_REPLICAS: int = 1
KSERVE_KUEUE_MAX_REPLICAS: int = 2
KSERVE_KUEUE_SCALED_REPLICAS: int = 2
KSERVE_KUEUE_EXPECTED_RUNNING_PODS: int = 1
KSERVE_KUEUE_EXPECTED_GATED_PODS: int = 1

KSERVE_KUEUE_ISVC_LABELS: dict[str, str] = {Labels.Kueue.QUEUE_NAME: KSERVE_KUEUE_LOCAL_QUEUE}


# Post-upgrade inference uses the external route; allow extra time after cluster upgrade.
KSERVE_KUEUE_INFERENCE_TIMEOUT: int = Timeout.TIMEOUT_5MIN
