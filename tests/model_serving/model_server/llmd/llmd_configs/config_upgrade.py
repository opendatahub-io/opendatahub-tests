"""Upgrade test configurations — auth + Kueue integration for upgrade validation."""

from utilities.constants import Labels

from .config_models import TinyLlamaOciConfig

LLMD_KUEUE_LOCAL_QUEUE = "upgrade-llmd-local-queue"
LLMD_KUEUE_CLUSTER_QUEUE = "upgrade-llmd-cluster-queue"
LLMD_KUEUE_RESOURCE_FLAVOR = "upgrade-llmd-flavor"
LLMD_KUEUE_CPU_QUOTA = 3
LLMD_KUEUE_MEMORY_QUOTA = "20Gi"


class UpgradeAuthKueueConfig(TinyLlamaOciConfig):
    """TinyLlama via OCI with auth enabled and Kueue queue label, for upgrade tests."""

    name = "llmisvc-auth-and-kueue"
    enable_auth = True

    @classmethod
    def container_resources(cls):
        """Sized so 2 replicas exceed the Kueue quota (cpu: 3) → 1 running, 1 gated."""
        return {
            "requests": {"cpu": "2", "memory": "6Gi"},
            "limits": {"cpu": "3", "memory": "20Gi"},
        }

    @classmethod
    def labels(cls):
        return {Labels.Kueue.QUEUE_NAME: LLMD_KUEUE_LOCAL_QUEUE}
