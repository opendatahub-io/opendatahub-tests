"""Constants for Kueue upgrade tests.

Resource names used by fixtures and test assertions.
Kept in a shared module so both conftest.py and test files
reference a single source of truth.
"""

UPGRADE_KUEUE_NAMESPACE: str = "upgrade-kueue-workbenches"
UPGRADE_KUEUE_NOTEBOOK_NAME: str = "upgrade-kueue-notebook"
UPGRADE_KUEUE_STOPPED_NOTEBOOK_NAME: str = "upgrade-kueue-stopped"
NEW_KUEUE_NOTEBOOK_NAME: str = "upgrade-kueue-new"
UPGRADE_KUEUE_BASELINE_CM_NAME: str = "upgrade-kueue-baseline"
UPGRADE_KUEUE_LOCAL_QUEUE_NAME: str = "upgrade-kueue-local-queue"
UPGRADE_KUEUE_CLUSTER_QUEUE_NAME: str = "upgrade-kueue-cluster-queue"
UPGRADE_KUEUE_RESOURCE_FLAVOR_NAME: str = "upgrade-kueue-flavor"
UPGRADE_KUEUE_HARDWARE_PROFILE_NAME: str = "upgrade-kueue-hwp"
