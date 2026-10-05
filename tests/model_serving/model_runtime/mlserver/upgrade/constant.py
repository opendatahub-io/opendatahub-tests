"""Constants shared between upgrade conftest and upgrade tests."""

UPGRADE_NAMESPACE: str = "mlserver-cpu-upgrade"
UPGRADE_SA_NAME: str = "mlserver-upgrade-sa"
UPGRADE_SECRET_NAME: str = "mlserver-upgrade-s3"
UPGRADE_ISVC_NAME: str = "sklearn"
UPGRADE_BASELINE_CM: str = "mlserver-upgrade-baseline"
UPGRADE_RESTART_KEY: str = "predictor_restart_counts"
