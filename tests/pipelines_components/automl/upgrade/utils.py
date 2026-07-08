import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_BASELINE_CONFIGMAP = "automl-upgrade-baseline"


def save_baseline_to_configmap(
    client: DynamicClient,
    namespace: str,
    baselines: dict,
) -> None:
    """Save baseline data to a ConfigMap for post-upgrade verification."""
    LOGGER.info(f"Saving baseline to ConfigMap {UPGRADE_BASELINE_CONFIGMAP} in namespace {namespace}")

    cm = ConfigMap(
        client=client,
        name=UPGRADE_BASELINE_CONFIGMAP,
        namespace=namespace,
        data={"baselines.yaml": yaml.dump(baselines)},
    )

    if cm.exists:
        raise AssertionError(
            f"ConfigMap {UPGRADE_BASELINE_CONFIGMAP} already exists in namespace {namespace}. "
            "This indicates a previous test run did not clean up properly."
        )

    cm.deploy()
    LOGGER.info("Baseline saved to ConfigMap")


def load_baseline_from_configmap(
    client: DynamicClient,
    namespace: str,
) -> dict:
    """Load baseline data from ConfigMap saved during pre-upgrade."""
    LOGGER.info(f"Loading baseline from ConfigMap {UPGRADE_BASELINE_CONFIGMAP} in namespace {namespace}")

    cm = ConfigMap(
        client=client,
        name=UPGRADE_BASELINE_CONFIGMAP,
        namespace=namespace,
    )

    if not cm.exists:
        raise AssertionError(
            f"Baseline ConfigMap {UPGRADE_BASELINE_CONFIGMAP} does not exist in namespace {namespace}. "
            "Cannot load baseline for post-upgrade verification."
        )

    baseline_yaml = cm.instance.data.get("baselines.yaml", "")
    baselines = yaml.safe_load(baseline_yaml) or {}

    LOGGER.info(f"Loaded baseline: {baselines}")
    return baselines
