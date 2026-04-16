import structlog
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.deployment import Deployment
from timeout_sampler import TimeoutSampler

from tests.model_registry.model_catalog.constants import CATALOG_CONTAINER, MODEL_CATALOG_DEPLOYMENT_NAME
from tests.model_registry.utils import execute_get_command

LOGGER = structlog.get_logger(name=__name__)

TEST_SOURCE_ALPHA_ID: str = "test_labeled_alpha"
TEST_SOURCE_BETA_ID: str = "test_labeled_beta"
TEST_MODEL_ALPHA_NAME: str = "qa-team/test-model-alpha"
TEST_MODEL_BETA_NAME: str = "qa-team/test-model-beta"


def build_labeled_configmap_data(source_id: str, source_name: str, model_name: str, cm_name: str) -> dict[str, str]:
    """Build ConfigMap data with sources.yaml and catalog-data.yaml for a labeled catalog source.

    Args:
        source_id: Unique identifier for the catalog source.
        source_name: Human-readable name for the catalog source.
        model_name: Name of the model to include in catalog data.
        cm_name: Name of the ConfigMap, used to construct the catalog path.

    Returns:
        Dictionary with 'sources.yaml' and 'catalog-data.yaml' keys.
    """
    sources_yaml = yaml.dump(
        {
            "catalogs": [
                {
                    "name": source_name,
                    "id": source_id,
                    "enabled": True,
                    "type": "yaml",
                    "properties": {"yamlCatalogPath": f"/data/labeled-sources/{cm_name}/catalog-data.yaml"},
                    "labels": [source_name],
                }
            ],
            "labels": [{"name": source_name, "assetType": "models", "description": f"Test source: {source_name}"}],
        },
        default_flow_style=False,
    )
    catalog_data_yaml = yaml.dump(
        {
            "source": source_name,
            "models": [
                {
                    "name": model_name,
                    "provider": "QA Team",
                    "description": f"Test model for {source_name}",
                    "language": [],
                    "license": "apache-2.0",
                    "tasks": ["text-generation"],
                    "createTimeSinceEpoch": "1713200000000",
                    "lastUpdateTimeSinceEpoch": "1713200000000",
                    "customProperties": {"model_type": {"metadataType": "MetadataStringValue", "string_value": "test"}},
                    "artifacts": [
                        {
                            "uri": f"oci://example.com/{model_name}:latest",
                            "createTimeSinceEpoch": "1713200000000",
                            "lastUpdateTimeSinceEpoch": "1713200000000",
                            "customProperties": {},
                        }
                    ],
                }
            ],
        },
        default_flow_style=False,
    )
    return {"sources.yaml": sources_yaml, "catalog-data.yaml": catalog_data_yaml}


def get_deployment_catalog_args(admin_client: DynamicClient, namespace: str) -> list[str]:
    """Retrieve the container args from the model-catalog deployment.

    Args:
        admin_client: Kubernetes dynamic client.
        namespace: Namespace of the model-catalog deployment.

    Returns:
        List of container argument strings.
    """
    deployment = Deployment(
        name=MODEL_CATALOG_DEPLOYMENT_NAME, namespace=namespace, client=admin_client, ensure_exists=True
    )
    catalog_container = next(
        (
            container
            for container in deployment.instance.spec.template.spec.containers
            if container.name == CATALOG_CONTAINER
        ),
        None,
    )
    assert catalog_container is not None, f"Container {CATALOG_CONTAINER!r} not found in deployment"
    return catalog_container.args or []


def wait_for_deployment_args_contain(
    admin_client: DynamicClient,
    namespace: str,
    expected_substring: str,
    timeout: int = 120,
) -> None:
    """Wait until the model-catalog deployment args contain the expected substring.

    Args:
        admin_client: Kubernetes dynamic client.
        namespace: Namespace of the model-catalog deployment.
        expected_substring: Substring expected to appear in deployment args.
        timeout: Maximum time to wait in seconds.
    """
    for sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=get_deployment_catalog_args,
        admin_client=admin_client,
        namespace=namespace,
    ):
        if any(expected_substring in arg for arg in sample):
            return


def wait_for_deployment_args_not_contain(
    admin_client: DynamicClient,
    namespace: str,
    unwanted_substring: str,
    timeout: int = 120,
) -> None:
    """Wait until the model-catalog deployment args no longer contain the substring.

    Args:
        admin_client: Kubernetes dynamic client.
        namespace: Namespace of the model-catalog deployment.
        unwanted_substring: Substring that should no longer appear in deployment args.
        timeout: Maximum time to wait in seconds.
    """
    for sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=get_deployment_catalog_args,
        admin_client=admin_client,
        namespace=namespace,
    ):
        if not any(unwanted_substring in arg for arg in sample):
            return


def wait_for_source_models_loaded(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    source_id: str,
    timeout: int = 120,
) -> None:
    """Wait until a catalog source has at least one model loaded in the API.

    Args:
        model_catalog_rest_url: List of model catalog REST URL(s).
        model_registry_rest_headers: Headers for API requests.
        source_id: Catalog source ID to check.
        timeout: Maximum time to wait in seconds.
    """
    for sample in TimeoutSampler(
        wait_timeout=timeout,
        sleep=10,
        func=execute_get_command,
        url=f"{model_catalog_rest_url[0]}models?source={source_id}",
        headers=model_registry_rest_headers,
    ):
        if sample.get("size", 0) > 0:
            LOGGER.info(f"Source '{source_id}' has {sample['size']} model(s) loaded")
            return
