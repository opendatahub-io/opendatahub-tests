from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.infra import get_product_version

LOGGER = get_logger(name=__name__)


def get_cluster_service_version(client: DynamicClient, prefix: str, namespace: str) -> ClusterServiceVersion:
    csvs = ClusterServiceVersion.get(client=client, namespace=namespace)
    LOGGER.info(f"Looking for {prefix} CSV in namespace {namespace}")
    matching_csvs = [csv for csv in csvs if csv.name.startswith(prefix)]

    if not matching_csvs:
        raise ResourceNotFoundError(f"No ClusterServiceVersion found starting with prefix '{prefix}'")

    if len(matching_csvs) > 1:
        raise ResourceNotUniqueError(
            f"Multiple ClusterServiceVersions found"
            f" starting with prefix '{prefix}':"
            f" {[csv.name for csv in matching_csvs]}"
        )
    LOGGER.info(f"Found cluster service version: {matching_csvs[0].name}")
    return matching_csvs[0]


def get_csv_related_images(admin_client: DynamicClient, csv_name: str | None = None) -> list[dict[str, str]]:
    """Get relatedImages from the CSV.

    Args:
        admin_client: The kubernetes client
        csv_name: Optional CSV name. If not provided, will use {operator_name}.{version}
                 where operator_name is determined by the distribution (rhods-operator for OpenShift AI,
                 opendatahub-operator for Open Data Hub)

    Returns:
        List of related images from the CSV
    """

    if csv_name is None:
        distribution = py_config["distribution"]
        operator_name = "opendatahub-operator" if distribution == "upstream" else "rhods-operator"
        csv_name = f"{operator_name}.{get_product_version(admin_client=admin_client)}"

    return get_cluster_service_version(
        client=admin_client,
        prefix=csv_name,
        namespace=py_config["applications_namespace"],
    ).instance.spec.relatedImages
