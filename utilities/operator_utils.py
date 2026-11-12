import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.resource import Resource
from pytest_testconfig import config as py_config

from utilities.constants import QUAY_REGISTRY

LOGGER = structlog.get_logger(name=__name__)


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
        csv_name: Optional CSV name. If not provided, the product CSV is looked up by its operator name
                 prefix, which is determined by the distribution (rhods-operator for OpenShift AI,
                 opendatahub-operator for Open Data Hub)

    Returns:
        List of related images from the CSV
    """

    if csv_name is None:
        distribution = py_config["distribution"]
        # Match on the operator name only: the version suffix is not written consistently across builds
        # (OLM names ODH CSVs `<operator>.v<version>` while RHOAI CSVs omit the `v`), and there is exactly
        # one product CSV in the applications namespace.
        csv_name = "opendatahub-operator." if distribution == "upstream" else "rhods-operator."

    return get_cluster_service_version(
        client=admin_client,
        prefix=csv_name,
        namespace=py_config["applications_namespace"],
    ).instance.spec.relatedImages


def get_expected_image_registry() -> str:
    """Get the registry the product images are expected to be served from.

    Open Data Hub builds, including midstream pre-release builds, publish their component images to
    quay.io; released OpenShift AI builds are mirrored to registry.redhat.io.

    Returns:
        The expected registry host
    """
    if py_config["distribution"] == "upstream":
        LOGGER.info(f"Open Data Hub distribution; images are expected from {QUAY_REGISTRY}")
        return QUAY_REGISTRY

    return Resource.ApiGroup.IMAGE_REGISTRY
