import structlog
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.resource import Resource
from ocp_resources.subscription import Subscription
from pytest_testconfig import config as py_config

from utilities.constants import QUAY_IMAGE_CHANNELS, QUAY_REGISTRY
from utilities.infra import get_product_version

LOGGER = structlog.get_logger(name=__name__)

OPERATOR_SUBSCRIPTION_PREFIXES: tuple[str, ...] = ("rhods-operator", "rhoai-operator", "opendatahub-operator")


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


def get_operator_subscription(admin_client: DynamicClient) -> Subscription:
    """Get the RHOAI/ODH operator subscription.

    Args:
        admin_client: The kubernetes client

    Returns:
        The operator Subscription

    Raises:
        ResourceNotFoundError: If no operator subscription is found
    """
    namespace = py_config["operator_namespace"]

    for subscription in Subscription.get(client=admin_client, namespace=namespace):
        if subscription.name.startswith(OPERATOR_SUBSCRIPTION_PREFIXES):
            return subscription

    raise ResourceNotFoundError(
        f"No subscription starting with {OPERATOR_SUBSCRIPTION_PREFIXES} found in namespace {namespace}"
    )


def get_operator_channel(admin_client: DynamicClient) -> str:
    """Get the channel the RHOAI/ODH operator is subscribed to, e.g. `odh-stable`, `stable`, `fast`.

    Args:
        admin_client: The kubernetes client

    Returns:
        The subscription channel
    """
    channel = get_operator_subscription(admin_client=admin_client).instance.spec.channel
    LOGGER.info(f"Operator subscription channel: {channel}")
    return channel


def get_expected_image_registry(admin_client: DynamicClient) -> str:
    """Get the registry the product images are expected to be served from.

    Builds installed from `odh-stable` and other channels listed in `QUAY_IMAGE_CHANNELS` are published
    to quay.io only; released channels (`stable`, `fast`, `eus-*`) are mirrored to registry.redhat.io.

    Args:
        admin_client: The kubernetes client

    Returns:
        The expected registry host
    """
    channel = get_operator_channel(admin_client=admin_client)

    if channel in QUAY_IMAGE_CHANNELS:
        LOGGER.warning(f"Operator is installed from the {channel} channel; images are expected from {QUAY_REGISTRY}")
        return QUAY_REGISTRY

    return Resource.ApiGroup.IMAGE_REGISTRY
