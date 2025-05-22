from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError, ResourceNotUniqueError
from simple_logger.logger import get_logger

from ocp_resources.cluster_service_version import ClusterServiceVersion
from ocp_resources.subscription import Subscription
from utilities.exceptions import ResourceMismatchError

LOGGER = get_logger(name=__name__)


def get_cluster_service_version(client: DynamicClient, prefix: str, namespace: str) -> ClusterServiceVersion:
    csvs = ClusterServiceVersion.get(dyn_client=client, namespace=namespace)
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


def validate_operator_subscription_channel(
    client: DynamicClient, operator_name: str, namespace: str, channel_name: str
) -> None:
    operator_subscriptions = [
        subscription
        for subscription in Subscription.get(dyn_client=client, namespace=namespace)
        if subscription.instance.spec.name == operator_name
    ]

    if not operator_subscriptions:
        raise ResourceNotFoundError(f"Subscription not found for operator {operator_name}")
    if len(operator_subscriptions) > 1:
        raise ResourceNotUniqueError(f"Multiple subscriptions found for operator {operator_name}")
    subscription_channel = operator_subscriptions[0].instance.spec.channel
    if not subscription_channel or subscription_channel != channel_name:
        raise ResourceMismatchError(
            f"For Operator {operator_name}, Subscription points to {subscription_channel}, expected: {channel_name}"
        )
    LOGGER.info(f"Operator {operator_name} subscription channel is {subscription_channel}")
