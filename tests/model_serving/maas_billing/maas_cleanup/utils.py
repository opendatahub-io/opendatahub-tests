from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role import ClusterRole
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.deployment import Deployment
from ocp_resources.service_account import ServiceAccount

MAAS_CONTROLLER_RESOURCE_NAME = "maas-controller"


def get_surviving_maas_controller_resources(
    admin_client: DynamicClient,
    applications_namespace: str,
) -> list[str]:
    """Return 'Kind/name' strings for any maas-controller bundle resources still present on the cluster.

    MaaS CRDs are intentionally retained after disable and are not checked here.
    """
    surviving: list[str] = []

    resource_checks: list[tuple[str, bool]] = [
        (
            f"Deployment/{MAAS_CONTROLLER_RESOURCE_NAME}",
            Deployment(
                client=admin_client,
                name=MAAS_CONTROLLER_RESOURCE_NAME,
                namespace=applications_namespace,
            ).exists,
        ),
        (
            f"ServiceAccount/{MAAS_CONTROLLER_RESOURCE_NAME}",
            ServiceAccount(
                client=admin_client,
                name=MAAS_CONTROLLER_RESOURCE_NAME,
                namespace=applications_namespace,
            ).exists,
        ),
        (
            f"ClusterRole/{MAAS_CONTROLLER_RESOURCE_NAME}",
            ClusterRole(
                client=admin_client,
                name=MAAS_CONTROLLER_RESOURCE_NAME,
            ).exists,
        ),
        (
            f"ClusterRoleBinding/{MAAS_CONTROLLER_RESOURCE_NAME}",
            ClusterRoleBinding(
                client=admin_client,
                name=MAAS_CONTROLLER_RESOURCE_NAME,
            ).exists,
        ),
    ]

    for resource_label, exists in resource_checks:
        if exists:
            surviving.append(resource_label)

    return surviving
