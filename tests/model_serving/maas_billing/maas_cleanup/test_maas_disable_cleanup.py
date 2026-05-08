import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.cluster_role import ClusterRole
from ocp_resources.cluster_role_binding import ClusterRoleBinding
from ocp_resources.custom_resource_definition import CustomResourceDefinition
from ocp_resources.deployment import Deployment
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config

from tests.model_serving.maas_billing.maas_cleanup.utils import (
    MAAS_CONTROLLER_RESOURCE_NAME,
    get_surviving_maas_controller_resources,
)
from utilities.constants import ApiGroups

MAAS_CRDS: tuple[str, ...] = (
    f"maasmodelrefs.{ApiGroups.MAAS_IO}",
    f"maasauthpolicies.{ApiGroups.MAAS_IO}",
    f"maassubscriptions.{ApiGroups.MAAS_IO}",
    f"tenants.{ApiGroups.MAAS_IO}",
)


@pytest.mark.tier3
@pytest.mark.usefixtures("maas_disabled_for_cleanup_test")
class TestMaaSDisableCleanup:
    """Verify MaaS controller bundle is fully cleaned up when managementState is set to Removed."""

    def test_disable_maas_removes_controller_deployment(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify maas-controller Deployment is deleted after managementState is set to Removed."""
        applications_namespace = py_config["applications_namespace"]
        controller_deployment = Deployment(
            client=admin_client,
            name=MAAS_CONTROLLER_RESOURCE_NAME,
            namespace=applications_namespace,
        )
        assert controller_deployment.wait_deleted(timeout=600), (
            f"maas-controller Deployment still exists in namespace '{applications_namespace}'"
            f" after managementState set to Removed"
        )

    def test_disable_maas_removes_crds(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify all MaaS CRDs are removed after managementState is set to Removed."""
        surviving_crds = [
            crd_name for crd_name in MAAS_CRDS if CustomResourceDefinition(client=admin_client, name=crd_name).exists
        ]
        assert not surviving_crds, (
            f"MaaS CRDs still present after managementState set to Removed: {', '.join(surviving_crds)}"
        )

    def test_disable_maas_removes_cluster_rbac(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify MaaS ClusterRole, ClusterRoleBinding, and ServiceAccount are removed after disable."""
        applications_namespace = py_config["applications_namespace"]
        rbac_resources = {
            f"ClusterRole/{MAAS_CONTROLLER_RESOURCE_NAME}": ClusterRole(
                client=admin_client, name=MAAS_CONTROLLER_RESOURCE_NAME
            ),
            f"ClusterRoleBinding/{MAAS_CONTROLLER_RESOURCE_NAME}": ClusterRoleBinding(
                client=admin_client, name=MAAS_CONTROLLER_RESOURCE_NAME
            ),
            f"ServiceAccount/{MAAS_CONTROLLER_RESOURCE_NAME}": ServiceAccount(
                client=admin_client,
                name=MAAS_CONTROLLER_RESOURCE_NAME,
                namespace=applications_namespace,
            ),
        }
        surviving_rbac = [label for label, resource in rbac_resources.items() if resource.exists]
        assert not surviving_rbac, (
            f"MaaS RBAC resources still present after managementState set to Removed: {', '.join(surviving_rbac)}"
        )

    def test_disable_maas_no_bundle_resources_remain(
        self,
        admin_client: DynamicClient,
    ) -> None:
        """Verify no maas-controller bundle resources remain after managementState is set to Removed."""
        applications_namespace = py_config["applications_namespace"]
        surviving_resources = get_surviving_maas_controller_resources(
            admin_client=admin_client,
            applications_namespace=applications_namespace,
            crd_names=MAAS_CRDS,
        )
        assert not surviving_resources, (
            f"maas-controller bundle resources still present after managementState set to Removed:"
            f" {', '.join(surviving_resources)}"
        )
