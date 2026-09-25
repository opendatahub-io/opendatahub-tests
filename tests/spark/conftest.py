"""Shared fixtures for Spark install and upgrade tests."""

from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount
from pytest_testconfig import config as py_config

from tests.spark.utils import (
    SPARK_WORKLOAD_ROLE_BINDING_NAME,
    SPARK_WORKLOAD_ROLE_NAME,
    SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME,
    UPGRADE_NAMESPACE,
    create_spark_pi_application_spec,
    get_spark_network_policies,
    get_spark_role_bindings,
    get_spark_roles,
    get_spark_service_accounts,
    recreate_network_policy_in_namespace,
    recreate_role_binding_in_namespace,
    recreate_role_in_namespace,
    recreate_service_account_in_namespace,
)
from utilities.constants import DscComponents
from utilities.infra import create_ns
from utilities.resources.spark_application import SparkApplication

LOGGER = structlog.get_logger(name=__name__)


@pytest.fixture(scope="session")
def dsc_resource(  # noqa: UFN001
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    """Reset Spark before and after pre-upgrade runs when resource deletion is requested."""
    if (
        not pytestconfig.option.delete_pre_upgrade_resources
        or not pytestconfig.option.pre_upgrade
        or pytestconfig.option.post_upgrade
    ):
        yield dsc_resource
        return

    namespace = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)
    if namespace.exists:
        LOGGER.info(f"Deleting previous Spark test namespace {UPGRADE_NAMESPACE}")
        namespace.clean_up(wait=True, timeout=300)

    LOGGER.info("Resetting Spark Operator to Removed before pre-upgrade setup")
    editor = ResourceEditor(
        patches={
            dsc_resource: {
                "spec": {"components": {"sparkoperator": {"managementState": DscComponents.ManagementState.REMOVED}}}
            }
        }
    )
    editor.update()
    dsc_resource.wait_for_condition(condition="SparkOperatorReady", status="False", reason="Removed", timeout=300)
    try:
        yield dsc_resource
    finally:
        LOGGER.info("Restoring Spark Operator to Removed after pre-upgrade cleanup")
        editor.update()
        dsc_resource.wait_for_condition(condition="SparkOperatorReady", status="False", reason="Removed", timeout=300)


@pytest.fixture(scope="session")
def pre_upgrade_spark_dsc_patch(
    pytestconfig: pytest.Config,
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    """Enable Spark for pre-upgrade or install tests, restoring its original state after install tests."""
    if pytestconfig.option.post_upgrade:
        yield dsc_resource
        return

    current_state = dsc_resource.instance.spec.components.get("sparkoperator", {}).get("managementState")
    component_patch = {"sparkoperator": {"managementState": DscComponents.ManagementState.MANAGED}}
    editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}})

    if not pytestconfig.option.pre_upgrade:
        # An install smoke run must leave the operator's original configuration intact.
        with editor:
            dsc_resource.wait_for_condition(condition="SparkOperatorReady", status="True", timeout=300)
            yield dsc_resource
        if current_state == DscComponents.ManagementState.REMOVED:
            dsc_resource.wait_for_condition(
                condition="SparkOperatorReady", status="False", reason="Removed", timeout=300
            )
        return

    assert current_state != DscComponents.ManagementState.MANAGED, (
        "Spark Operator is already in Managed state. This indicates a previous test did not clean up properly."
    )
    LOGGER.info("Setting Spark Operator to Managed state")
    editor.update()
    dsc_resource.wait_for_condition(condition="SparkOperatorReady", status="True", timeout=300)
    yield dsc_resource


@pytest.fixture(scope="session")
def spark_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
    pre_upgrade_spark_dsc_patch: DataScienceCluster,
) -> Generator[Namespace, Any, Any]:
    """Create or reference the upgrade namespace.

    Pre-upgrade: Creates fresh namespace (cleans up existing if needed)
    Post-upgrade: Deletes the namespace before removing Spark Operator after all selected tests
    """
    ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        try:
            yield ns
        finally:
            if teardown_resources:
                LOGGER.info("Deleting Spark test namespace before removing Spark Operator")
                ns.clean_up(wait=True, timeout=300)
                LOGGER.info("Restoring Spark Operator to Removed after post-upgrade cleanup")
                ResourceEditor(
                    patches={
                        pre_upgrade_spark_dsc_patch: {
                            "spec": {
                                "components": {
                                    "sparkoperator": {"managementState": DscComponents.ManagementState.REMOVED}
                                }
                            }
                        }
                    }
                ).update()
                pre_upgrade_spark_dsc_patch.wait_for_condition(
                    condition="SparkOperatorReady", status="False", reason="Removed", timeout=300
                )

    else:
        # Pre-upgrade: namespace should NOT exist from previous runs
        if ns.exists:
            raise AssertionError(
                f"Namespace {UPGRADE_NAMESPACE} already exists. "
                "This indicates a previous test run did not clean up properly."
            )

        with create_ns(
            admin_client=admin_client,
            name=UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def spark_role_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[list[Role], Any, Any]:
    """Copy only spark-role from the applications namespace into the test namespace."""
    if pytestconfig.option.post_upgrade:
        resources = get_spark_roles(client=admin_client, namespace=spark_namespace_fixture.name)
        assert resources, "Required pre-upgrade resources are missing from the test namespace"
        yield resources
        return

    apps_namespace = py_config["applications_namespace"]
    source_roles = get_spark_roles(client=admin_client, namespace=apps_namespace)
    assert any(role.name == SPARK_WORKLOAD_ROLE_NAME for role in source_roles), (
        f"Required Spark workload Role {SPARK_WORKLOAD_ROLE_NAME} was not found in namespace {apps_namespace}"
    )
    LOGGER.info(f"Discovered {len(source_roles)} spark Role(s) in {apps_namespace}: {[r.name for r in source_roles]}")

    created_roles = []
    for source_role in source_roles:
        role = recreate_role_in_namespace(
            client=admin_client,
            source_role=source_role,
            target_namespace=spark_namespace_fixture.name,
            teardown=teardown_resources,
        )
        created_roles.append(role)

    yield created_roles


@pytest.fixture(scope="session")
def service_account_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[list[ServiceAccount], Any, Any]:
    """Copy only spark-operator-spark from the applications namespace into the test namespace."""
    if pytestconfig.option.post_upgrade:
        resources = get_spark_service_accounts(client=admin_client, namespace=spark_namespace_fixture.name)
        assert resources, "Required pre-upgrade resources are missing from the test namespace"
        yield resources
        return

    apps_namespace = py_config["applications_namespace"]
    src_sas = get_spark_service_accounts(client=admin_client, namespace=apps_namespace)
    assert src_sas, (
        f"Required Spark workload ServiceAccount {SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME} "
        f"was not found in namespace {apps_namespace}"
    )
    LOGGER.info(f"Discovered {len(src_sas)} spark ServiceAccount(s) in {apps_namespace}: {[sa.name for sa in src_sas]}")

    created_sas = []
    for source_sa in src_sas:
        sa = recreate_service_account_in_namespace(
            client=admin_client,
            source_sa=source_sa,
            target_namespace=spark_namespace_fixture.name,
            teardown=teardown_resources,
        )
        created_sas.append(sa)

    yield created_sas


@pytest.fixture(scope="session")
def spark_workload_service_account(service_account_fixture: list[ServiceAccount]) -> ServiceAccount:
    """Select the Spark workload service account by name, independent of discovery order."""
    for service_account in service_account_fixture:
        if service_account.name == SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME:
            return service_account
    raise AssertionError(f"Required Spark workload ServiceAccount {SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME} was not found")


@pytest.fixture(scope="session")
def role_binding_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    spark_workload_service_account: ServiceAccount,
    spark_role_fixture: list[Role],
    teardown_resources: bool,
) -> Generator[list[RoleBinding], Any, Any]:
    """Copy only spark-role-binding from the applications namespace into the test namespace."""
    if pytestconfig.option.post_upgrade:
        resources = get_spark_role_bindings(client=admin_client, namespace=spark_namespace_fixture.name)
        assert resources, "Required pre-upgrade resources are missing from the test namespace"
        yield resources
        return

    apps_namespace = py_config["applications_namespace"]
    source_rbs = get_spark_role_bindings(client=admin_client, namespace=apps_namespace)
    assert any(role_binding.name == SPARK_WORKLOAD_ROLE_BINDING_NAME for role_binding in source_rbs), (
        f"Required Spark workload RoleBinding {SPARK_WORKLOAD_ROLE_BINDING_NAME} "
        f"was not found in namespace {apps_namespace}"
    )
    LOGGER.info(
        f"Discovered {len(source_rbs)} spark RoleBinding(s) in {apps_namespace}: {[rb.name for rb in source_rbs]}"
    )

    created_rbs = []
    for source_rb in source_rbs:
        rb = recreate_role_binding_in_namespace(
            client=admin_client,
            source_rb=source_rb,
            target_namespace=spark_namespace_fixture.name,
            teardown=teardown_resources,
        )
        created_rbs.append(rb)

    yield created_rbs


@pytest.fixture(scope="session")
def network_policy_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[list[NetworkPolicy], Any, Any]:
    """Copy only spark-operator-allow-internal from the applications namespace into the test namespace."""
    if pytestconfig.option.post_upgrade:
        resources = get_spark_network_policies(client=admin_client, namespace=spark_namespace_fixture.name)
        assert resources, "Required pre-upgrade resources are missing from the test namespace"
        yield resources
        return

    apps_namespace = py_config["applications_namespace"]
    source_nps = get_spark_network_policies(client=admin_client, namespace=apps_namespace)
    LOGGER.info(
        f"Discovered {len(source_nps)} spark NetworkPolicy(s) in {apps_namespace}: {[np.name for np in source_nps]}"
    )

    created_nps = []
    for source_np in source_nps:
        np = recreate_network_policy_in_namespace(
            client=admin_client,
            source_np=source_np,
            target_namespace=spark_namespace_fixture.name,
            teardown=teardown_resources,
        )
        created_nps.append(np)

    yield created_nps


@pytest.fixture(scope="session")
def spark_application_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    spark_workload_service_account: ServiceAccount,
    role_binding_fixture: list[RoleBinding],
    network_policy_fixture: list[NetworkPolicy],
    teardown_resources: bool,
) -> Generator[SparkApplication, Any, Any]:
    """Create or reference a SparkApplication for upgrade testing.

    Pre-upgrade: Creates SparkApplication with spark-pi workload
    Post-upgrade: References existing SparkApplication and cleans up
    """
    spark_app_name = "upgrade-spark-pi"

    spark_app_kwargs = {
        "client": admin_client,
        "name": spark_app_name,
        "namespace": spark_namespace_fixture.name,
    }

    spark_app = SparkApplication(**spark_app_kwargs)

    if pytestconfig.option.post_upgrade:
        yield spark_app
    else:
        spec = create_spark_pi_application_spec(
            name=spark_app_name,
            namespace=spark_namespace_fixture.name,
            service_account=spark_workload_service_account.name,
        )

        spark_app_instance = SparkApplication(
            client=admin_client,
            kind_dict=spec,
        )
        spark_app_instance.deploy()

        yield spark_app_instance


@pytest.fixture(scope="session")
def spark_pi_application(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    spark_workload_service_account: ServiceAccount,
    role_binding_fixture: list[RoleBinding],
    network_policy_fixture: list[NetworkPolicy],
    teardown_resources: bool,
    spark_application_fixture: SparkApplication,
) -> Generator[SparkApplication, Any, Any]:
    """Reuse the pre-upgrade Pi setup or create a fresh Pi application after upgrade."""
    if not pytestconfig.option.post_upgrade:
        yield spark_application_fixture
        return

    spark_app_name = f"post-upgrade-spark-pi-{shortuuid.uuid()[:8].lower()}"

    spec = create_spark_pi_application_spec(
        name=spark_app_name,
        namespace=spark_namespace_fixture.name,
        service_account=spark_workload_service_account.name,
    )

    with SparkApplication(client=admin_client, kind_dict=spec, teardown=teardown_resources) as spark_app:
        yield spark_app
