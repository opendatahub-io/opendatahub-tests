"""Pytest fixtures for Spark upgrade tests."""

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

from tests.spark.upgrade.utils import (
    SPARK_WORKLOAD_ROLE_NAME,
    SPARK_WORKLOAD_SERVICE_ACCOUNT_NAME,
    capture_spark_application_baseline,
    create_spark_pi_application_spec,
    get_spark_network_policies,
    get_spark_role_bindings,
    get_spark_roles,
    get_spark_service_accounts,
    load_baseline_from_configmap,
    recreate_network_policy_in_namespace,
    recreate_role_binding_in_namespace,
    recreate_role_in_namespace,
    recreate_service_account_in_namespace,
    resubmit_spark_application,
    save_baseline_to_configmap,
)
from utilities.constants import DscComponents
from utilities.infra import create_ns
from utilities.resources.spark_application import SparkApplication

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-spark-operator"


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
) -> DataScienceCluster:
    """Enable Spark Operator in DSC before upgrade tests.

    Spark Operator is Tech Preview and not managed by default.
    This fixture sets it to Managed state for upgrade testing.
    Only runs during pre-upgrade phase.
    """
    # Only enable during pre-upgrade phase
    if pytestconfig.option.post_upgrade:
        return dsc_resource

    original_components = dsc_resource.instance.spec.components
    component_patch = {"sparkoperator": {"managementState": DscComponents.ManagementState.MANAGED}}

    current_state = original_components.get("sparkoperator", {}).get("managementState")
    if current_state == DscComponents.ManagementState.MANAGED:
        raise AssertionError(
            "Spark Operator is already in Managed state. This indicates a previous test did not clean up properly."
        )
    else:
        LOGGER.info("Setting Spark Operator to Managed state")
        editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}})
        editor.update()

        # Wait for Spark Operator to be ready
        LOGGER.info("Waiting for Spark Operator to be ready")
        dsc_resource.wait_for_condition(condition="SparkOperatorReady", status="True", timeout=300)

        return dsc_resource


@pytest.fixture(scope="class")
def post_upgrade_spark_dsc_patch(
    pytestconfig: pytest.Config,
    dsc_resource: DataScienceCluster,
) -> Generator[DataScienceCluster, Any, Any]:
    """Restore Spark Operator to Removed state after new SparkApplication tests.

    Since Spark Operator is Tech Preview, it should be set back to Removed
    state after testing to match the default cluster state.
    Only runs during post-upgrade phase.
    """
    yield dsc_resource

    # Only restore during post-upgrade phase
    if not pytestconfig.option.post_upgrade:
        return

    original_components = dsc_resource.instance.spec.components
    component_patch = {"sparkoperator": {"managementState": DscComponents.ManagementState.REMOVED}}

    current_state = original_components.get("sparkoperator", {}).get("managementState")
    if current_state == DscComponents.ManagementState.REMOVED:
        raise AssertionError(
            "Spark Operator is already in Removed state during post-upgrade. "
            "This indicates Spark Operator was not enabled during pre-upgrade tests."
        )
    else:
        LOGGER.info("Setting Spark Operator back to Removed state")
        editor = ResourceEditor(patches={dsc_resource: {"spec": {"components": component_patch}}})
        editor.update()


@pytest.fixture(scope="session")
def spark_upgrade_baseline_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
) -> dict[str, dict]:
    """Load pre-upgrade baseline values from the cluster ConfigMap.

    Only available during post-upgrade runs. Returns an empty dict during
    pre-upgrade so fixtures that depend on it can be unconditionally wired.
    """
    if not pytestconfig.option.post_upgrade:
        return {}

    return load_baseline_from_configmap(
        client=admin_client,
        namespace=UPGRADE_NAMESPACE,
    )


@pytest.fixture(scope="session")
def spark_namespace_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    teardown_resources: bool,
    pre_upgrade_spark_dsc_patch: DataScienceCluster,
) -> Generator[Namespace, Any, Any]:
    """Create or reference the upgrade namespace.

    Pre-upgrade: Creates fresh namespace (cleans up existing if needed)
    Post-upgrade: References existing namespace and cleans up after tests
    """
    ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        # Post-upgrade: namespace should exist from pre-upgrade
        yield ns
        if teardown_resources:
            ns.clean_up()

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
    """Recreate Spark operator Roles and provide a namespace-scoped Spark Pi workload Role."""
    if pytestconfig.option.post_upgrade:
        stale_roles = get_spark_roles(client=admin_client, namespace=spark_namespace_fixture.name)
        for role in stale_roles:
            role.clean_up()

    apps_namespace = py_config["applications_namespace"]
    source_roles = get_spark_roles(client=admin_client, namespace=apps_namespace)
    assert source_roles, (
        f"No spark-related Roles found in namespace {apps_namespace}. "
        "Ensure the Spark Operator is enabled and has created RBAC resources."
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

    with Role(
        client=admin_client,
        name=SPARK_WORKLOAD_ROLE_NAME,
        namespace=spark_namespace_fixture.name,
        rules=[
            {
                "apiGroups": [""],
                "resources": ["pods", "services", "configmaps"],
                "verbs": ["get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"],
            }
        ],
        teardown=teardown_resources,
    ) as workload_role:
        yield [*created_roles, workload_role]


@pytest.fixture(scope="session")
def service_account_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[list[ServiceAccount], Any, Any]:
    """Discover spark ServiceAccounts from the applications namespace and recreate in upgrade namespace.

    Pre-upgrade: Discovers SAs from operator's namespace, recreates in upgrade namespace
    Post-upgrade: References existing SAs in upgrade namespace
    """
    if pytestconfig.option.post_upgrade:
        stale_sas = get_spark_service_accounts(client=admin_client, namespace=spark_namespace_fixture.name)
        for sa in stale_sas:
            sa.clean_up()

    apps_namespace = py_config["applications_namespace"]
    src_sas = get_spark_service_accounts(client=admin_client, namespace=apps_namespace)
    assert src_sas, (
        f"No spark-related ServiceAccounts found in namespace {apps_namespace}. "
        "Ensure the Spark Operator is enabled and has created RBAC resources."
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
    """Recreate operator RoleBindings and bind Spark Pi permissions to its workload service account."""
    if pytestconfig.option.post_upgrade:
        stale_rbs = get_spark_role_bindings(client=admin_client, namespace=spark_namespace_fixture.name)
        for rb in stale_rbs:
            rb.clean_up()

    apps_namespace = py_config["applications_namespace"]
    source_rbs = get_spark_role_bindings(client=admin_client, namespace=apps_namespace)
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

    with RoleBinding(
        client=admin_client,
        name=SPARK_WORKLOAD_ROLE_NAME,
        namespace=spark_namespace_fixture.name,
        subjects_kind=ServiceAccount.kind,
        subjects_name=spark_workload_service_account.name,
        subjects_namespace=spark_namespace_fixture.name,
        role_ref_kind=Role.kind,
        role_ref_name=SPARK_WORKLOAD_ROLE_NAME,
        teardown=teardown_resources,
    ) as workload_role_binding:
        yield [*created_rbs, workload_role_binding]


@pytest.fixture(scope="session")
def network_policy_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[list[NetworkPolicy], Any, Any]:
    """Discover spark NetworkPolicies from the applications namespace and recreate in upgrade namespace.

    Pre-upgrade: Discovers NetworkPolicies from operator's namespace, recreates in upgrade namespace
    Post-upgrade: References existing NetworkPolicies in upgrade namespace
    """
    if pytestconfig.option.post_upgrade:
        stale_nps = get_spark_network_policies(client=admin_client, namespace=spark_namespace_fixture.name)
        for np in stale_nps:
            np.clean_up()

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


@pytest.fixture(scope="class")
def resubmitted_spark_application_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_application_fixture: SparkApplication,
    teardown_resources: bool,
) -> Generator[SparkApplication | None, Any, Any]:
    """Re-run the pre-upgrade SparkApplication on the upgraded operator.

    Verifies the customer can actively *use* their existing pre-upgrade workload
    after the upgrade, not just that it survived. Deletes the existing (COMPLETED)
    resource and recreates it with the same name and spec so the upgraded operator
    reconciles and runs it again.

    Pre-upgrade: Returns None (only runs post-upgrade)
    Post-upgrade: Re-submits the existing SparkApplication
    """
    if not pytestconfig.option.post_upgrade:
        yield None
        return

    yield resubmit_spark_application(
        client=admin_client,
        spark_app=spark_application_fixture,
        teardown=teardown_resources,
    )


@pytest.fixture(scope="session")
def new_spark_application_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    spark_workload_service_account: ServiceAccount,
    role_binding_fixture: list[RoleBinding],
    network_policy_fixture: list[NetworkPolicy],
    teardown_resources: bool,
) -> Generator[SparkApplication | None, Any, Any]:
    """Create a new SparkApplication post-upgrade to test control plane.

    Pre-upgrade: Returns None (only runs post-upgrade)
    Post-upgrade: Creates a fresh SparkApplication
    """
    if not pytestconfig.option.post_upgrade:
        yield None
        return

    spark_app_name = f"post-upgrade-spark-pi-{shortuuid.uuid()[:8].lower()}"

    spec = create_spark_pi_application_spec(
        name=spark_app_name,
        namespace=spark_namespace_fixture.name,
        service_account=spark_workload_service_account.name,
    )

    # Deploy SparkApplication using kind_dict
    spark_app = SparkApplication(
        client=admin_client,
        kind_dict=spec,
    )
    spark_app.deploy()

    try:
        yield spark_app
    finally:
        if teardown_resources:
            spark_app.clean_up()


def _capture_and_save_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_app: SparkApplication,
) -> None:
    """Capture SparkApplication baseline values and persist to ConfigMap.

    No-op during post-upgrade runs.
    """
    if pytestconfig.option.post_upgrade:
        return

    baselines = {
        spark_app.name: capture_spark_application_baseline(
            client=admin_client,
            spark_app=spark_app,
        ),
    }
    save_baseline_to_configmap(
        client=admin_client,
        namespace=UPGRADE_NAMESPACE,
        baselines=baselines,
    )


@pytest.fixture(scope="session")
def spark_capture_upgrade_baseline(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_application_fixture: SparkApplication,
) -> None:
    """Capture baseline values for the SparkApplication."""
    _capture_and_save_baseline(
        pytestconfig=pytestconfig,
        admin_client=admin_client,
        spark_app=spark_application_fixture,
    )
