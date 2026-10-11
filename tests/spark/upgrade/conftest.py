"""Pytest fixtures for Spark upgrade tests."""

from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.namespace import Namespace
from ocp_resources.resource import ResourceEditor
from ocp_resources.role import Role
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount

from tests.spark.upgrade.utils import (
    capture_spark_application_baseline,
    create_spark_pi_application_spec,
    load_baseline_from_configmap,
    save_baseline_to_configmap,
)
from utilities.constants import DscComponents
from utilities.infra import create_ns
from utilities.resources.spark_application import SparkApplication

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-spark-operator"
SPARK_SERVICE_ACCOUNT = "spark-operator-spark"


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
) -> Generator[Role, Any, Any]:
    """Create or reference the Spark Role with necessary permissions.

    Pre-upgrade: Creates Role
    Post-upgrade: References existing Role and cleans up
    """
    role_kwargs = {
        "client": admin_client,
        "name": "spark-operator-role",
        "namespace": spark_namespace_fixture.name,
    }

    role = Role(**role_kwargs)

    if pytestconfig.option.post_upgrade:
        yield role
    else:
        role_instance = Role(
            **role_kwargs,
            rules=[
                {
                    "apiGroups": [""],
                    "resources": ["pods", "services", "configmaps"],
                    "verbs": ["create", "get", "list", "watch", "delete", "patch", "update"],
                },
                {
                    "apiGroups": [""],
                    "resources": ["pods/log"],
                    "verbs": ["get"],
                },
            ],
            teardown=teardown_resources,
        )
        role_instance.deploy()
        yield role_instance


@pytest.fixture(scope="session")
def service_account_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    spark_role_fixture: Role,
    teardown_resources: bool,
) -> Generator[ServiceAccount, Any, Any]:
    """Create or reference the Spark service account with RoleBinding.

    Pre-upgrade: Creates service account and RoleBinding
    Post-upgrade: References existing service account and cleans up
    """
    sa_kwargs = {
        "client": admin_client,
        "name": SPARK_SERVICE_ACCOUNT,
        "namespace": spark_namespace_fixture.name,
    }

    sa = ServiceAccount(**sa_kwargs)

    if pytestconfig.option.post_upgrade:
        yield sa
    else:
        sa = ServiceAccount(**sa_kwargs, teardown=teardown_resources)
        sa.deploy()

        # Create RoleBinding
        rb = RoleBinding(
            client=admin_client,
            name="spark-operator-rolebinding",
            namespace=spark_namespace_fixture.name,
            subjects_kind="ServiceAccount",
            subjects_name=SPARK_SERVICE_ACCOUNT,
            role_ref_kind="Role",
            role_ref_name=spark_role_fixture.name,
            teardown=teardown_resources,
        )
        rb.deploy()

        yield sa


@pytest.fixture(scope="session")
def spark_application_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    service_account_fixture: ServiceAccount,
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
        # Create SparkApplication spec
        spec = create_spark_pi_application_spec(
            name=spark_app_name,
            namespace=spark_namespace_fixture.name,
            service_account=service_account_fixture.name,
        )

        # Deploy SparkApplication using kind_dict
        spark_app_instance = SparkApplication(
            client=admin_client,
            kind_dict=spec,
        )
        spark_app_instance.deploy()

        yield spark_app_instance


@pytest.fixture(scope="session")
def new_spark_application_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    service_account_fixture: ServiceAccount,
    teardown_resources: bool,
) -> Generator[SparkApplication | None, Any, Any]:
    """Create a new SparkApplication post-upgrade to test control plane.

    Pre-upgrade: Returns None (only runs post-upgrade)
    Post-upgrade: Creates a fresh SparkApplication
    """
    if not pytestconfig.option.post_upgrade:
        yield None
        return

    # Generate unique name for post-upgrade test (lowercase for RFC 1123)
    spark_app_name = f"post-upgrade-spark-pi-{shortuuid.uuid()[:8].lower()}"

    # Create SparkApplication spec
    spec = create_spark_pi_application_spec(
        name=spark_app_name,
        namespace=spark_namespace_fixture.name,
        service_account=service_account_fixture.name,
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
