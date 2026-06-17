"""Pytest fixtures for Spark upgrade tests."""

from collections.abc import Generator
from typing import Any

import pytest
import shortuuid
import structlog
from kubernetes.dynamic import DynamicClient
from ocp_resources.namespace import Namespace
from ocp_resources.service_account import ServiceAccount

from tests.spark.upgrade.utils import (
    capture_spark_application_baseline,
    create_spark_pi_application_spec,
    load_baseline_from_configmap,
    save_baseline_to_configmap,
)
from utilities.infra import create_ns
from utilities.resources.spark_application import SparkApplication

LOGGER = structlog.get_logger(name=__name__)

UPGRADE_NAMESPACE = "upgrade-spark-operator"
SPARK_SERVICE_ACCOUNT = "spark-operator-spark"


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
) -> Generator[Namespace, Any, Any]:
    """Create or reference the upgrade namespace.

    Pre-upgrade: Creates namespace with teardown=False
    Post-upgrade: References existing namespace and cleans up after tests
    """
    ns = Namespace(client=admin_client, name=UPGRADE_NAMESPACE)

    if pytestconfig.option.post_upgrade:
        yield ns
        ns.clean_up()

    else:
        with create_ns(
            admin_client=admin_client,
            name=UPGRADE_NAMESPACE,
            model_mesh_enabled=False,
            add_dashboard_label=True,
            teardown=teardown_resources,
        ) as ns:
            yield ns


@pytest.fixture(scope="session")
def service_account_fixture(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ServiceAccount, Any, Any]:
    """Create or reference the Spark service account.

    Pre-upgrade: Creates service account
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
        sa.clean_up()

    else:
        with ServiceAccount(**sa_kwargs, teardown=teardown_resources) as sa:
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
        spark_app.clean_up()

    else:
        # Create SparkApplication spec
        spec = create_spark_pi_application_spec(
            name=spark_app_name,
            namespace=spark_namespace_fixture.name,
            service_account=service_account_fixture.name,
        )

        # Deploy SparkApplication
        spark_app_instance = SparkApplication(**spark_app_kwargs)
        spark_app_instance.deploy(body=spec)

        try:
            yield spark_app_instance
        finally:
            if teardown_resources:
                spark_app_instance.clean_up()


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

    # Generate unique name for post-upgrade test
    spark_app_name = f"post-upgrade-spark-pi-{shortuuid.uuid()[:8]}"

    spark_app_kwargs = {
        "client": admin_client,
        "name": spark_app_name,
        "namespace": spark_namespace_fixture.name,
    }

    # Create SparkApplication spec
    spec = create_spark_pi_application_spec(
        name=spark_app_name,
        namespace=namespace_fixture.name,
        service_account=service_account_fixture.name,
    )

    # Deploy SparkApplication
    spark_app = SparkApplication(**spark_app_kwargs)
    spark_app.deploy(body=spec)

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
