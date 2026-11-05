"""Fixtures for Spark execution continuity and resubmission across upgrades."""

import json
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.namespace import Namespace
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.role_binding import RoleBinding
from ocp_resources.service_account import ServiceAccount

from tests.spark.upgrade.utils import resubmit_spark_application
from tests.spark.utils import (
    CONSUMER_NETWORK_POLICY_NAME,
    consumer_network_policy_manifest,
    create_gated_continuity_spark_spec,
    create_spark_connect_spec,
    ensure_consumer_network_policy,
    operand_network_policies_present,
)
from utilities.resources.scheduled_spark_application import ScheduledSparkApplication
from utilities.resources.spark_application import SparkApplication
from utilities.resources.spark_connect import SparkConnect


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
    if pytestconfig.option.post_upgrade:
        yield resubmit_spark_application(
            client=admin_client,
            spark_app=spark_application_fixture,
            teardown=teardown_resources,
        )
    else:
        yield None


@contextmanager
def _spark_continuity_control(
    admin_client: DynamicClient,
    namespace: str,
    name: str,
    post_upgrade: bool,
    teardown: bool,
) -> Generator[ConfigMap, Any, Any]:
    """Create or reference a continuity-control ConfigMap holding the workload, release gate, and baseline.

    Each continuity test needs its own control ConfigMap so their release gates and baselines do not
    collide when both run in the same upgrade session.

    Args:
        admin_client: Kubernetes client.
        namespace: Namespace to create or reference the ConfigMap in.
        name: ConfigMap name.
        post_upgrade: Whether the post-upgrade lane is selected (reference instead of create).
        teardown: Whether to clean up the ConfigMap on teardown.

    Yields:
        ConfigMap: The continuity-control ConfigMap.
    """
    control = ConfigMap(client=admin_client, name=name, namespace=namespace)
    if post_upgrade:
        assert control.exists, "Run the in-progress pre-upgrade test and preserve its resources first"
        yield control
    else:
        with ConfigMap(
            client=admin_client,
            name=name,
            namespace=namespace,
            data={"workload.py": Path(__file__).with_name("continuity_workload.py").read_text(), "release": "false"},
            teardown=teardown,
        ) as control:
            yield control


@pytest.fixture()
def spark_continuity_control(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ConfigMap, Any, Any]:
    """Provide the directly-created job's workload script, release gate, and execution baseline."""
    with _spark_continuity_control(
        admin_client=admin_client,
        namespace=spark_namespace_fixture.name,
        name="spark-upgrade-continuity",
        post_upgrade=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    ) as control:
        yield control


@pytest.fixture()
def scheduled_spark_continuity_control(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ConfigMap, Any, Any]:
    """Provide the scheduled job's workload script, release gate, and execution baseline."""
    with _spark_continuity_control(
        admin_client=admin_client,
        namespace=spark_namespace_fixture.name,
        name="scheduled-spark-upgrade-continuity",
        post_upgrade=pytestconfig.option.post_upgrade,
        teardown=teardown_resources,
    ) as control:
        yield control


@pytest.fixture()
def spark_continuity_application(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_continuity_control: ConfigMap,
    spark_workload_service_account: ServiceAccount,
    role_binding_fixture: list[RoleBinding],
    teardown_resources: bool,
) -> Generator[SparkApplication, Any, Any]:
    """Create before upgrade, or reference after upgrade, the gated Spark execution."""
    application = SparkApplication(
        client=admin_client, name="upgrade-spark-in-progress", namespace=spark_continuity_control.namespace
    )
    if pytestconfig.option.post_upgrade:
        assert application.exists, "The in-progress SparkApplication did not survive"
        yield application
    else:
        manifest = {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "SparkApplication",
            "metadata": {"name": application.name, "namespace": application.namespace},
            "spec": create_gated_continuity_spark_spec(
                name=application.name,
                namespace=application.namespace,
                service_account=spark_workload_service_account.name,
                control_config_map_name=spark_continuity_control.name,
            ),
        }
        with SparkApplication(client=admin_client, kind_dict=manifest, teardown=teardown_resources) as application:
            yield application


@pytest.fixture()
def scheduled_continuity_application(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    scheduled_spark_continuity_control: ConfigMap,
    spark_workload_service_account: ServiceAccount,
    role_binding_fixture: list[RoleBinding],
    teardown_resources: bool,
) -> Generator[ScheduledSparkApplication, Any, Any]:
    """Create before upgrade, or reference after upgrade, the gated scheduled Spark execution.

    The ScheduledSparkApplication mints a child run from the same release-gated workload used by
    ``spark_continuity_application``. ``Forbid`` keeps exactly one run in flight so the held run is
    deterministically the one adopted by the upgraded operator.
    """
    scheduled_application = ScheduledSparkApplication(
        client=admin_client,
        name="upgrade-scheduled-spark-in-progress",
        namespace=scheduled_spark_continuity_control.namespace,
    )
    if pytestconfig.option.post_upgrade:
        assert scheduled_application.exists, "The in-progress ScheduledSparkApplication did not survive"
        yield scheduled_application
    else:
        manifest = {
            "apiVersion": "sparkoperator.k8s.io/v1beta2",
            "kind": "ScheduledSparkApplication",
            "metadata": {"name": scheduled_application.name, "namespace": scheduled_application.namespace},
            "spec": {
                "schedule": "@every 2m",
                "concurrencyPolicy": "Forbid",
                "successfulRunHistoryLimit": 5,
                "failedRunHistoryLimit": 5,
                "template": create_gated_continuity_spark_spec(
                    name=scheduled_application.name,
                    namespace=scheduled_application.namespace,
                    service_account=spark_workload_service_account.name,
                    control_config_map_name=scheduled_spark_continuity_control.name,
                ),
            },
        }
        with ScheduledSparkApplication(
            client=admin_client, kind_dict=manifest, teardown=teardown_resources
        ) as scheduled_application:
            yield scheduled_application


@pytest.fixture()
def spark_connect_continuity_control(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ConfigMap, Any, Any]:
    """Persist the pre-upgrade Spark Connect server identity across the two pytest processes."""
    control = ConfigMap(
        client=admin_client, name="spark-connect-upgrade-continuity", namespace=spark_namespace_fixture.name
    )
    if pytestconfig.option.post_upgrade:
        assert control.exists, "Run the in-progress pre-upgrade test and preserve its resources first"
        yield control
    else:
        with ConfigMap(
            client=admin_client, name=control.name, namespace=control.namespace, data={}, teardown=teardown_resources
        ) as control:
            yield control


@pytest.fixture()
def spark_connect_continuity_application(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    spark_workload_service_account: ServiceAccount,
    role_binding_fixture: list[RoleBinding],
    teardown_resources: bool,
) -> Generator[SparkConnect, Any, Any]:
    """Create before upgrade, or reference after upgrade, a long-lived Spark Connect server.

    The server is held running across the operator upgrade so the post-upgrade test can prove the
    upgraded operator adopts the same server pod (strong continuity) and it still serves queries.
    """
    spark_connect = SparkConnect(
        client=admin_client, name="upgrade-spark-connect", namespace=spark_namespace_fixture.name
    )
    if pytestconfig.option.post_upgrade:
        assert spark_connect.exists, "The Spark Connect server did not survive the upgrade"
        yield spark_connect
    else:
        manifest = create_spark_connect_spec(
            name=spark_connect.name,
            namespace=spark_connect.namespace,
            service_account=spark_workload_service_account.name,
        )
        with SparkConnect(client=admin_client, kind_dict=manifest, teardown=teardown_resources) as spark_connect:
            yield spark_connect


@pytest.fixture()
def spark_connect_consumer_network_policy(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    spark_connect_continuity_application: SparkConnect,
    spark_connect_continuity_control: ConfigMap,
    teardown_resources: bool,
) -> Generator[NetworkPolicy | None, Any, Any]:
    """Keep the continuity Spark Connect server reachable by its client across the upgrade.

    The policy is needed only once the operator enforces operand NetworkPolicies. Pre-upgrade it is
    created only if the feature is already active (and then persists across the upgrade, since it is
    user-managed and not operator-GC'd); yields None otherwise. Post-upgrade, where the operator always
    enforces the policies, it asserts the policy survived an already-active feature, or creates it when
    the feature turns on during the upgrade.

    Yields:
        NetworkPolicy | None: The consumer policy, or None when operand policies are absent pre-upgrade.
    """
    namespace = spark_namespace_fixture.name
    policy = NetworkPolicy(client=admin_client, name=CONSUMER_NETWORK_POLICY_NAME, namespace=namespace)

    if not pytestconfig.option.post_upgrade:
        with ensure_consumer_network_policy(
            client=admin_client, namespace=namespace, teardown=teardown_resources
        ) as policy:
            yield policy
    else:
        baseline = json.loads(spark_connect_continuity_control.instance.data["baseline"])
        if baseline.get("operand_policies_present"):
            # The feature was already active before the upgrade: the user-managed policy must survive
            # untouched, so assert it still exists with the same identity (not deleted and recreated).
            assert policy.exists, "User-managed consumer NetworkPolicy did not survive the upgrade"
            assert policy.instance.metadata.uid == baseline.get("consumer_policy_uid"), (
                "User-managed consumer NetworkPolicy was recreated across the upgrade (uid changed)"
            )
            yield policy
            if teardown_resources:
                policy.clean_up(wait=True)
        else:
            # The feature turns on during the upgrade: create the policy now that operand policies exist.
            assert operand_network_policies_present(client=admin_client, namespace=namespace, timeout=300), (
                "The upgraded operator did not create its operand NetworkPolicies"
            )
            with NetworkPolicy(
                client=admin_client,
                kind_dict=consumer_network_policy_manifest(namespace=namespace),
                teardown=teardown_resources,
            ) as policy:
                yield policy
