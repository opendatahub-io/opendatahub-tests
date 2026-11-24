"""Spark Operator upgrade tests.

Pre-upgrade tests leave a task in progress. Post-upgrade tests verify continuity
and resubmission of the Pi workload prepared by the shared Spark smoke test.
"""

import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.network_policy import NetworkPolicy
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutSampler

from tests.spark.upgrade.utils import scheduled_running_execution, spark_running_execution
from tests.spark.utils import (
    SPARK_APPLICATION_EXECUTOR_POLICY_SUFFIX,
    SPARK_CONNECT_EXECUTOR_POLICY_SUFFIX,
    run_spark_connect_query,
    scheduled_last_run_application,
    verify_operand_network_policies,
    verify_spark_app_completed,
    verify_spark_connect_query,
    wait_for_spark_connect_ready,
)
from utilities.resources.scheduled_spark_application import ScheduledSparkApplication
from utilities.resources.spark_application import SparkApplication
from utilities.resources.spark_connect import SparkConnect


class TestPostUpgradeReuseExistingSparkApplication:
    """Verify a completed pre-upgrade workload can be submitted again after upgrade."""

    @pytest.mark.post_upgrade
    def test_reuse_existing_spark_application_post_upgrade(self, resubmitted_spark_application_fixture):
        """Given a completed pre-upgrade job, when resubmitted after upgrade, then it completes successfully."""
        assert resubmitted_spark_application_fixture is not None, "Fixture returned None; only runs post-upgrade"
        verify_spark_app_completed(spark_app=resubmitted_spark_application_fixture)


@pytest.mark.pre_upgrade
def test_spark_job_in_progress_pre_upgrade(
    admin_client: DynamicClient,
    spark_continuity_application: SparkApplication,
    spark_continuity_control: ConfigMap,
) -> None:
    """Given a Spark job, when its executor starts a task, then preserve that execution for upgrade."""
    for baseline in TimeoutSampler(
        wait_timeout=300,
        sleep=5,
        func=spark_running_execution,
        client=admin_client,
        spark_app=spark_continuity_application,
        exceptions_dict={AssertionError: []},
    ):
        ResourceEditor(patches={spark_continuity_control: {"data": {"baseline": json.dumps(baseline)}}}).update()
        break


@pytest.mark.post_upgrade
def test_spark_job_in_progress_completes_post_upgrade(
    admin_client: DynamicClient,
    spark_continuity_application: SparkApplication,
    spark_continuity_control: ConfigMap,
) -> None:
    """Given a task started before upgrade, when released afterward, then the same execution completes."""
    control_data = spark_continuity_control.instance.data
    assert control_data.get("release") == "false", "This execution was already released"
    assert control_data.get("baseline"), "The pre-upgrade running-execution baseline is missing"
    baseline = json.loads(control_data["baseline"])
    current = spark_running_execution(client=admin_client, spark_app=spark_continuity_application)
    assert current == baseline, f"Execution changed across upgrade: before={baseline}, after={current}"

    # The upgraded operator (RHOAI 3.6+) reconciles its operand NetworkPolicies onto the running job.
    verify_operand_network_policies(
        client=admin_client,
        namespace=spark_continuity_application.namespace,
        workload_name=spark_continuity_application.name,
        executor_policy_suffix=SPARK_APPLICATION_EXECUTOR_POLICY_SUFFIX,
        owner_uid=spark_continuity_application.instance.metadata.uid,
    )

    ResourceEditor(patches={spark_continuity_control: {"data": {"release": "true"}}}).update()
    verify_spark_app_completed(spark_app=spark_continuity_application)
    assert spark_continuity_application.instance.metadata.uid == baseline["uid"], "Application was replaced"
    driver = Pod(
        client=admin_client,
        name=spark_continuity_application.instance.status.driverInfo.podName,
        namespace=spark_continuity_application.namespace,
    )
    assert driver.instance.metadata.uid == baseline["pods"][driver.name]["uid"], "Driver was replaced"
    assert "UPGRADE_TASK_COMPLETED: 49" in driver.log(), "The original task did not return the expected result"


@pytest.mark.pre_upgrade
def test_scheduled_spark_job_in_progress_pre_upgrade(
    admin_client: DynamicClient,
    scheduled_continuity_application: ScheduledSparkApplication,
    scheduled_spark_continuity_control: ConfigMap,
) -> None:
    """Given a scheduled Spark job, when its run starts a task, then preserve that execution for upgrade."""
    for baseline in TimeoutSampler(
        wait_timeout=600,
        sleep=10,
        func=scheduled_running_execution,
        client=admin_client,
        scheduled_app=scheduled_continuity_application,
        exceptions_dict={AssertionError: []},
    ):
        ResourceEditor(
            patches={scheduled_spark_continuity_control: {"data": {"baseline": json.dumps(baseline)}}}
        ).update()
        break


@pytest.mark.post_upgrade
def test_scheduled_spark_job_in_progress_completes_post_upgrade(
    admin_client: DynamicClient,
    scheduled_continuity_application: ScheduledSparkApplication,
    scheduled_spark_continuity_control: ConfigMap,
) -> None:
    """Given a scheduled run started before upgrade, when released afterward, then the same run completes."""
    control_data = scheduled_spark_continuity_control.instance.data
    assert control_data.get("release") == "false", "This execution was already released"
    assert control_data.get("baseline"), "The pre-upgrade running-execution baseline is missing"
    baseline = json.loads(control_data["baseline"])

    child = scheduled_last_run_application(client=admin_client, scheduled_app=scheduled_continuity_application)
    assert child.name == baseline["run_name"], (
        f"Scheduled run changed across upgrade: before={baseline['run_name']}, after={child.name}"
    )
    current = {"run_name": child.name, **spark_running_execution(client=admin_client, spark_app=child)}
    assert current == baseline, f"Execution changed across upgrade: before={baseline}, after={current}"

    # The scheduled child reconciles through the SparkApplication controller, so it gets the same policies.
    verify_operand_network_policies(
        client=admin_client,
        namespace=child.namespace,
        workload_name=child.name,
        executor_policy_suffix=SPARK_APPLICATION_EXECUTOR_POLICY_SUFFIX,
        owner_uid=child.instance.metadata.uid,
    )

    ResourceEditor(patches={scheduled_spark_continuity_control: {"data": {"release": "true"}}}).update()
    verify_spark_app_completed(spark_app=child)
    assert child.instance.metadata.uid == baseline["uid"], "Scheduled run was replaced"
    driver = Pod(
        client=admin_client,
        name=child.instance.status.driverInfo.podName,
        namespace=child.namespace,
    )
    assert driver.instance.metadata.uid == baseline["pods"][driver.name]["uid"], "Driver was replaced"
    assert "UPGRADE_TASK_COMPLETED: 49" in driver.log(), "The original task did not return the expected result"


@pytest.mark.pre_upgrade
def test_spark_connect_serves_pre_upgrade(
    admin_client: DynamicClient,
    spark_connect_continuity_application: SparkConnect,
    spark_connect_client_pod: Pod,
    spark_connect_continuity_control: ConfigMap,
    spark_connect_consumer_network_policy: NetworkPolicy | None,
) -> None:
    """Given a running Spark Connect server, when a query succeeds, then record its server identity.

    Also records whether the operator's operand NetworkPolicies are already active, so the post-upgrade
    test knows whether the user-managed consumer policy should have survived or is newly created.
    """
    service_name = wait_for_spark_connect_ready(client=admin_client, spark_connect=spark_connect_continuity_application)
    output = run_spark_connect_query(
        client_pod=spark_connect_client_pod,
        service_name=service_name,
        namespace=spark_connect_continuity_application.namespace,
    )
    verify_spark_connect_query(output=output)

    server_pod = Pod(
        client=admin_client,
        name=spark_connect_continuity_application.instance.status.server.podName,
        namespace=spark_connect_continuity_application.namespace,
    )
    baseline = {
        "pod_name": server_pod.name,
        "uid": server_pod.instance.metadata.uid,
        "operand_policies_present": spark_connect_consumer_network_policy is not None,
    }
    if spark_connect_consumer_network_policy is not None:
        # Record the policy identity so the post-upgrade test can prove it survived untouched.
        baseline["consumer_policy_uid"] = spark_connect_consumer_network_policy.instance.metadata.uid
    ResourceEditor(patches={spark_connect_continuity_control: {"data": {"baseline": json.dumps(baseline)}}}).update()


@pytest.mark.post_upgrade
def test_spark_connect_survives_post_upgrade(
    admin_client: DynamicClient,
    spark_connect_continuity_application: SparkConnect,
    spark_connect_client_pod: Pod,
    spark_connect_continuity_control: ConfigMap,
    spark_connect_consumer_network_policy: NetworkPolicy | None,
) -> None:
    """Given a pre-upgrade Spark Connect server, when the operator is upgraded, then it still serves queries.

    Also verifies the upgraded operator reconciled its operand NetworkPolicies onto the surviving server,
    while the user-managed consumer policy (supplied by the fixture) keeps the client's gRPC access open.
    """
    assert spark_connect_continuity_control.instance.data.get("baseline"), "The pre-upgrade baseline is missing"
    baseline = json.loads(spark_connect_continuity_control.instance.data["baseline"])
    service_name = wait_for_spark_connect_ready(client=admin_client, spark_connect=spark_connect_continuity_application)

    verify_operand_network_policies(
        client=admin_client,
        namespace=spark_connect_continuity_application.namespace,
        workload_name=spark_connect_continuity_application.name,
        executor_policy_suffix=SPARK_CONNECT_EXECUTOR_POLICY_SUFFIX,
        owner_uid=spark_connect_continuity_application.instance.metadata.uid,
    )

    server_pod = Pod(
        client=admin_client,
        name=spark_connect_continuity_application.instance.status.server.podName,
        namespace=spark_connect_continuity_application.namespace,
    )
    assert server_pod.name == baseline["pod_name"], "Spark Connect server pod was replaced across upgrade"
    assert server_pod.instance.metadata.uid == baseline["uid"], (
        "Spark Connect server pod was recreated across upgrade "
        "(the operator should adopt the existing pod; a new uid indicates it was rolled or evicted)"
    )

    output = run_spark_connect_query(
        client_pod=spark_connect_client_pod,
        service_name=service_name,
        namespace=spark_connect_continuity_application.namespace,
    )
    verify_spark_connect_query(output=output)
