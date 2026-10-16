"""Spark Operator upgrade tests.

Pre-upgrade tests leave a task in progress. Post-upgrade tests verify continuity
and resubmission of the Pi workload prepared by the shared Spark smoke test.
"""

import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutSampler

from tests.spark.upgrade.utils import scheduled_running_execution, spark_running_execution
from tests.spark.utils import scheduled_last_run_application, verify_spark_app_completed
from utilities.resources.scheduled_spark_application import ScheduledSparkApplication
from utilities.resources.spark_application import SparkApplication


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
