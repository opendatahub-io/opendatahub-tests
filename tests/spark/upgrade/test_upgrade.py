"""Spark Operator upgrade tests.

Pre-upgrade tests complete a Pi workload and leave a separate task in progress.
Post-upgrade tests verify resubmission, continuity of the running task, and execution
of newly created applications.
"""

import json

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from timeout_sampler import TimeoutSampler

from tests.spark.upgrade.utils import (
    spark_running_execution,
    verify_spark_app_completed,
    verify_spark_app_generation,
)
from utilities.resources.spark_application import SparkApplication


@pytest.mark.usefixtures("pre_upgrade_spark_dsc_patch")
class TestPreUpgradeSpark:
    """Validate Spark workload execution before an operator upgrade.

    Steps:
        0. Enable Spark Operator in DSC (Tech Preview component)
        1. Deploy a SparkApplication (spark-pi) resource
        2. Verify the application completes successfully
    """

    @pytest.mark.pre_upgrade
    def test_spark_pi_pre_upgrade_execution(self, spark_application_fixture):
        """Test SparkApplication (spark-pi) execution before upgrade"""
        verify_spark_app_completed(spark_app=spark_application_fixture)


class TestPostUpgradeReuseExistingSparkApplication:
    """Verify a completed pre-upgrade workload can be submitted again after upgrade."""

    @pytest.mark.post_upgrade
    def test_reuse_existing_spark_application_post_upgrade(self, resubmitted_spark_application_fixture):
        """Given a completed pre-upgrade job, when resubmitted after upgrade, then it completes successfully."""
        assert resubmitted_spark_application_fixture is not None, "Fixture returned None; only runs post-upgrade"
        verify_spark_app_completed(spark_app=resubmitted_spark_application_fixture)


@pytest.mark.pre_upgrade
@pytest.mark.tier1
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
@pytest.mark.tier1
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


@pytest.mark.usefixtures("post_upgrade_spark_dsc_patch")
class TestPostUpgradeNewSparkApplication:
    """Verify that the upgraded control plane can create new SparkApplications.

    Creates a fresh SparkApplication on the upgraded spark-operator to validate
    that the creation path works, not just preservation of pre-existing resources.
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="new_spark_app_created")
    def test_create_new_spark_application_post_upgrade(self, new_spark_application_fixture):
        """Verify a new SparkApplication can be created on the upgraded control plane"""
        assert new_spark_application_fixture is not None, "Fixture returned None; only runs post-upgrade"
        assert new_spark_application_fixture.exists, (
            f"Newly created SparkApplication {new_spark_application_fixture.name} does not exist"
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="new_spark_app_execution", depends=["new_spark_app_created"])
    def test_new_spark_application_post_upgrade_execution(self, new_spark_application_fixture):
        """Verify new SparkApplication completes successfully on upgraded operator"""
        verify_spark_app_completed(spark_app=new_spark_application_fixture)

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["new_spark_app_execution"])
    def test_new_spark_application_post_upgrade_generation(self, new_spark_application_fixture):
        """Verify newly created SparkApplication has generation=1 (fresh resource, after execution)"""
        verify_spark_app_generation(
            spark_app=new_spark_application_fixture,
            expected_generation=1,
        )
