"""Spark Operator upgrade tests.

Pre-upgrade tests deploy SparkApplication resources and capture baseline state.
Post-upgrade tests validate that resources survived the upgrade and new resources can be created.
"""

import pytest

from tests.spark.upgrade.utils import (
    get_spark_app_baseline,
    verify_pods_not_restarted,
    verify_spark_app_completed,
    verify_spark_app_generation,
)


@pytest.mark.usefixtures("pre_upgrade_spark_dsc_patch", "spark_capture_upgrade_baseline")
class TestPreUpgradeSpark:
    """Validate Spark workload execution before an operator upgrade.

    Steps:
        0. Enable Spark Operator in DSC (Tech Preview component)
        1. Deploy a SparkApplication (spark-pi) resource
        2. Verify the application completes successfully
        3. Capture baseline values (generation, restart counts, state) to ConfigMap
    """

    @pytest.mark.pre_upgrade
    def test_spark_pi_pre_upgrade_execution(self, spark_application_fixture):
        """Test SparkApplication (spark-pi) execution before upgrade"""
        verify_spark_app_completed(spark_app=spark_application_fixture)


class TestPostUpgradeSpark:
    """Validate SparkApplication integrity and execution after an operator upgrade.

    Steps:
        1. Verify the SparkApplication still exists after the upgrade
        2. Verify the SparkApplication was not modified during the upgrade
        3. Verify pods have not restarted beyond pre-upgrade baseline
        4. Verify the SparkApplication is still in COMPLETED state
    """

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(name="spark_app_exists")
    def test_spark_application_post_upgrade_exists(self, spark_application_fixture):
        """Test that SparkApplication exists after upgrade"""
        assert spark_application_fixture.exists, f"SparkApplication {spark_application_fixture.name} does not exist"

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["spark_app_exists"])
    def test_spark_application_post_upgrade_not_modified(
        self,
        spark_application_fixture,
        spark_upgrade_baseline_fixture,
    ):
        """Test that SparkApplication is not modified during upgrade"""
        baseline = get_spark_app_baseline(
            baselines=spark_upgrade_baseline_fixture,
            spark_app_name=spark_application_fixture.name,
        )
        verify_spark_app_generation(
            spark_app=spark_application_fixture,
            expected_generation=baseline["generation"],
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["spark_app_exists"])
    def test_spark_application_post_upgrade_pods_not_restarted(
        self,
        admin_client,
        spark_application_fixture,
        spark_upgrade_baseline_fixture,
    ):
        """Verify SparkApplication pods have not restarted beyond pre-upgrade baseline"""
        baseline = get_spark_app_baseline(
            baselines=spark_upgrade_baseline_fixture,
            spark_app_name=spark_application_fixture.name,
        )
        verify_pods_not_restarted(
            client=admin_client,
            spark_app=spark_application_fixture,
            baseline_restart_counts=baseline["pod_restart_counts"],
        )

    @pytest.mark.post_upgrade
    @pytest.mark.dependency(depends=["spark_app_exists"])
    def test_spark_application_post_upgrade_still_completed(self, spark_application_fixture):
        """Test that SparkApplication is still in COMPLETED state after upgrade"""
        verify_spark_app_completed(spark_app=spark_application_fixture)


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
