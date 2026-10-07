"""Spark workload smoke coverage for install and upgrade lanes."""

import pytest

from tests.spark.utils import verify_spark_app_completed
from utilities.resources.spark_application import SparkApplication


@pytest.mark.smoke
@pytest.mark.install
@pytest.mark.pre_upgrade
@pytest.mark.post_upgrade
def test_spark_pi_execution(spark_pi_application: SparkApplication) -> None:
    """Given an installed operator, when a Pi application is created, then it completes successfully."""
    assert spark_pi_application.exists, f"SparkApplication {spark_pi_application.name} was not created"
    verify_spark_app_completed(spark_app=spark_pi_application)
