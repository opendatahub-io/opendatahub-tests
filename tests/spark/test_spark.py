"""Spark workload smoke coverage for install and upgrade lanes."""

import pytest

from tests.spark.utils import verify_spark_app_completed


@pytest.mark.smoke
@pytest.mark.install
@pytest.mark.pre_upgrade
@pytest.mark.post_upgrade
@pytest.mark.parametrize(
    "application_fixture_name",
    [
        pytest.param("spark_pi_application", id="test_spark_application"),
        pytest.param("scheduled_spark_pi_application", id="test_scheduled_spark_application"),
    ],
)
def test_spark_pi_execution(request: pytest.FixtureRequest, application_fixture_name: str) -> None:
    """Given an installed operator, when a Pi application runs, then it completes successfully.

    Covers both a directly-created SparkApplication and a ScheduledSparkApplication's first run.
    """
    spark_application = request.getfixturevalue(argname=application_fixture_name)
    assert spark_application.exists, f"SparkApplication {spark_application.name} was not created"
    verify_spark_app_completed(spark_app=spark_application)
