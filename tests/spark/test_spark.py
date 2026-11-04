"""Spark workload smoke coverage for install and upgrade lanes."""

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.pod import Pod

from tests.spark.utils import run_spark_connect_query, verify_spark_app_completed, verify_spark_connect_query
from utilities.resources.spark_connect import SparkConnect


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


@pytest.mark.smoke
@pytest.mark.install
@pytest.mark.pre_upgrade
@pytest.mark.post_upgrade
def test_spark_connect_query(
    admin_client: DynamicClient,
    spark_connect_application: SparkConnect,
    spark_connect_client_pod: Pod,
) -> None:
    """Given a Spark Connect server, when a client runs a remote query, then it returns correct results."""
    service_name = spark_connect_application.instance.status.server.serviceName
    output = run_spark_connect_query(
        client_pod=spark_connect_client_pod,
        service_name=service_name,
        namespace=spark_connect_application.namespace,
    )
    verify_spark_connect_query(output=output)
