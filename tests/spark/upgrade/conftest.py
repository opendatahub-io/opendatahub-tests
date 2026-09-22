"""Fixtures for Spark execution continuity and resubmission across upgrades."""

from collections.abc import Generator
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
from tests.spark.utils import create_spark_pi_application_spec
from utilities.resources.spark_application import SparkApplication


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
    if not pytestconfig.option.post_upgrade:
        yield None
        return

    yield resubmit_spark_application(
        client=admin_client,
        spark_app=spark_application_fixture,
        teardown=teardown_resources,
    )


@pytest.fixture()
def spark_continuity_control(
    pytestconfig: pytest.Config,
    admin_client: DynamicClient,
    spark_namespace_fixture: Namespace,
    teardown_resources: bool,
) -> Generator[ConfigMap, Any, Any]:
    """Provide the persisted workload script, release gate, and execution baseline."""
    control = ConfigMap(client=admin_client, name="spark-upgrade-continuity", namespace=spark_namespace_fixture.name)
    if pytestconfig.option.post_upgrade:
        assert control.exists, "Run the in-progress pre-upgrade test and preserve its resources first"
        yield control
        return
    with ConfigMap(
        client=admin_client,
        name=control.name,
        namespace=control.namespace,
        data={"workload.py": Path(__file__).with_name("continuity_workload.py").read_text(), "release": "false"},
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
    network_policy_fixture: list[NetworkPolicy],
    teardown_resources: bool,
) -> Generator[SparkApplication, Any, Any]:
    """Create before upgrade, or reference after upgrade, the gated Spark execution."""
    application = SparkApplication(
        client=admin_client, name="upgrade-spark-in-progress", namespace=spark_continuity_control.namespace
    )
    if pytestconfig.option.post_upgrade:
        assert application.exists, "The in-progress SparkApplication did not survive"
        yield application
        return
    manifest = create_spark_pi_application_spec(
        name=application.name,
        namespace=application.namespace,
        service_account=spark_workload_service_account.name,
    )
    spec = manifest["spec"]
    spec.pop("mainClass")
    spec.update({
        "type": "Python",
        "pythonVersion": "3",
        "mainApplicationFile": "local:///opt/spark/upgrade/workload.py",
        "sparkConf": {"spark.task.maxFailures": "1", "spark.speculation": "false"},
    })
    spec["volumes"].append({"name": "upgrade-control", "configMap": {"name": spark_continuity_control.name}})
    volumes = spec.pop("volumes")
    for role in ("driver", "executor"):
        mounts = spec[role].pop("volumeMounts")
        mounts.append({"name": "upgrade-control", "mountPath": "/opt/spark/upgrade", "readOnly": True})
        # Native pod templates carry mounts even when the mutating webhook does not inject them.
        spec[role]["template"] = {
            "spec": {
                "volumes": volumes,
                "containers": [{"name": f"spark-kubernetes-{role}", "volumeMounts": mounts}],
            }
        }
    with SparkApplication(client=admin_client, kind_dict=manifest, teardown=teardown_resources) as application:
        yield application
