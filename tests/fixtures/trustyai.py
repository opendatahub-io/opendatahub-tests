from collections.abc import Generator

import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from ocp_resources.resource import ResourceEditor
from pytest_testconfig import py_config

from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.infra import get_data_science_cluster


@pytest.fixture(scope="class")
def trustyai_operator_deployment(admin_client: DynamicClient) -> Deployment:
    return Deployment(
        client=admin_client,
        name=f"{TRUSTYAI_SERVICE_NAME}-operator-controller-manager",
        namespace=py_config["applications_namespace"],
        ensure_exists=True,
    )


@pytest.fixture(scope="class")
def patched_dsc_lmeval_allow_all(
    admin_client, trustyai_operator_deployment: Deployment
) -> Generator[DataScienceCluster]:
    """Enable LMEval PermitOnline and PermitCodeExecution flags in the Datascience cluster."""
    dsc = get_data_science_cluster(client=admin_client)
    with ResourceEditor(
        patches={
            dsc: {
                "spec": {
                    "components": {
                        "trustyai": {
                            "eval": {
                                "lmeval": {
                                    "permitCodeExecution": "allow",
                                    "permitOnline": "allow",
                                }
                            }
                        }
                    }
                }
            }
        }
    ):
        num_replicas: int = trustyai_operator_deployment.instance.spec.replicas
        trustyai_operator_deployment.scale_replicas(replica_count=0)
        trustyai_operator_deployment.scale_replicas(replica_count=num_replicas)
        trustyai_operator_deployment.wait_for_replicas()
        yield dsc
