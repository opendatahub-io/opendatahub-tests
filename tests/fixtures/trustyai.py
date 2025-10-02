import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment

from typing import Generator

from pytest_testconfig import py_config

from utilities.constants import TRUSTYAI_SERVICE_NAME
from utilities.infra import get_data_science_cluster
from utilities.trustyai_utils import patch_dsc_trustyai_lmeval_config


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
) -> Generator[DataScienceCluster, None, None]:
    """Enable LMEval PermitOnline and PermitCodeExecution flags in the Datascience cluster."""
    dsc = get_data_science_cluster(client=admin_client)
    yield from patch_dsc_trustyai_lmeval_config(
        dsc=dsc,
        trustyai_operator_deployment=trustyai_operator_deployment,
        permit_code_execution=True,
        permit_online=True,
    )
