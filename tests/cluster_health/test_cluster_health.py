import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.node import Node
from ocp_utilities.infra import assert_nodes_in_healthy_condition, assert_nodes_schedulable
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.general import wait_for_pods_running
from utilities.infra import wait_for_dsc_status_ready, wait_for_dsci_status_ready

LOGGER = get_logger(name=__name__)


@pytest.mark.cluster_health
def test_cluster_node_healthy(nodes: list[Node]):
    assert_nodes_in_healthy_condition(nodes=nodes, healthy_node_condition_type={"KubeletReady": "True"})
    assert_nodes_schedulable(nodes=nodes)


@pytest.mark.cluster_health
def test_data_science_cluster_initialization_healthy(dsci_resource: DSCInitialization):
    wait_for_dsci_status_ready(dsci_resource=dsci_resource)


@pytest.mark.cluster_health
def test_data_science_cluster_healthy(dsc_resource: DataScienceCluster):
    wait_for_dsc_status_ready(dsc_resource=dsc_resource)


@pytest.mark.parametrize(
    "namespace_name",
    [
        pytest.param(
            py_config["operator_namespace"],
            id="test_operator_namespace_pod_healthy",
        ),
        pytest.param(
            py_config["applications_namespace"],
            id="test_application_namespace_pod_healthy",
        ),
    ],
)
@pytest.mark.cluster_health
def test_pods_cluster_healthy(admin_client: DynamicClient, namespace_name: str):
    LOGGER.info(f"Testing Pods in namespace {namespace_name} for cluster health")
    wait_for_pods_running(admin_client=admin_client, namespace_name=namespace_name)
