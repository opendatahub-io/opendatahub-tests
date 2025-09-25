import pytest

from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.dsc_initialization import DSCInitialization
from ocp_resources.node import Node
from ocp_utilities.infra import assert_nodes_in_healthy_condition, assert_nodes_schedulable
from utilities.infra import wait_for_dsci_status_ready, wait_for_dsc_status_ready


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
