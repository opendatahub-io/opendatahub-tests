import pytest
from ocp_resources.node import Node
from ocp_utilities.infra import assert_nodes_in_healthy_condition, assert_nodes_schedulable
from simple_logger.logger import get_logger

LOGGER = get_logger(name=__name__)


@pytest.mark.cluster_health
def test_cluster_node_healthy(nodes: list[Node]):
    assert_nodes_in_healthy_condition(nodes=nodes, healthy_node_condition_type={"KubeletReady": "True"})
    assert_nodes_schedulable(nodes=nodes)
