import pytest
from ocp_resources.data_science_cluster import DataScienceCluster
from simple_logger.logger import get_logger

from tests.cluster_health.utils import wait_for_dsc_status_ready

LOGGER = get_logger(name=__name__)


@pytest.mark.operator_health
def test_data_science_cluster_healthy(dsc_resource: DataScienceCluster) -> None:
    """
    Checks if a data science cluster is healthy
    """
    wait_for_dsc_status_ready(dsc_resource=dsc_resource)
