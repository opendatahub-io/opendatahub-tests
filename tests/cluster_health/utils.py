from simple_logger.logger import get_logger

from ocp_resources.data_science_cluster import DataScienceCluster

from timeout_sampler import retry

LOGGER = get_logger(name=__name__)


class ResourceNotReadyError(Exception):
    pass


@retry(
    wait_timeout=120,
    sleep=5,
    exceptions_dict={ResourceNotReadyError: []},
)
def wait_for_dsc_status_ready(dsc_resource: DataScienceCluster) -> bool:
    LOGGER.info(f"Wait for DSC {dsc_resource.name} are {dsc_resource.Status.READY}.")
    if dsc_resource.status == dsc_resource.Status.READY:
        return True
    raise ResourceNotReadyError(
        f"DSC {dsc_resource.name} is not ready.\nCurrent status: {dsc_resource.instance.status}"
    )
