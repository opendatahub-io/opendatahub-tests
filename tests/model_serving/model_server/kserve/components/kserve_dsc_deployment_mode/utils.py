import json
from collections.abc import Generator
from typing import Any

from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from utilities.constants import Timeout

LOGGER = get_logger(name=__name__)


def wait_for_default_deployment_mode_in_cm(config_map: ConfigMap, deployment_mode: str) -> None:
    """
    Wait for default deployment mode to be set in inferenceservice-config configmap.

    Args:
        config_map (ConfigMap): ConfigMap object
        deployment_mode (str): Deployment mode

    Raises:
        TimeoutExpiredError: If default deployment mode value is not set in configmap

    """
    LOGGER.info(
        f"Wait for {{request_default_deployment_mode}} deployment mode to be set in {config_map.name} configmap"
    )
    for sample in TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_5MIN,
        sleep=5,
        func=lambda: config_map.instance.data,
    ):
        if sample:
            cm_default_deployment_mode = json.loads(sample.deploy)["defaultDeploymentMode"]
            if cm_default_deployment_mode == deployment_mode:
                break


def patch_dsc_default_deployment_mode(
    dsc_resource: DataScienceCluster,
    inferenceservice_config_cm: ConfigMap,
    request_default_deployment_mode: str,
) -> Generator[DataScienceCluster, Any, Any]:
    """
    Patch DataScienceCluster object with default deployment mode and wait for it to be set in configmap.

    Args:
        dsc_resource (DataScienceCluster): DataScienceCluster object
        inferenceservice_config_cm (ConfigMap): ConfigMap object
        request_default_deployment_mode (str): Deployment mode

    Yields:
        DataScienceCluster: DataScienceCluster object

    """
    with ResourceEditor(
        patches={
            dsc_resource: {
                "spec": {"components": {"kserve": {"defaultDeploymentMode": request_default_deployment_mode}}}
            }
        }
    ):
        wait_for_default_deployment_mode_in_cm(
            config_map=inferenceservice_config_cm,
            deployment_mode=request_default_deployment_mode,
        )
        yield dsc_resource
