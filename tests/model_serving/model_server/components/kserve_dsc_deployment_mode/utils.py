import json
from typing import Any, Generator

from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.resource import ResourceEditor
from simple_logger.logger import get_logger
from timeout_sampler import TimeoutSampler

from kubernetes.dynamic import DynamicClient
from utilities.constants import Timeout, DscComponents
from pytest_testconfig import config as py_config
from kubernetes.dynamic.exceptions import ResourceNotFoundError

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


def wait_for_service_config_in_cm(config_map: ConfigMap, service_config: str) -> None:
    """
    Wait for service config to be set in inferenceservice-config configmap.

    Args:
        config_map (ConfigMap): ConfigMap object
        service_config (str): Service configuration type ("Headless" or "Headed")

    Raises:
        TimeoutExpiredError: If service config value is not set in configmap
    """
    LOGGER.info(f"Waiting for {service_config} service config to be set in {config_map.name} configmap")
    expected_value = service_config == DscComponents.RawDeploymentServiceConfig.HEADLESS
    for sample in TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_5MIN,
        sleep=5,
        func=lambda: config_map.instance.data,
    ):
        if sample:
            service_data = json.loads(sample.get("service", "{}"))
            cm_service_config = service_data.get("serviceClusterIPNone")
            if cm_service_config == expected_value:
                LOGGER.info(f"Service config successfully set to {service_config}")
                break


def patch_raw_default_deployment_config(
    dsc_resource: DataScienceCluster,
    inferenceservice_config_cm: ConfigMap,
    request_default_deployment_config: str,
) -> Generator[DataScienceCluster, Any, Any]:
    """
    Patch DataScienceCluster object with service configuration and wait for it to be set in configmap.

    Args:
        dsc_resource (DataScienceCluster): DataScienceCluster object
        inferenceservice_config_cm (ConfigMap): ConfigMap object
        request_default_deployment_config (str): Service configuration type ("Headless" or "Headed")

    Yields:
        DataScienceCluster: DataScienceCluster object with updated service configuration
    """
    if request_default_deployment_config not in ["Headless", "Headed"]:
        raise ValueError("Service config must be either 'Headless' or 'Headed'")

    with ResourceEditor(
        patches={
            dsc_resource: {
                "spec": {"components": {"kserve": {"rawDeploymentServiceConfig": request_default_deployment_config}}}
            }
        }
    ):
        wait_for_service_config_in_cm(
            config_map=inferenceservice_config_cm,
            service_config=request_default_deployment_config,
        )
        yield dsc_resource


def get_service_cluster_ip(admin_client: DynamicClient) -> bool:
    """
    Get the service cluster IP configuration from the inferenceservice-config configmap.

    Args:
        admin_client (DynamicClient): Admin client for accessing the cluster

    Returns:
        bool: True if service is Headless (serviceClusterIPNone=True), False if Headed

    Raises:
        ResourceNotFoundError: If the configmap or service configuration cannot be read
    """
    try:
        config_map = ConfigMap(
            client=admin_client, namespace=py_config["applications_namespace"], name="inferenceservice-config"
        )
        service_data = json.loads(config_map.instance.get("data", {}).get("service", "{}"))
        service_cluster_ip = service_data.get("serviceClusterIPNone")
        return service_cluster_ip
    except (KeyError, json.JSONDecodeError) as e:
        raise ResourceNotFoundError(f"Failed to read service configuration: {e} from inferenceservice-config")
