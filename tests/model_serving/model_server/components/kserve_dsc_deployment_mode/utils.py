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


def wait_for_configmap_key_value(
    config_map: ConfigMap,
    data_key: str,
    expected_value: Any,
    nested_key_path: list[str],
) -> None:
    """
    Wait for a specific value to appear in a configmap under a given key.

    Args:
        config_map (ConfigMap): ConfigMap object
        data_key (str): Key in configmap.data to parse
        expected_value (Any): Expected value to wait for
        nested_key_path (list[str]): Optional list of keys to dig into the nested JSON
    """
    LOGGER.info(f"Waiting for {data_key} to be set to {expected_value} in {config_map.name}")
    nested_key_path = nested_key_path or []

    for sample in TimeoutSampler(
        wait_timeout=Timeout.TIMEOUT_5MIN,
        sleep=5,
        func=lambda: config_map.instance.data,
    ):
        if sample:
            try:
                parsed = json.loads(sample.get(data_key, "{}"))
                for key in nested_key_path:
                    parsed = parsed.get(key, {})
                if parsed == expected_value:
                    LOGGER.info(f"{data_key} set successfully to {expected_value}")
                    break
            except json.JSONDecodeError as e:
                LOGGER.warning(f"Could not decode {data_key}: {e}")


def patch_dsc_default_deployment_mode(
    dsc_resource: DataScienceCluster,
    config_map: ConfigMap,
    spec_key: str,
    config_key: str,
    expected_value: Any,
    nested_key_path: list[str],
) -> Generator[DataScienceCluster, Any, Any]:
    """
    Patch DataScienceCluster kserve component with a given value and wait for it to be applied in the configmap.

    Args:
        dsc_resource (DataScienceCluster): DSC object
        config_map (ConfigMap): ConfigMap to validate the result
        spec_key (str): Key in kserve spec to patch (e.g., "defaultDeploymentMode")
        config_key (str): Key in configmap data to check (e.g., "deploy" or "service")
        expected_value (Any): Value to patch and verify
        nested_key_path (list[str]): Nested path inside configmap[config_key] to verify

    Yields:
        DataScienceCluster: The patched resource
    """
    patch_value = expected_value
    if spec_key == "rawDeploymentServiceConfig":
        patch_value = (
            DscComponents.RawDeploymentServiceConfig.HEADLESS
            if expected_value
            else DscComponents.RawDeploymentServiceConfig.HEADED
        )

    with ResourceEditor(patches={dsc_resource: {"spec": {"components": {"kserve": {spec_key: patch_value}}}}}):
        wait_for_configmap_key_value(
            config_map=config_map,
            data_key=config_key,
            expected_value=expected_value,
            nested_key_path=nested_key_path,
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
        return service_data.get("serviceClusterIPNone")
    except (KeyError, json.JSONDecodeError) as e:
        raise ResourceNotFoundError(f"Failed to read service configuration: {e} from inferenceservice-config")
