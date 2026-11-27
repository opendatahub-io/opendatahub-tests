"""Read-only health check for the platform-to-Workbenches module handoff."""

import pytest
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.config_map import ConfigMap
from ocp_resources.data_science_cluster import DataScienceCluster
from pytest_testconfig import config as py_config
from semver import Version
from timeout_sampler import TimeoutExpiredError, TimeoutSampler

from tests.workbenches.utils import (
    SKIP_EXPECTED_LEGACY_WORKBENCHES,
    SKIP_EXPECTED_WORKBENCHES_NOT_MANAGED,
    expected_skip,
)
from utilities.constants import DscComponents
from utilities.resources.workbenches import Workbenches


@pytest.mark.smoke
@pytest.mark.component_health
def test_workbenches_module_health(admin_client: DynamicClient, dsc_resource: DataScienceCluster) -> None:
    """Given a modular platform with Workbenches managed in the DSC.

    When the platform provisions the singleton and publishes its configuration.
    Then the module is ready for the current generation and reports the configured platform release.
    """
    dsc = dsc_resource.instance.to_dict()
    management_state = dsc.get("spec", {}).get("components", {}).get("workbenches", {}).get("managementState")
    if management_state != DscComponents.ManagementState.MANAGED:
        pytest.skip(
            expected_skip(
                prefix=SKIP_EXPECTED_WORKBENCHES_NOT_MANAGED,
                detail=f"Workbenches is not managed in {dsc_resource.name}: managementState={management_state!r}",
            )
        )

    # The legacy controller also used the Workbenches API, so CRD presence cannot distinguish it.
    # Workbenches moved to a module in the 3.5 platform line (opendatahub-operator #3817).
    release_version = dsc.get("status", {}).get("release", {}).get("version", "")
    if release_version and Version.is_valid(version=release_version):
        version = Version.parse(version=release_version)
        if (version.major, version.minor) < (3, 5):
            pytest.skip(
                expected_skip(
                    prefix=SKIP_EXPECTED_LEGACY_WORKBENCHES,
                    detail=f"Platform release {release_version} predates the modular Workbenches operator",
                )
            )

    workbenches = Workbenches(client=admin_client, name="default-workbenches")
    platform_config = ConfigMap(
        client=admin_client, name="odh-workbenches-config", namespace=py_config["applications_namespace"]
    )
    config_source = f"ConfigMap {platform_config.namespace}/{platform_config.name} data.platformVersion"
    last_observation = f"API {workbenches.api_version} not discovered"
    try:
        for instance in TimeoutSampler(
            wait_timeout=300,
            sleep=5,
            func=lambda: workbenches.exists,
            exceptions_dict={ResourceNotFoundError: []},
        ):
            if instance is None:
                last_observation = "singleton does not exist"
                continue

            config = platform_config.exists
            config_data = config.to_dict().get("data", {}) if config is not None else {}
            expected_version = (config_data or {}).get("platformVersion", "").strip()
            if not expected_version:
                last_observation = f"{config_source} is missing or empty"
                continue

            snapshot = instance.to_dict()
            generation = snapshot["metadata"]["generation"]
            status = snapshot.get("status") or {}
            observed_generation = status.get("observedGeneration")
            conditions = {
                condition["type"]: condition
                for condition in status.get("conditions", [])
                if condition["type"] in {"Ready", "ProvisioningSucceeded"}
            }
            platform_version = next(
                (release.get("version") for release in status.get("releases", []) if release["name"] == "platform"),
                None,
            )
            if (
                all(conditions.get(name, {}).get("status") == "True" for name in ("Ready", "ProvisioningSucceeded"))
                and observed_generation == generation
                and platform_version == expected_version
            ):
                return

            last_observation = (
                f"Ready={conditions.get('Ready')!r}; "
                f"ProvisioningSucceeded={conditions.get('ProvisioningSucceeded')!r}; "
                f"observedGeneration={observed_generation!r}, metadata.generation={generation!r}; "
                f"status.releases[name=platform].version={platform_version!r}, "
                f"expected {expected_version!r} from {config_source}"
            )
    except TimeoutExpiredError:
        pytest.fail(f"Workbenches/{workbenches.name} did not converge: {last_observation}")
