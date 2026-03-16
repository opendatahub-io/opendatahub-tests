import pytest
from kubernetes.dynamic import DynamicClient
from ocp_resources.data_science_cluster import DataScienceCluster
from ocp_resources.deployment import Deployment
from pytest_testconfig import config as py_config
from simple_logger.logger import get_logger

from utilities.constants import DscComponents

LOGGER = get_logger(name=__name__)

KSERVE_CONTROLLER_DEPLOYMENTS: list[str] = [
    "kserve-controller-manager",
    "odh-model-controller",
]


def verify_kserve_health(admin_client: DynamicClient, dsc_resource: DataScienceCluster) -> None:
    """Verify that KServe components are healthy and ready to serve models.

    Checks management state, DSC ready condition, and controller deployment availability.
    Raises pytest.skip on any failure so downstream kserve tests are skipped.
    """
    applications_namespace = py_config["applications_namespace"]

    kserve_management_state = dsc_resource.instance.spec.components[DscComponents.KSERVE].managementState
    if kserve_management_state != DscComponents.ManagementState.MANAGED:
        pytest.skip(f"KServe managementState is {kserve_management_state}, expected Managed")

    kserve_ready = False
    for condition in dsc_resource.instance.status.conditions:
        if condition.type == DscComponents.COMPONENT_MAPPING[DscComponents.KSERVE]:
            if condition.status != "True":
                pytest.skip(f"KServe DSC condition is not ready: {condition.status}, reason: {condition.get('reason')}")
            kserve_ready = True
            break

    if not kserve_ready:
        pytest.skip("KserveReady condition not found in DSC status")

    for name in KSERVE_CONTROLLER_DEPLOYMENTS:
        deployment = Deployment(
            client=admin_client,
            name=name,
            namespace=applications_namespace,
        )
        if not deployment.exists:
            pytest.skip(f"KServe deployment {name} not found in {applications_namespace}")

        available = False
        for condition in deployment.instance.status.get("conditions", []):
            if condition.type == "Available":
                if condition.status != "True":
                    pytest.skip(f"KServe deployment {name} is not Available: {condition.get('reason')}")
                available = True
                break

        if not available:
            pytest.skip(f"KServe deployment {name} has no Available condition")

    LOGGER.info("KServe component health check passed")


@pytest.fixture(scope="session", autouse=True)
def kserve_health_check(
    request: pytest.FixtureRequest,
    admin_client: DynamicClient,
    dsc_resource: DataScienceCluster,
) -> None:
    """Session-scoped health gate for all kserve tests.

    Skips all tests under tests/model_serving/model_server/kserve/ when
    KServe components are not healthy.
    """
    if request.session.config.getoption("--skip-kserve-health-check"):
        LOGGER.warning("Skipping KServe health check, got --skip-kserve-health-check")
        return

    selected_markers = {mark.name for item in request.session.items for mark in item.iter_markers()}
    if "component_health" in selected_markers:
        LOGGER.info("Skipping KServe health gate because selected tests include component_health marker")
        return

    verify_kserve_health(admin_client=admin_client, dsc_resource=dsc_resource)
