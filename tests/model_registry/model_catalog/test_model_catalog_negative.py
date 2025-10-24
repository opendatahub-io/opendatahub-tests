import pytest
import re
from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger
from typing import Self

from ocp_resources.config_map import ConfigMap
from ocp_resources.pod import Pod
from ocp_resources.resource import ResourceEditor
from tests.model_registry.constants import DEFAULT_MODEL_CATALOG_CM
from tests.model_registry.model_catalog.constants import DEFAULT_CATALOGS, CATALOG_CONTAINER
from tests.model_registry.model_catalog.utils import validate_model_catalog_configmap_data
from tests.model_registry.utils import get_model_catalog_pod, is_model_catalog_ready
from timeout_sampler import TimeoutExpiredError

from utilities.general import wait_for_container_status

LOGGER = get_logger(name=__name__)

pytestmark = [
    pytest.mark.usefixtures(
        "updated_dsc_component_state_scope_session",
        "model_registry_namespace",
    )
]


@pytest.fixture(scope="class")
def updated_catalog_cm_negative(
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> ConfigMap:
    """Fixture for testing negative scenarios with custom catalog configmap"""
    patches = request.param

    with ResourceEditor(patches={catalog_config_map: patches}):
        try:
            is_model_catalog_ready(
                client=admin_client, model_registry_namespace=model_registry_namespace, consecutive_try=1
            )
        except TimeoutExpiredError:
            LOGGER.info("Model Catalog pod is expected to be unhealthy")
        yield catalog_config_map

    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)


@pytest.fixture(scope="class")
def model_catalog_pod_error_state(
    admin_client: DynamicClient,
    model_registry_namespace: str,
) -> Pod:
    """Fixture that returns a model catalog pod in error state"""
    pods = get_model_catalog_pod(
        client=admin_client,
        model_registry_namespace=model_registry_namespace,
        label_selector="app.kubernetes.io/name=model-catalog",
    )
    assert pods, "Model catalog pod not found in the namespace"
    model_catalog_pod = pods[0]
    assert wait_for_container_status(
        pod=model_catalog_pod,
        container_name=CATALOG_CONTAINER,
        expected_status=Pod.Status.CRASH_LOOPBACK_OFF,
        timeout=60,
        sleep=2,
    )
    return model_catalog_pod


def _get_default_catalog_str() -> str:
    """
    Create a catalog string using the first entry from DEFAULT_CATALOGS.
    Similar to get_catalog_str() but uses actual DEFAULT_CATALOGS values.
    """
    # Get the first catalog ID and data from DEFAULT_CATALOGS
    first_catalog_id = next(iter(DEFAULT_CATALOGS))
    first_catalog_data = DEFAULT_CATALOGS[first_catalog_id]

    catalog_str = f"""
- name: {first_catalog_data["name"]}
  id: {first_catalog_id}
  type: {first_catalog_data["type"]}
  properties:
    yamlCatalogPath: {first_catalog_data["properties"]["yamlCatalogPath"]}
"""

    return f"""catalogs:
{catalog_str}
"""


@pytest.mark.skip_must_gather
class TestDefaultCatalogNegative:
    """Negative tests for default catalog configuration"""

    @pytest.mark.parametrize(
        "model_catalog_config_map, modified_sources_yaml",
        [
            pytest.param(
                {"configmap_name": DEFAULT_MODEL_CATALOG_CM},
                """
catalogs:
  - name: Modified Catalog
    id: modified_catalog
    type: yaml
    properties:
      yamlCatalogPath: /shared-data/modified-catalog.yaml
""",
                id="test_modify_catalog_structure",
            ),
        ],
        indirect=["model_catalog_config_map"],
    )
    def test_modify_default_catalog_configmap_reconciles(
        self: Self, model_catalog_config_map: ConfigMap, modified_sources_yaml: str
    ):
        """
        Test that attempting to modify the default catalog configmap raises an exception.
        This validates that the default catalog configmap is protected from modifications.
        """
        # Attempt to modify the configmap - this should raise an exception
        patches = {"data": {"sources.yaml": modified_sources_yaml}}

        with ResourceEditor(patches={model_catalog_config_map: patches}):
            # This block should raise an exception due to configmap protection
            LOGGER.info("Attempting to modify protected configmap")

        # Verify the configmap was not actually modified
        validate_model_catalog_configmap_data(
            configmap=model_catalog_config_map, num_catalogs=len(DEFAULT_CATALOGS.keys())
        )


@pytest.mark.skip_must_gather
class TestCustomCatalogNegative:
    """Negative tests for custom catalog configuration"""

    @pytest.mark.parametrize(
        "updated_catalog_cm_negative",
        [
            pytest.param(
                {"data": {"sources.yaml": _get_default_catalog_str()}},
                id="test_default_catalog_in_custom_configmap",
            ),
        ],
        indirect=True,
    )
    def test_default_catalog_rejected_in_custom_configmap(
        self: Self, updated_catalog_cm_negative: ConfigMap, model_catalog_pod_error_state: Pod
    ):
        """
        Test that attempting to add a default catalog to custom configmap is rejected.
        This validates that default catalogs cannot be added to custom catalog configmap.
        """
        # Check model catalog pod logs for the expected error
        pod_log = model_catalog_pod_error_state.log(container=CATALOG_CONTAINER)

        # Look for error pattern using regex
        error_pattern = r"Error: error loading catalog sources: (.*): source (.*) exists from multiple origins"
        match = re.search(error_pattern, pod_log)

        if match:
            sources_file = match.group(1)
            source_name = match.group(2)
            LOGGER.warning(
                f"Found expected error in pod {model_catalog_pod_error_state.name} log: "
                f"sources file '{sources_file}', source '{source_name}' exists from multiple origins"
            )
        else:
            LOGGER.info(f"Pod log is: {pod_log}")
            pytest.fail(f"Expected error pattern not found in pod {model_catalog_pod_error_state.name} log")
