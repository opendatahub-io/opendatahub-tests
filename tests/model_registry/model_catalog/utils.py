import json
from typing import Any
import time

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

import requests
from timeout_sampler import retry

from ocp_resources.pod import Pod
from tests.model_registry.model_catalog.constants import (
    DEFAULT_CATALOGS,
)
from tests.model_registry.utils import get_model_catalog_pod, get_rest_headers
from utilities.general import wait_for_pods_running

LOGGER = get_logger(name=__name__)


class ResourceNotFoundError(Exception):
    pass


def _execute_get_call(url: str, headers: dict[str, str], verify: bool | str = False) -> requests.Response:
    LOGGER.info(f"Executing get call: {url}")
    resp = requests.get(url=url, headers=headers, verify=verify, timeout=60)
    if resp.status_code not in [200, 201]:
        raise ResourceNotFoundError(f"Get call failed for resource: {url}, {resp.status_code}: {resp.text}")
    return resp


@retry(wait_timeout=60, sleep=5, exceptions_dict={ResourceNotFoundError: []})
def wait_for_model_catalog_api(url: str, headers: dict[str, str], verify: bool | str = False) -> requests.Response:
    return _execute_get_call(url=f"{url}sources", headers=headers, verify=verify)


def execute_get_command(url: str, headers: dict[str, str], verify: bool | str = False) -> dict[Any, Any]:
    resp = _execute_get_call(url=url, headers=headers, verify=verify)
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        LOGGER.error(f"Unable to parse {resp.text}")
        raise


def validate_model_catalog_enabled(pod: Pod) -> bool:
    for container in pod.instance.spec.containers:
        for env in container.env:
            if env.name == "ENABLE_MODEL_CATALOG":
                return True
    return False


def is_model_catalog_ready(client: DynamicClient, model_registry_namespace: str, consecutive_try: int = 6):
    model_catalog_pods = get_model_catalog_pod(client=client, model_registry_namespace=model_registry_namespace)
    # We can wait for the pods to reflect updated catalog, however, deleting them ensures the updated config is
    # applied immediately.
    for pod in model_catalog_pods:
        pod.delete()
    # After the deletion, we need to wait for the pod to be spinned up and get to ready state.
    assert wait_for_model_catalog_pod_created(client=client, model_registry_namespace=model_registry_namespace)
    wait_for_pods_running(
        admin_client=client, namespace_name=model_registry_namespace, number_of_consecutive_checks=consecutive_try
    )


class PodNotFound(Exception):
    """Pod not found"""

    pass


@retry(wait_timeout=30, sleep=5, exceptions_dict={PodNotFound: []})
def wait_for_model_catalog_pod_created(client: DynamicClient, model_registry_namespace: str) -> bool:
    pods = get_model_catalog_pod(client=client, model_registry_namespace=model_registry_namespace)
    if pods:
        return True
    raise PodNotFound("Model catalog pod not found")


def validate_model_catalog_resource(
    kind: Any, admin_client: DynamicClient, namespace: str, expected_resource_count: int
) -> None:
    resource = list(kind.get(namespace=namespace, label_selector="component=model-catalog", dyn_client=admin_client))
    assert resource
    LOGGER.info(f"Validating resource: {kind}: Found {len(resource)})")
    assert len(resource) == expected_resource_count, (
        f"Unexpected number of {kind} resources found: {[res.name for res in resource]}"
    )


def validate_default_catalog(catalogs: list[dict[Any, Any]]) -> None:
    errors = ""
    for catalog in catalogs:
        expected_catalog = DEFAULT_CATALOGS.get(catalog["id"])
        assert expected_catalog, f"Unexpected catalog: {catalog}"
        for field in ["type", "name", "properties"]:
            if catalog[field] != expected_catalog[field]:
                errors += f"For {catalog['id']} expected {field}={expected_catalog[field]}, but got {catalog[field]}"

    assert not errors, errors


def get_catalog_str(ids: list[str]) -> str:
    catalog_str: str = ""
    for index, id in enumerate(ids):
        catalog_str += f"""
- name: Sample Catalog {index}
  id: {id}
  type: yaml
  enabled: true
  properties:
    yamlCatalogPath: {id.replace("_", "-")}.yaml
"""
    return f"""catalogs:
{catalog_str}
"""


def get_sample_yaml_str(models: list[str]) -> str:
    model_str: str = ""
    for model in models:
        model_str += f"""
{get_model_str(model=model)}
"""
    return f"""source: Hugging Face
models:
{model_str}
"""


def get_model_str(model: str) -> str:
    current_time = int(time.time() * 1000)
    return f"""
- name: {model}
  description: test description.
  readme: |-
    # test read me information {model}
  provider: Mistral AI
  logo: temp placeholder logo
  license: apache-2.0
  licenseLink: https://www.apache.org/licenses/LICENSE-2.0.txt
  libraryName: transformers
  artifacts:
    - uri: https://huggingface.co/{model}/resolve/main/consolidated.safetensors
  createTimeSinceEpoch: \"{str(current_time - 10000)}\"
  lastUpdateTimeSinceEpoch: \"{str(current_time)}\"
"""


def get_validate_default_model_catalog_source(token: str, model_catalog_url: str) -> None:
    LOGGER.info("Attempting client connection with token")
    result = execute_get_command(
        url=model_catalog_url,
        headers=get_rest_headers(token=token),
    )["items"]
    assert result
    assert len(result) == 2, f"Expected no custom models to be present. Actual: {result}"
    ids_actual = [entry["id"] for entry in result]
    assert sorted(ids_actual) == sorted(DEFAULT_CATALOGS.keys()), (
        f"Actual default catalog entries: {ids_actual},Expected: {DEFAULT_CATALOGS.keys()}"
    )


def extract_schema_fields(openapi_schema: dict[Any, Any], schema_name: str) -> tuple[set[str], set[str]]:
    """
    Extract all and required fields from an OpenAPI schema for validation.

    Args:
        openapi_schema: The parsed OpenAPI schema dictionary
        schema_name: Name of the schema to extract (e.g., "CatalogModel", "CatalogModelArtifact")

    Returns:
        Tuple of (all_fields, required_fields) excluding server-generated fields and timestamps.
    """

    def _extract_properties_and_required(schema: dict[Any, Any]) -> tuple[set[str], set[str]]:
        """Recursively extract properties and required fields from a schema."""
        props = set(schema.get("properties", {}).keys())
        required = set(schema.get("required", []))

        # Properties from allOf (inheritance/composition)
        if "allOf" in schema:
            for item in schema["allOf"]:
                sub_schema = item
                if "$ref" in item:
                    # Follow reference and recursively extract
                    ref_schema_name = item["$ref"].split("/")[-1]
                    sub_schema = openapi_schema["components"]["schemas"][ref_schema_name]
                sub_props, sub_required = _extract_properties_and_required(schema=sub_schema)
                props.update(sub_props)
                required.update(sub_required)

        return props, required

    target_schema = openapi_schema["components"]["schemas"][schema_name]
    all_properties, required_fields = _extract_properties_and_required(schema=target_schema)

    # Exclude fields that shouldn't be compared
    excluded_fields = {
        "id",  # Server-generated
        "externalId",  # Server-generated
        "createTimeSinceEpoch",  # Timestamps may differ
        "lastUpdateTimeSinceEpoch",  # Timestamps may differ
        "artifacts",  # CatalogModel only
        "source_id",  # CatalogModel only
    }

    return all_properties - excluded_fields, required_fields - excluded_fields
