import json
import time
from typing import Any

import requests
from kubernetes.dynamic import DynamicClient
from kubernetes.dynamic.exceptions import ResourceNotFoundError
from ocp_resources.pod import Pod
from simple_logger.logger import get_logger
from timeout_sampler import retry

from tests.model_registry.model_catalog.constants import HF_MODELS
from tests.model_registry.utils import get_model_catalog_pod
from utilities.constants import PodNotFound
from utilities.general import wait_for_pods_running

LOGGER = get_logger(name=__name__)


def get_postgres_pod_in_namespace(admin_client: DynamicClient, namespace: str = "rhoai-model-registries") -> Pod:
    """Get the PostgreSQL pod for model catalog database."""
    postgres_pods = list(
        Pod.get(
            client=admin_client, namespace=namespace, label_selector="app.kubernetes.io/name=model-catalog-postgres"
        )
    )
    assert postgres_pods, f"No PostgreSQL pod found in namespace {namespace}"
    return postgres_pods[0]


def execute_database_query(admin_client: DynamicClient, query: str, namespace: str = "rhoai-model-registries") -> str:
    """
    Execute a SQL query against the model catalog database.

    Args:
        query: SQL query to execute
        namespace: OpenShift namespace containing the PostgreSQL pod

    Returns:
        Raw database query result as string
    """
    postgres_pod = get_postgres_pod_in_namespace(admin_client=admin_client, namespace=namespace)

    return postgres_pod.execute(
        command=["psql", "-U", "catalog_user", "-d", "model_catalog", "-c", query],
        container="postgresql",
    )


def parse_psql_output(psql_output: str) -> dict[str, Any]:
    """
    Parse psql CLI output into appropriate Python data structures.

    Handles two main formats:
    1. Single column: Returns {"values": [list_of_values]}
    2. Two columns with array_agg: Returns {"properties": {name: [values]}}

    Args:
        psql_output: Raw psql command output

    Returns:
        Dictionary with parsed data in appropriate format
    """
    lines = psql_output.strip().split("\n")

    # Find the header line to determine format
    header_line = None
    separator_line = None

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        # Look for separator line (all dashes and pipes)
        if line.replace("-", "").replace("+", "").replace("|", "").strip() == "":
            separator_line = i
            if i > 0:
                header_line = i - 1
            break

    if header_line is None:
        return {"values": []}

    header = lines[header_line].strip()

    # Determine format based on header
    if "|" in header and "array_agg" in header:
        # Two-column format with array aggregation
        return {"properties": _parse_array_agg_format(lines, separator_line + 1)}
    else:
        # Single column format
        return {"values": _parse_single_column_format(lines, separator_line + 1)}


def _parse_array_agg_format(lines: list[str], data_start: int) -> dict[str, list[str]]:
    """Parse two-column format with PostgreSQL array aggregation."""
    result = {}

    for line in lines[data_start:]:
        line = line.strip()
        if not line or "|" not in line:
            continue

        # Skip summary lines like "(X rows)"
        if line.startswith("(") and "row" in line:
            break

        # Parse data row: "property_name | {val1,val2,val3}"
        parts = line.split("|", 1)
        if len(parts) != 2:
            continue

        property_name = parts[0].strip()
        array_str = parts[1].strip()

        # Parse PostgreSQL array format: {val1,val2,val3}
        if array_str.startswith("{") and array_str.endswith("}"):
            # Remove braces and split by comma
            values_str = array_str[1:-1]
            if values_str:
                # Handle escaped commas and quotes properly
                values = [v.strip().strip('"') for v in values_str.split(",")]
                result[property_name] = values
            else:
                result[property_name] = []

    return result


def _parse_single_column_format(lines: list[str], data_start: int) -> list[str]:
    """Parse single column format."""
    result = []

    for line in lines[data_start:]:
        line = line.strip()
        if not line:
            continue

        # Skip summary lines like "(X rows)"
        if line.startswith("(") and "row" in line:
            break

        result.append(line)

    return result


def get_models_from_catalog_api(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    page_size: int = 100,
    source_label: str | None = None,
    q: str | None = None,
    order_by: str | None = None,
    sort_order: str | None = None,
    additional_params: str = "",
) -> dict[str, Any]:
    """
    Helper method to get models from catalog API with optional filtering and sorting

    Args:
        model_catalog_rest_url: REST URL for model catalog
        model_registry_rest_headers: Headers for model registry REST API
        page_size: Number of results per page
        source_label: Source label(s) to filter by (must be comma-separated for multiple filters)
        q: Free-form keyword search to filter models
        order_by: Field to order results by (ID, NAME, CREATE_TIME, LAST_UPDATE_TIME)
        sort_order: Sort order (ASC or DESC)
        additional_params: Additional query parameters (e.g., "&filterQuery=name='model_name'")

    Returns:
        Dictionary containing the API response
    """
    base_url = f"{model_catalog_rest_url[0]}models"

    # Build params dictionary for proper URL encoding
    params = {"pageSize": page_size}

    if source_label:
        params["sourceLabel"] = source_label

    if q:
        params["q"] = q

    if order_by:
        params["orderBy"] = order_by

    if sort_order:
        params["sortOrder"] = sort_order

    # Parse additional_params string into params dict for proper URL encoding
    if additional_params:
        # Remove leading & if present
        clean_params = additional_params.lstrip("&")
        # Split by & and then by = to get key-value pairs
        for param in clean_params.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)  # Split only on first = to handle values with =
                params[key] = value

    return execute_get_command(url=base_url, headers=model_registry_rest_headers, params=params)


def get_hf_catalog_str(ids: list[str], excluded_models: list[str] | None = None) -> str:
    """
    Generate a HuggingFace catalog configuration string in YAML format.
    Similar to get_catalog_str() but for HuggingFace catalogs.

    Args:
        ids (list): List of model set identifiers that correspond to keys in MODELS dict
        excluded_models (list, optional): List of model names to exclude from the catalog

    Returns:
        str: YAML formatted catalog configuration with multiple catalog entries
    """
    catalog_entries = ""

    for source_id in ids:
        if source_id not in HF_MODELS:
            raise ValueError(f"Model ID '{source_id}' not found in MODELS dictionary")
        name = f"HuggingFace Source {source_id}"
        # Build catalog entry
        catalog_entry = f"""
- name: {name}
  id: huggingface_{source_id}
  type: "hf"
  enabled: true
  includedModels:
  {get_included_model_str(models=HF_MODELS[source_id])}"""

        # Add excludedModels if provided
        if excluded_models:
            catalog_entry += f"""
  excludedModels:
  {get_excluded_model_str(models=excluded_models)}"""

        catalog_entry += f"""
  labels:
  - {name}
"""
        catalog_entries += catalog_entry

    # Combine all catalog entries
    return f"""catalogs:
    {catalog_entries}
    """


def get_included_model_str(models: list[str]) -> str:
    included_models: str = ""
    for model_name in models:
        included_models += f"""
    - {model_name}
"""
    return included_models


def get_excluded_model_str(models: list[str]) -> str:
    excluded_models: str = ""
    for model_name in models:
        excluded_models += f"""
    - {model_name}
"""
    return excluded_models


def assert_source_error_state_message(
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    expected_error_message: str,
    source_id: str,
):
    results = execute_get_command(
        url=f"{model_catalog_rest_url[0]}sources",
        headers=model_registry_rest_headers,
    )["items"]
    # pick the relevant source first by id:
    matched_source = [result for result in results if result["id"] == source_id]
    assert matched_source, f"Matched expected source not found: {results}"
    assert matched_source[0]["status"] == "error"
    assert expected_error_message in matched_source[0]["error"], (
        f"Expected error: {expected_error_message} not found in {matched_source[0]['error']}"
    )


class TransientUnauthorizedError(Exception):
    """Exception for transient 401 Unauthorized errors that should be retried."""


def execute_get_call(
    url: str, headers: dict[str, str], verify: bool | str = False, params: dict[str, Any] | None = None
) -> requests.Response:
    LOGGER.info(f"Executing get call: {url}")
    if params:
        LOGGER.info(f"params: {params}")
    resp = requests.get(url=url, headers=headers, verify=verify, timeout=60, params=params)
    LOGGER.info(f"Encoded url from requests library: {resp.url}")
    if resp.status_code not in [200, 201]:
        # Raise custom exception for 401 errors that can be retried (OAuth/kube-rbac-proxy initialization)
        if resp.status_code == 401:
            raise TransientUnauthorizedError(f"Get call failed for resource: {url}, 401: {resp.text}")
        # Raise regular exception for other errors (400, 403, 404, etc.) that should fail immediately
        raise ResourceNotFoundError(f"Get call failed for resource: {url}, {resp.status_code}: {resp.text}")
    return resp


def execute_get_command(
    url: str, headers: dict[str, str], verify: bool | str = False, params: dict[str, Any] | None = None
) -> dict[Any, Any]:
    resp = execute_get_call(url=url, headers=headers, verify=verify, params=params)
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        LOGGER.error(f"Unable to parse {resp.text}")
        raise


@retry(wait_timeout=90, sleep=5, exceptions_dict={ResourceNotFoundError: [], TransientUnauthorizedError: []})
def wait_for_model_catalog_api(url: str, headers: dict[str, str], verify: bool | str = False) -> requests.Response:
    """
    Wait for model catalog API to be ready and fully initialized checks both /sources and /models endpoints
    to ensure OAuth/kube-rbac-proxy is fully initialized.
    """
    LOGGER.info(f"Waiting for model catalog API at {url}sources")
    execute_get_call(url=f"{url}sources", headers=headers, verify=verify)
    LOGGER.info(f"Verifying model catalog API readiness at {url}models")

    return execute_get_call(url=f"{url}models", headers=headers, verify=verify)


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
  createTimeSinceEpoch: \"{current_time - 10000!s}\"
  lastUpdateTimeSinceEpoch: \"{current_time!s}\"
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


def wait_for_model_catalog_pod_ready_after_deletion(
    client: DynamicClient, model_registry_namespace: str, consecutive_try: int = 6
) -> bool:
    model_catalog_pods = get_model_catalog_pod(
        client=client,
        model_registry_namespace=model_registry_namespace,
    )
    # We can wait for the pods to reflect updated catalog, however, deleting them ensures the updated config is
    # applied immediately.
    for pod in model_catalog_pods:
        pod.delete()
    # After the deletion, we need to wait for the pod to be spinned up and get to ready state.
    assert wait_for_model_catalog_pod_created(client=client, model_registry_namespace=model_registry_namespace)
    wait_for_pods_running(
        admin_client=client, namespace_name=model_registry_namespace, number_of_consecutive_checks=consecutive_try
    )
    return True


@retry(wait_timeout=30, sleep=5, exceptions_dict={PodNotFound: []})
def wait_for_model_catalog_pod_created(client: DynamicClient, model_registry_namespace: str) -> bool:
    pods = get_model_catalog_pod(client=client, model_registry_namespace=model_registry_namespace)
    if pods:
        return True
    raise PodNotFound("Model catalog pod not found")
