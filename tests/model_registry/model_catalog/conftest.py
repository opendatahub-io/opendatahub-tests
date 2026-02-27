import random
from collections.abc import Generator
from typing import Any

import pytest
import requests
import yaml
from kubernetes.dynamic import DynamicClient
from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor
from ocp_resources.route import Route
from ocp_resources.service_account import ServiceAccount
from simple_logger.logger import get_logger

from tests.model_registry.constants import (
    CUSTOM_CATALOG_ID1,
    DEFAULT_CUSTOM_MODEL_CATALOG,
)
from tests.model_registry.model_catalog.catalog_config.utils import get_models_from_database_by_source
from tests.model_registry.model_catalog.constants import (
    CATALOG_CONTAINER,
    DEFAULT_CATALOG_FILE,
    DEFAULT_CATALOGS,
    REDHAT_AI_CATALOG_ID,
    SAMPLE_MODEL_NAME3,
)
from tests.model_registry.model_catalog.utils import (
    execute_get_command,
    get_model_str,
    get_models_from_catalog_api,
    wait_for_model_catalog_api,
    wait_for_model_catalog_pod_ready_after_deletion,
)
from tests.model_registry.utils import (
    get_model_catalog_pod,
    get_mr_user_token,
    get_rest_headers,
)
from utilities.infra import create_inference_token, get_openshift_token, login_with_user_password

LOGGER = get_logger(name=__name__)


@pytest.fixture()
def sparse_override_catalog_source(
    request: pytest.FixtureRequest,
    admin_client,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[dict]:
    """
    Creates a sparse override for an existing default catalog source.

    Requires parameterization via request.param dict containing:
    - "id": catalog ID to override (required)
    - "field_name": name of the field to override (required)
    - "field_value": value for the field (required)
    """
    # Get fields from pytest param
    param = getattr(request, "param", None)
    assert param, "sparse_override_catalog_source requires request.param dict"

    catalog_id = param["id"]
    field_name = param["field_name"]
    field_value = param["field_value"]

    # Capture CURRENT catalog state from API before applying sparse override
    response = execute_get_command(url=f"{model_catalog_rest_url[0]}sources", headers=model_registry_rest_headers)
    items = response.get("items", [])
    original_catalog = next((item for item in items if item.get("id") == catalog_id), None)
    assert original_catalog is not None, f"Original catalog '{catalog_id}' not found in sources"

    # Create sparse override YAML with only id and the field to override
    catalog_override = {"id": catalog_id, field_name: field_value}
    sparse_catalog_yaml = yaml.dump(
        {"catalogs": [catalog_override]},
        default_flow_style=False,
    )

    # Write sparse override to custom ConfigMap
    sources_cm = ConfigMap(
        name=DEFAULT_CUSTOM_MODEL_CATALOG,
        client=admin_client,
        namespace=model_registry_namespace,
    )
    patches = {"data": {"sources.yaml": sparse_catalog_yaml}}

    with ResourceEditor(patches={sources_cm: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield {
            "catalog_id": catalog_id,
            "field_name": field_name,
            "field_value": field_value,
            "original_catalog": original_catalog,
        }

    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)


@pytest.fixture(scope="class")
def model_catalog_config_map(
    request: pytest.FixtureRequest, admin_client: DynamicClient, model_registry_namespace: str
) -> ConfigMap:
    """Parameterized fixture that takes a dict with configmap_name key and ensures it exists"""
    param = getattr(request, "param", {})
    configmap_name = param.get("configmap_name", "model-catalog-default-sources")
    return ConfigMap(name=configmap_name, client=admin_client, namespace=model_registry_namespace, ensure_exists=True)


@pytest.fixture(scope="class")
def updated_catalog_config_map(
    pytestconfig: pytest.Config,
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[ConfigMap]:
    if pytestconfig.option.post_upgrade or pytestconfig.option.pre_upgrade:
        yield catalog_config_map
    else:
        patches = {"data": {"sources.yaml": request.param["sources_yaml"]}}
        if "sample_yaml" in request.param:
            for key in request.param["sample_yaml"]:
                patches["data"][key] = request.param["sample_yaml"][key]

        with ResourceEditor(patches={catalog_config_map: patches}):
            wait_for_model_catalog_pod_ready_after_deletion(
                client=admin_client, model_registry_namespace=model_registry_namespace
            )
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
            yield catalog_config_map
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)


@pytest.fixture(scope="class")
def expected_catalog_values(request: pytest.FixtureRequest) -> dict[str, str]:
    return request.param


@pytest.fixture(scope="class")
def is_huggingface(request: pytest.FixtureRequest) -> dict[str, str]:
    return request.param


@pytest.fixture(scope="function")
def update_configmap_data_add_model(
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[ConfigMap]:
    patches = catalog_config_map.instance.to_dict()
    patches["data"][f"{CUSTOM_CATALOG_ID1.replace('_', '-')}.yaml"] += get_model_str(model=SAMPLE_MODEL_NAME3)
    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield catalog_config_map
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)


@pytest.fixture(scope="class")
def user_token_for_api_calls(
    is_byoidc: bool,
    admin_client: DynamicClient,
    request: pytest.FixtureRequest,
    original_user: str,
    api_server_url: str,
    user_credentials_rbac: dict[str, str],
    service_account: ServiceAccount,
    model_catalog_rest_url: list[str],
) -> Generator[str]:
    param = getattr(request, "param", {})
    user = param.get("user_type", "admin")
    LOGGER.info("User used: %s", user)

    token = None
    if user == "admin":
        LOGGER.info("Logging in as admin user")
        token = get_openshift_token()
    elif user == "test":
        if not is_byoidc:
            login_with_user_password(
                api_address=api_server_url,
                user=user_credentials_rbac["username"],
                password=user_credentials_rbac["password"],
            )
            token = get_openshift_token()
        else:
            token = get_mr_user_token(admin_client=admin_client, user_credentials_rbac=user_credentials_rbac)
    elif user == "sa_user":
        token = create_inference_token(service_account)
        # retries on 401 errors for OAuth/kube-rbac-proxy initialization
        headers = get_rest_headers(token=token)
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=headers)
    else:
        raise RuntimeError(f"Unknown user type: {user}")

    yield token

    # Cleanup: log back in as original user if needed
    if user == "test" and not is_byoidc:
        LOGGER.info(f"Logging in as {original_user}")
        login_with_user_password(
            api_address=api_server_url,
            user=original_user,
        )


@pytest.fixture(scope="function")
def randomly_picked_model_from_catalog_api_by_source(
    model_catalog_rest_url: list[str],
    user_token_for_api_calls: str,
    model_registry_rest_headers: dict[str, str],
    request: pytest.FixtureRequest,
) -> tuple[dict[Any, Any], str, str]:
    """
    Pick a random model from a specific catalog if a model name is not provided. If model name is provided, verify
    that it exists and is associated with a given catalog and return the same.

    Supports parameterized headers via 'header_type':
    - 'user_token': Uses user_token_for_api_calls (default for user-specific tests)
    - 'registry': Uses model_registry_rest_headers (for catalog/registry tests)
    - 'model_name': Name of the model

    Accepts 'catalog_id' or 'source' (alias) to specify the catalog. Accepts 'model_name' to specify the model to
    look for.
    """
    param = getattr(request, "param", {})
    # Support both 'catalog_id' and 'source' for backward compatibility
    catalog_id = param.get("catalog_id") or param.get("source", REDHAT_AI_CATALOG_ID)
    header_type = param.get("header_type", "user_token")
    model_name = param.get("model_name")
    random_model = None
    # Select headers based on header_type
    if header_type == "registry":
        headers = model_registry_rest_headers
    else:
        headers = get_rest_headers(token=user_token_for_api_calls)

    if not model_name:
        LOGGER.info(f"Picking random model from catalog: {catalog_id} with header_type: {header_type}")
        models_response = execute_get_command(
            url=f"{model_catalog_rest_url[0]}models?source={catalog_id}&pageSize=100",
            headers=headers,
        )
        models = models_response.get("items", [])
        assert models, f"No models found for catalog: {catalog_id}"
        LOGGER.info(f"{len(models)} models found in catalog {catalog_id}")
        random_model = random.choice(seq=models)
        model_name = random_model.get("name")
        assert model_name, "Model name not found in random model"
        assert random_model.get("source_id") == catalog_id, f"Catalog ID (source_id) mismatch for model {model_name}"
        LOGGER.info(f"Testing model '{model_name}' from catalog '{catalog_id}'")
    else:
        LOGGER.info(f"Looking for pre-selected model: {model_name} from catalog: {catalog_id}")
        # check if the model exists:
        random_model = execute_get_command(
            url=f"{model_catalog_rest_url[0]}sources/{catalog_id}/models/{model_name}",
            headers=headers,
        )
        assert random_model["source_id"] == catalog_id, f"Catalog ID (source_id) mismatch for model {model_name}"
        LOGGER.info(f"Using model '{model_name}' from catalog '{catalog_id}'")
    return random_model, model_name, catalog_id


@pytest.fixture(scope="class")
def default_model_catalog_yaml_content(
    request: pytest.FixtureRequest, admin_client: DynamicClient, model_registry_namespace: str
) -> dict[Any, Any]:
    """
    Fetch and parse catalog YAML from the catalog pod.

    Defaults to DEFAULT_CATALOG_FILE if not parameterized.
    Use with @pytest.mark.parametrize indirect parameter to specify a different catalog:

    Args:
        request.param: Optional catalog ID, if not provided, uses DEFAULT_CATALOG_FILE.

    Returns:
        Parsed YAML content as dictionary
    """
    # If parameterized, get the catalog file path from the catalog ID
    # Otherwise, use DEFAULT_CATALOG_FILE
    catalog_id = getattr(request, "param", None)
    if catalog_id:
        catalog_file_path = DEFAULT_CATALOGS[catalog_id]["properties"]["yamlCatalogPath"]
    else:
        catalog_file_path = DEFAULT_CATALOG_FILE

    model_catalog_pod = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)[0]
    return yaml.safe_load(model_catalog_pod.execute(command=["cat", catalog_file_path], container=CATALOG_CONTAINER))


@pytest.fixture(scope="class")
def default_catalog_api_response(
    model_catalog_rest_url: list[str], model_registry_rest_headers: dict[str, str]
) -> dict[Any, Any]:
    """Fetch all models from default catalog API (used for data validation tests)"""
    return execute_get_command(
        url=f"{model_catalog_rest_url[0]}models?source={REDHAT_AI_CATALOG_ID}&pageSize=100",
        headers=model_registry_rest_headers,
    )


@pytest.fixture(scope="class")
def catalog_openapi_schema() -> dict[Any, Any]:
    """Fetch and cache the catalog OpenAPI schema (fetched once per class)"""
    OPENAPI_SCHEMA_URL = "https://raw.githubusercontent.com/kubeflow/model-registry/main/api/openapi/catalog.yaml"
    response = requests.get(OPENAPI_SCHEMA_URL, timeout=10)
    response.raise_for_status()
    return yaml.safe_load(response.text)


@pytest.fixture
def models_from_filter_query(
    request,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> list[str]:
    """
    Fixture that runs get_models_from_catalog_api with the given filter_query,
    asserts that models are returned, and returns list of model names.
    """
    filter_query = request.param

    models = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        additional_params=f"&filterQuery={filter_query}",
    )["items"]

    assert models, f"No models returned from filter query: {filter_query}"

    model_names = [model["name"] for model in models]
    LOGGER.info(f"Filter query '{filter_query}' returned {len(model_names)} models: {', '.join(model_names)}")

    return model_names


@pytest.fixture()
def labels_configmap_patch(
    admin_client: DynamicClient,
    model_registry_namespace: str,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[dict[str, Any]]:
    # Get the editable ConfigMap
    sources_cm = ConfigMap(name=DEFAULT_CUSTOM_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace)

    # Parse current data and add test label
    current_data = yaml.safe_load(sources_cm.instance.data["sources.yaml"])

    new_label = {
        "name": "test-dynamic",
        "displayName": "Dynamic Test Label",
        "description": "A label added during test execution",
    }

    if "labels" not in current_data:
        current_data["labels"] = []
    current_data["labels"].append(new_label)

    patches = {"data": {"sources.yaml": yaml.dump(current_data, default_flow_style=False)}}

    with ResourceEditor(patches={sources_cm: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield patches
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)


@pytest.fixture()
def skip_on_huggingface_source(is_huggingface: bool) -> None:
    if is_huggingface:
        pytest.skip(reason="Huggingface models does not support artifacts endpoints")


@pytest.fixture()
def updated_catalog_config_map_scope_function(
    pytestconfig: pytest.Config,
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[ConfigMap]:
    patches = {"data": {"sources.yaml": request.param}}
    with ResourceEditor(patches={catalog_config_map: patches}):
        wait_for_model_catalog_pod_ready_after_deletion(
            client=admin_client, model_registry_namespace=model_registry_namespace
        )
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield catalog_config_map
    wait_for_model_catalog_pod_ready_after_deletion(
        client=admin_client, model_registry_namespace=model_registry_namespace
    )
    wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)


@pytest.fixture(scope="class")
def catalog_config_map(admin_client: DynamicClient, model_registry_namespace: str) -> ConfigMap:
    return ConfigMap(name=DEFAULT_CUSTOM_MODEL_CATALOG, client=admin_client, namespace=model_registry_namespace)


@pytest.fixture(scope="class")
def model_catalog_routes(admin_client: DynamicClient, model_registry_namespace: str) -> list[Route]:
    return list(
        Route.get(namespace=model_registry_namespace, label_selector="component=model-catalog", client=admin_client)
    )


@pytest.fixture(scope="class")
def model_catalog_rest_url(model_registry_namespace: str, model_catalog_routes: list[Route]) -> list[str]:
    assert model_catalog_routes, f"Model catalog routes does not exist in {model_registry_namespace}"
    route_urls = [
        f"https://{route.instance.spec.host}:443/api/model_catalog/v1alpha1/" for route in model_catalog_routes
    ]
    assert route_urls, (
        "Model catalog routes information could not be found from "
        f"routes:{[route.name for route in model_catalog_routes]}"
    )
    return route_urls


@pytest.fixture(scope="function")
def baseline_redhat_ai_models(
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
    model_registry_namespace: str,
) -> dict[str, set[str] | int]:
    """
    fixture providing baseline model data for redhat_ai_models source.

    Returns:
        Dictionary with 'api_models', 'db_models', and 'count' keys
    """

    api_response = get_models_from_catalog_api(
        model_catalog_rest_url=model_catalog_rest_url,
        model_registry_rest_headers=model_registry_rest_headers,
        source_label="Red Hat AI",
    )
    api_models = {model["name"] for model in api_response.get("items", [])}

    db_models = get_models_from_database_by_source(
        admin_client=admin_client, source_id=REDHAT_AI_CATALOG_ID, namespace=model_registry_namespace
    )

    return {"api_models": api_models, "db_models": db_models, "count": len(api_models)}
