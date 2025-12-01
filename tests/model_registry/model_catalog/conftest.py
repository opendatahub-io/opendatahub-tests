import random
from typing import Generator, Any
import requests

from simple_logger.logger import get_logger
import yaml
import pytest
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor

from ocp_resources.service_account import ServiceAccount
from tests.model_registry.model_catalog.constants import (
    SAMPLE_MODEL_NAME3,
    DEFAULT_CATALOG_FILE,
    CATALOG_CONTAINER,
    REDHAT_AI_CATALOG_ID,
)
from tests.model_registry.model_catalog.utils import get_models_from_catalog_api
from tests.model_registry.constants import CUSTOM_CATALOG_ID1
from tests.model_registry.utils import (
    get_rest_headers,
    is_model_catalog_ready,
    get_model_catalog_pod,
    wait_for_model_catalog_api,
    execute_get_command,
    get_model_str,
    get_mr_user_token,
)
from utilities.infra import get_openshift_token, create_inference_token, login_with_user_password


LOGGER = get_logger(name=__name__)


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
) -> Generator[ConfigMap, None, None]:
    if pytestconfig.option.post_upgrade or pytestconfig.option.pre_upgrade:
        yield catalog_config_map
    else:
        patches = {"data": {"sources.yaml": request.param["sources_yaml"]}}
        if "sample_yaml" in request.param:
            for key in request.param["sample_yaml"]:
                patches["data"][key] = request.param["sample_yaml"][key]

        with ResourceEditor(patches={catalog_config_map: patches}):
            is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
            wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
            yield catalog_config_map
        is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)


@pytest.fixture(scope="class")
def expected_catalog_values(request: pytest.FixtureRequest) -> dict[str, str]:
    return request.param


@pytest.fixture(scope="function")
def update_configmap_data_add_model(
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[ConfigMap, None, None]:
    patches = catalog_config_map.instance.to_dict()
    patches["data"][f"{CUSTOM_CATALOG_ID1.replace('_', '-')}.yaml"] += get_model_str(model=SAMPLE_MODEL_NAME3)
    with ResourceEditor(patches={catalog_config_map: patches}):
        is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)
        wait_for_model_catalog_api(url=model_catalog_rest_url[0], headers=model_registry_rest_headers)
        yield catalog_config_map
    is_model_catalog_ready(client=admin_client, model_registry_namespace=model_registry_namespace)


@pytest.fixture(scope="class")
def user_token_for_api_calls(
    is_byoidc: bool,
    admin_client: DynamicClient,
    request: pytest.FixtureRequest,
    original_user: str,
    api_server_url: str,
    user_credentials_rbac: dict[str, str],
    service_account: ServiceAccount,
) -> Generator[str, None, None]:
    param = getattr(request, "param", {})
    user = param.get("user_type", "admin")
    LOGGER.info("User used: %s", user)
    if user == "admin":
        LOGGER.info("Logging in as admin user")
        yield get_openshift_token()
    elif user == "test":
        if not is_byoidc:
            login_with_user_password(
                api_address=api_server_url,
                user=user_credentials_rbac["username"],
                password=user_credentials_rbac["password"],
            )
            yield get_openshift_token()
            LOGGER.info(f"Logging in as {original_user}")
            login_with_user_password(
                api_address=api_server_url,
                user=original_user,
            )
        else:
            yield get_mr_user_token(admin_client=admin_client, user_credentials_rbac=user_credentials_rbac)
    elif user == "sa_user":
        yield create_inference_token(service_account)
    else:
        raise RuntimeError(f"Unknown user type: {user}")


@pytest.fixture(scope="function")
def randomly_picked_model_from_catalog_api_by_source(
    model_catalog_rest_url: list[str],
    user_token_for_api_calls: str,
    model_registry_rest_headers: dict[str, str],
    request: pytest.FixtureRequest,
) -> tuple[dict[Any, Any], str, str]:
    """Pick a random model from a specific catalog (function-scoped for test isolation)

    Supports parameterized headers via 'header_type':
    - 'user_token': Uses user_token_for_api_calls (default for user-specific tests)
    - 'registry': Uses model_registry_rest_headers (for catalog/registry tests)

    Accepts 'catalog_id' or 'source' (alias) to specify the catalog.
    """
    param = getattr(request, "param", {})
    # Support both 'catalog_id' and 'source' for backward compatibility
    catalog_id = param.get("catalog_id") or param.get("source", REDHAT_AI_CATALOG_ID)
    header_type = param.get("header_type", "user_token")

    # Select headers based on header_type
    if header_type == "registry":
        headers = model_registry_rest_headers
    else:
        headers = get_rest_headers(token=user_token_for_api_calls)

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

    return random_model, model_name, catalog_id


@pytest.fixture(scope="class")
def default_model_catalog_yaml_content(admin_client: DynamicClient, model_registry_namespace: str) -> dict[Any, Any]:
    model_catalog_pod = get_model_catalog_pod(client=admin_client, model_registry_namespace=model_registry_namespace)[0]
    return yaml.safe_load(model_catalog_pod.execute(command=["cat", DEFAULT_CATALOG_FILE], container=CATALOG_CONTAINER))


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
