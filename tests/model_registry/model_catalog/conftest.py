import random
from typing import Generator, Any
import requests

from simple_logger.logger import get_logger
import yaml
import pytest
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor

from ocp_resources.route import Route
from ocp_resources.service_account import ServiceAccount
from tests.model_registry.model_catalog.constants import (
    SAMPLE_MODEL_NAME3,
    DEFAULT_CATALOG_FILE,
    CATALOG_CONTAINER,
    REDHAT_AI_CATALOG_ID,
)
from tests.model_registry.constants import CUSTOM_CATALOG_ID1
from tests.model_registry.utils import (
    get_rest_headers,
    is_model_catalog_ready,
    get_model_catalog_pod,
    wait_for_model_catalog_api,
    execute_get_command,
    get_model_str,
)
from utilities.infra import get_openshift_token, login_with_user_password, create_inference_token
from utilities.user_utils import UserTestSession


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
def model_catalog_routes(admin_client: DynamicClient, model_registry_namespace: str) -> list[Route]:
    return list(
        Route.get(namespace=model_registry_namespace, label_selector="component=model-catalog", dyn_client=admin_client)
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


@pytest.fixture(scope="class")
def updated_catalog_config_map(
    request: pytest.FixtureRequest,
    catalog_config_map: ConfigMap,
    model_registry_namespace: str,
    admin_client: DynamicClient,
    model_catalog_rest_url: list[str],
    model_registry_rest_headers: dict[str, str],
) -> Generator[ConfigMap, None, None]:
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
    request: pytest.FixtureRequest,
    original_user: str,
    api_server_url: str,
    test_idp_user: UserTestSession,
    service_account: ServiceAccount,
) -> Generator[str, None, None]:
    param = getattr(request, "param", {})
    user = param.get("user_type", "admin")
    LOGGER.info("User used: %s", user)
    if user == "admin":
        LOGGER.info("Logging in as admin user")
        yield get_openshift_token()
    elif user == "test":
        login_with_user_password(
            api_address=api_server_url,
            user=test_idp_user.username,
            password=test_idp_user.password,
        )
        yield get_openshift_token()
        LOGGER.info(f"Logging in as {original_user}")
        login_with_user_password(
            api_address=api_server_url,
            user=original_user,
        )
    elif user == "sa_user":
        yield create_inference_token(service_account)
    else:
        raise RuntimeError(f"Unknown user type: {user}")


@pytest.fixture(scope="class")
def randomly_picked_model(
    model_catalog_rest_url: list[str], user_token_for_api_calls: str, request: pytest.FixtureRequest
) -> dict[Any, Any]:
    """Pick a random model"""
    param = getattr(request, "param", {})
    source = param.get("source", REDHAT_AI_CATALOG_ID)
    LOGGER.info(f"Picking random model from {source}")
    url = f"{model_catalog_rest_url[0]}models?source={source}&pageSize=100"
    result = execute_get_command(
        url=url,
        headers=get_rest_headers(token=user_token_for_api_calls),
    )["items"]
    assert result, f"Expected Default models to be present. Actual: {result}"
    LOGGER.info(f"{len(result)} models found")
    return random.choice(seq=result)


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
