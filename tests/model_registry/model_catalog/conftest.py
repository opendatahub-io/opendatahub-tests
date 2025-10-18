import random
from typing import Generator, Any

from simple_logger.logger import get_logger

import pytest
from kubernetes.dynamic import DynamicClient

from ocp_resources.config_map import ConfigMap
from ocp_resources.resource import ResourceEditor

from ocp_resources.service_account import ServiceAccount
from tests.model_registry.constants import CUSTOM_CATALOG_ID1
from tests.model_registry.model_catalog.constants import SAMPLE_MODEL_NAME3, DEFAULT_CATALOG_ID
from tests.model_registry.utils import (
    get_rest_headers,
    is_model_catalog_ready,
    wait_for_model_catalog_api,
    get_model_str,
    execute_get_command,
    get_custom_model_catalog_cm_data,
)
from utilities.infra import get_openshift_token, login_with_user_password, create_inference_token
from utilities.user_utils import UserTestSession

LOGGER = get_logger(name=__name__)


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
        yield
    else:
        with ResourceEditor(
            patches={
                catalog_config_map: get_custom_model_catalog_cm_data(
                    catalog_config_map=catalog_config_map, param=request.param
                )
            }
        ):
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
def randomly_picked_model_from_default_catalog(
    model_catalog_rest_url: list[str], user_token_for_api_calls: str
) -> dict[Any, Any]:
    url = f"{model_catalog_rest_url[0]}models?source={DEFAULT_CATALOG_ID}&pageSize=100"
    result = execute_get_command(
        url=url,
        headers=get_rest_headers(token=user_token_for_api_calls),
    )["items"]
    assert result, f"Expected Default models to be present. Actual: {result}"
    LOGGER.info(f"{len(result)} models found")
    return random.choice(seq=result)
