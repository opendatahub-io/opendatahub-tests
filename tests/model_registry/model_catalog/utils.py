import json
from typing import Any

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

import requests
from timeout_sampler import retry

from class_generator.parsers.explain_parser import ResourceNotFoundError
from ocp_resources.pod import Pod
from tests.model_registry.model_catalog.constants import (
    DEFAULT_CATALOG_NAME,
    DEFAULT_CATALOG_ID,
    CATALOG_TYPE,
    DEFAULT_CATALOG_FILE,
)
from tests.model_registry.utils import get_model_catalog_pod

LOGGER = get_logger(name=__name__)


def _execute_get_call(url: str, headers: dict[str, str], verify: bool | str = False) -> requests.Response:
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


def is_model_catalog_ready(client: DynamicClient, model_registry_namespace: str, consecutive_try: int = 3):
    model_catalog_pods = get_model_catalog_pod(client=client, model_registry_namespace=model_registry_namespace)
    # We can wait for the pods to reflect updated catalog, however, deleting them ensures the updated config is
    # applied immediately.
    for pod in model_catalog_pods:
        pod.delete()
    # After the deletion, we need to wait for the pod to be spinned up and get to ready state.
    for _ in range(consecutive_try):
        wait_for_model_catalog_update(client=client, model_registry_namespace=model_registry_namespace)


@retry(wait_timeout=30, sleep=5)
def wait_for_model_catalog_update(client: DynamicClient, model_registry_namespace: str):
    pods = get_model_catalog_pod(client=client, model_registry_namespace=model_registry_namespace)
    if pods:
        pods[0].wait_for_status(status=Pod.Status.RUNNING)
        return True
    return False


def validate_model_catalog_resource(kind: Any, admin_client: DynamicClient, namespace: str) -> None:
    resource = list(kind.get(namespace=namespace, label_selector="component=model-catalog", dyn_client=admin_client))
    assert resource
    assert len(resource) == 1, f"Unexpected number of {kind} resources found: {[res.name for res in resource]}"


def validate_default_catalog(default_catalog) -> None:
    assert default_catalog["name"] == DEFAULT_CATALOG_NAME
    assert default_catalog["id"] == DEFAULT_CATALOG_ID
    assert default_catalog["type"] == CATALOG_TYPE
    assert default_catalog["properties"].get("yamlCatalogPath") == DEFAULT_CATALOG_FILE
