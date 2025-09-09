import json
from typing import Any

from kubernetes.dynamic import DynamicClient
from simple_logger.logger import get_logger

import requests
from timeout_sampler import retry

from ocp_resources.pod import Pod
from tests.model_registry.model_catalog.constants import (
    DEFAULT_CATALOG_NAME,
    DEFAULT_CATALOG_ID,
    CATALOG_TYPE,
    DEFAULT_CATALOG_FILE,
)
from tests.model_registry.utils import get_model_catalog_pod, wait_for_pods_running

LOGGER = get_logger(name=__name__)


class ResourceNotFoundError(Exception):
    pass


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


def validate_model_catalog_resource(kind: Any, admin_client: DynamicClient, namespace: str) -> None:
    resource = list(kind.get(namespace=namespace, label_selector="component=model-catalog", dyn_client=admin_client))
    assert resource
    assert len(resource) == 1, f"Unexpected number of {kind} resources found: {[res.name for res in resource]}"


def validate_default_catalog(default_catalog) -> None:
    assert default_catalog["name"] == DEFAULT_CATALOG_NAME
    assert default_catalog["id"] == DEFAULT_CATALOG_ID
    assert default_catalog["type"] == CATALOG_TYPE
    assert default_catalog["properties"].get("yamlCatalogPath") == DEFAULT_CATALOG_FILE


def get_catalog_str(ids: list[str]) -> str:
    catalog_str: str = ""
    for id in ids:
        catalog_str += f"""
- name: Sample Catalog
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
"""
